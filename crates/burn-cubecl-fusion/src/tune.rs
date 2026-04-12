use crate::CubeFusionHandle;
use burn_fusion::stream::{Context, ContextOwned};
use burn_ir::{HandleContainer, TensorId, TensorIr};
use cubecl::Runtime;
use hashbrown::HashMap;
use std::{cell::UnsafeCell, sync::Arc};

/// Raw pointer wrapper that is `Send` when the pointee is `Send`.
#[repr(transparent)]
pub(crate) struct SendPtr<T: ?Sized>(*mut T);

// SAFETY: the caller upholds the non-aliasing invariant; when `T: Send`,
// moving the pointer across threads is sound.
unsafe impl<T: ?Sized + Send> Send for SendPtr<T> {}

/// Thread-safe shared storage for newly created output handles.
///
/// # Safety
///
/// All access is sequential on the same thread during autotuning; the `Sync`
/// impl is required for [`Arc`] but concurrent access never occurs.
pub(crate) struct SharedNewHandles<R: Runtime>(UnsafeCell<Vec<(TensorId, CubeFusionHandle<R>)>>);

unsafe impl<R: Runtime> Sync for SharedNewHandles<R> {}

impl<R: Runtime> SharedNewHandles<R> {
    fn new() -> Self {
        Self(UnsafeCell::new(Vec::new()))
    }

    /// SAFETY: caller must ensure no concurrent access (guaranteed by
    /// sequential execution).
    unsafe fn push(&self, id: TensorId, handle: CubeFusionHandle<R>) {
        unsafe { &mut *self.0.get() }.push((id, handle));
    }

    /// SAFETY: caller must ensure no concurrent access and that all writers
    /// have finished.
    unsafe fn take(&self) -> Vec<(TensorId, CubeFusionHandle<R>)> {
        std::mem::take(unsafe { &mut *self.0.get() })
    }

    /// SAFETY: caller must ensure no concurrent access.
    unsafe fn clear(&self) {
        unsafe { &mut *self.0.get() }.clear();
    }
}

/// Fusion input for autotuning.
///
/// The [`Original`](Self::Original) variant wraps the real [`Context`] via a
/// raw pointer (valid only inside [`LocalTuner::execute`](cubecl::tune::LocalTuner::execute)).
/// The [`Fork`](Self::Fork) variant owns a forked context used for benchmark runs.
///
/// # Sequential execution and rollback
///
/// All tune functions (fused and fallback) run on the same thread, sequentially.
/// When a fused optimization fails on the [`Original`](Self::Original) path, the
/// optimization is responsible for rolling back any modifications it made
/// (e.g. restoring input handle strides). For the [`Fork`](Self::Fork) path,
/// failures are discarded: the fork is dropped and the original context is
/// untouched.
///
/// # Output handle persistence (clone contract)
///
/// When an [`Original`](Self::Original) is cloned, the resulting [`Fork`](Self::Fork)
/// shares a [`SharedNewHandles`] with it. Forked executions collect newly
/// produced output handles on drop. If the original is then dropped without
/// [`execute`](Self::execute) being called, the collected handles are persisted
/// to the real context. This upholds the [`Clone`] contract: outputs produced
/// by a forked execution are visible in the original context even when the
/// original path was never taken.
pub(crate) enum TuneInput<R: Runtime, O> {
    Original {
        /// Valid for the duration of `LocalTuner::execute`. Deliberately never
        /// stored in [`Fork`](Self::Fork) — on wasm a benchmark may outlive
        /// the execute call and the pointer would dangle.
        ptr: SendPtr<Context<'static, CubeFusionHandle<R>>>,
        new_handles: Arc<SharedNewHandles<R>>,
        /// If false at drop time, forked output handles must be persisted.
        executed: bool,
        optimization: Arc<O>,
    },
    Fork {
        context: Box<ContextOwned<CubeFusionHandle<R>>>,
        new_handles: Arc<SharedNewHandles<R>>,
        /// Snapshot of handle IDs at fork time, so drop can identify which
        /// outputs were produced by the benchmark run.
        original_ids: Vec<TensorId>,
        optimization: Arc<O>,
    },
}

impl<R: Runtime, O> TuneInput<R, O> {
    /// Create a new autotune input from a [`Context`] and an optimization.
    pub(crate) fn new(context: &mut Context<CubeFusionHandle<R>>, optimization: O) -> Self {
        // Erase the context lifetime so `TuneInput` can be `'static`, as
        // required by `LocalTuner::execute`.
        #[allow(clippy::unnecessary_cast)]
        let ptr = core::ptr::from_mut(context) as *mut Context<'static, _>;

        Self::Original {
            ptr: SendPtr(ptr),
            new_handles: Arc::new(SharedNewHandles::new()),
            executed: false,
            optimization: Arc::new(optimization),
        }
    }

    /// Read-only access to the tensor map for autotune key generation.
    pub(crate) fn tensors(&self) -> &HashMap<TensorId, TensorIr> {
        match self {
            // SAFETY: shared borrow only; `&self` excludes any `&mut`, which
            // only comes from `execute(self)` consuming `self`.
            Self::Original { ptr, .. } => unsafe { &*ptr.0 }.tensors,
            Self::Fork { context, .. } => context.tensors(),
        }
    }

    /// Read-only access to the handle container for autotune key generation.
    pub(crate) fn handles(&self) -> &HandleContainer<CubeFusionHandle<R>> {
        match self {
            // SAFETY: same as `tensors`.
            Self::Original { ptr, .. } => unsafe { &*ptr.0 }.handles,
            Self::Fork { context, .. } => context.handles(),
        }
    }

    /// Retrieve the optimization for the current input.
    pub(crate) fn optimization(&self) -> &O {
        match self {
            Self::Original { optimization, .. } | Self::Fork { optimization, .. } => optimization,
        }
    }

    /// Consume the input and run a closure with mutable access to the
    /// [`Context`] and the optimization. Consuming `self` is what makes
    /// the `&mut Context` sound: no other borrow can exist once it's gone.
    pub(crate) fn execute<F, T>(mut self, f: F) -> T
    where
        F: FnOnce(&mut Context<'_, CubeFusionHandle<R>>, &O) -> T,
    {
        match &mut self {
            Self::Original {
                ptr,
                executed,
                optimization,
                ..
            } => {
                // Suppresses drop-time persistence — the closure runs on the
                // real context directly, so there's nothing to drain.
                *executed = true;
                // SAFETY: `self` is consumed, no other borrow via this
                // `TuneInput` can exist, and the ptr is live (still inside
                // `LocalTuner::execute`).
                f(unsafe { &mut *ptr.0 }, optimization)
            }
            Self::Fork {
                context,
                optimization,
                ..
            } => f(&mut context.as_context(), optimization),
        }
    }
}

impl<R: Runtime, O> Clone for TuneInput<R, O> {
    fn clone(&self) -> Self {
        // Cloning always produces a `Fork` — this drives benchmark isolation
        // and lets the clone outlive `LocalTuner::execute` on wasm.
        let (forked, new_handles, optimization) = match self {
            Self::Original {
                ptr,
                new_handles,
                optimization,
                ..
            } => {
                // SAFETY: shared borrow only; `&self` excludes `execute`.
                let ctx = unsafe { &*ptr.0 };
                (ctx.fork(), new_handles.clone(), optimization.clone())
            }
            Self::Fork {
                context,
                new_handles,
                optimization,
                ..
            } => (context.fork(), new_handles.clone(), optimization.clone()),
        };
        let original_ids = forked.handles().handle_ids().copied().collect();
        // Each new fork resets the handles saved by the previous execution,
        // so no memory leak is created by keeping discarded handles.
        // SAFETY: sequential execution, no concurrent access.
        unsafe { new_handles.clear() };
        Self::Fork {
            context: Box::new(forked),
            new_handles,
            original_ids,
            optimization,
        }
    }
}

impl<R: Runtime, O> Drop for TuneInput<R, O> {
    fn drop(&mut self) {
        match self {
            Self::Original {
                ptr,
                new_handles,
                executed,
                ..
            } => {
                if *executed {
                    return;
                }
                // The original was never executed; persist output handles
                // that were produced by forked executions.
                // SAFETY: still inside `LocalTuner::execute`; all forks are
                // dropped (sequential execution), so no concurrent writers.
                let context = unsafe { &mut *ptr.0 };
                for (id, handle) in unsafe { new_handles.take() } {
                    context.handles.register_handle(id, handle);
                }
            }
            Self::Fork {
                context,
                new_handles,
                original_ids,
                ..
            } => {
                let fork_handles = context.handles();
                for id in fork_handles.handle_ids() {
                    if !original_ids.contains(id)
                        && let Some(handle) = fork_handles.get_handle_ref(id)
                    {
                        // SAFETY: sequential execution, no concurrent access.
                        unsafe { new_handles.push(*id, handle.clone()) };
                    }
                }
            }
        }
    }
}
