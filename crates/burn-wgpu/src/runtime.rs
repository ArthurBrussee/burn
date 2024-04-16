use crate::{
    compiler::wgsl,
    compute::{WgpuServer, WgpuStorage},
    FloatElement, GraphicsApi, IntElement, WgpuDevice,
};
use alloc::sync::Arc;
use burn_common::stub::RwLock;
use burn_compute::{
    channel::MutexComputeChannel,
    client::ComputeClient,
    memory_management::{DeallocStrategy, SimpleMemoryManagement, SliceStrategy},
    tune::Tuner,
    ComputeRuntime,
};
use burn_jit::Runtime;
use std::marker::PhantomData;
use wgpu::{AdapterInfo, DeviceDescriptor};

/// Runtime that uses the [wgpu] crate with the wgsl compiler.
///
/// The [graphics api](GraphicsApi), the [float element](FloatElement) and the
/// [int element](IntElement) types are passed as generic.
#[derive(Debug)]
pub struct WgpuRuntime<G: GraphicsApi, F: FloatElement, I: IntElement> {
    _g: PhantomData<G>,
    _f: PhantomData<F>,
    _i: PhantomData<I>,
}

/// The compute instance is shared across all [wgpu runtimes](WgpuRuntime).
static RUNTIME: ComputeRuntime<WgpuDevice, Server, MutexComputeChannel<Server>> =
    ComputeRuntime::new();

type Server = WgpuServer<SimpleMemoryManagement<WgpuStorage>>;

impl<G: GraphicsApi, F: FloatElement, I: IntElement> Runtime for WgpuRuntime<G, F, I> {
    type FullPrecisionRuntime = WgpuRuntime<G, f32, i32>;
    type Compiler = wgsl::WgslCompiler<F, I>;
    type Server = WgpuServer<SimpleMemoryManagement<WgpuStorage>>;

    type Channel = MutexComputeChannel<WgpuServer<SimpleMemoryManagement<WgpuStorage>>>;
    type Device = WgpuDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self::Server, Self::Channel> {
        RUNTIME.client(device, move || {
            let (adapter, device_wgpu, queue) = pollster::block_on(create_wgpu_setup::<G>(&device));
            create_client(adapter, device_wgpu, queue, RuntimeOptions::default())
        })
    }

    fn name() -> &'static str {
        "wgpu"
    }
}

/// The values that control how a WGPU Runtime will perform its calculations.
pub struct RuntimeOptions {
    /// How the buffers are deallocated.
    pub dealloc_strategy: DeallocStrategy,
    /// Control the slicing strategy.
    pub slice_strategy: SliceStrategy,
    /// Control the amount of compute tasks to be aggregated into a single GPU command.
    pub max_tasks: usize,
}

impl Default for RuntimeOptions {
    fn default() -> Self {
        let max_tasks = match std::env::var("BURN_WGPU_MAX_TASKS") {
            Ok(value) => value
                .parse::<usize>()
                .expect("BURN_WGPU_MAX_TASKS should be a positive integer."),
            Err(_) => 64, // 64 tasks by default
        };

        Self {
            dealloc_strategy: DeallocStrategy::new_period_tick(max_tasks * 2),
            slice_strategy: SliceStrategy::Ratio(0.8),
            max_tasks,
        }
    }
}

pub fn init_existing_device(
    custom_id: usize,
    adapter: Arc<wgpu::Adapter>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    options: RuntimeOptions,
) -> WgpuDevice {
    let client = create_client(adapter, device, queue, options);
    let device = WgpuDevice::Existing(custom_id);
    RUNTIME.register(&device, client);
    device
}

/// Init the client sync, useful to configure the runtime options.
pub fn init_sync<G: GraphicsApi>(device: &WgpuDevice, options: RuntimeOptions) {
    let (adapter, device_wgpu, queue) = pollster::block_on(create_wgpu_setup::<G>(device));
    let client = create_client(adapter, device_wgpu, queue, options);
    RUNTIME.register(device, client)
}

/// Init the client async, necessary for wasm.
pub async fn init_async<G: GraphicsApi>(device: &WgpuDevice, options: RuntimeOptions) {
    let (adapter, device_wgpu, queue) = create_wgpu_setup::<G>(device).await;
    let client = create_client(adapter, device_wgpu, queue, options);
    RUNTIME.register(device, client)
}

async fn create_wgpu_setup<G: GraphicsApi>(
    device: &WgpuDevice,
) -> (Arc<wgpu::Adapter>, Arc<wgpu::Device>, Arc<wgpu::Queue>) {
    let (device_wgpu, queue, adapter) = select_device::<G>(device).await;
    log::info!(
        "Created wgpu compute server on device {:?} => {:?}",
        device,
        adapter.get_info()
    );
    (Arc::new(adapter), Arc::new(device_wgpu), Arc::new(queue))
}

fn create_client(
    adapter: Arc<wgpu::Adapter>,
    device_wgpu: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    options: RuntimeOptions,
) -> ComputeClient<
    WgpuServer<SimpleMemoryManagement<WgpuStorage>>,
    MutexComputeChannel<WgpuServer<SimpleMemoryManagement<WgpuStorage>>>,
> {
    let storage = WgpuStorage::new(device_wgpu.clone());
    let memory_management =
        SimpleMemoryManagement::new(storage, options.dealloc_strategy, options.slice_strategy);
    let server = WgpuServer::new(memory_management, device_wgpu, queue, options.max_tasks);
    let channel = MutexComputeChannel::new(server);
    let tuner_device_id = tuner_device_id(adapter.get_info());
    ComputeClient::new(channel, Arc::new(RwLock::new(Tuner::new(&tuner_device_id))))
}

/// Select the wgpu device and queue based on the provided [device](WgpuDevice).
pub async fn select_device<G: GraphicsApi>(
    device: &WgpuDevice,
) -> (wgpu::Device, wgpu::Queue, wgpu::Adapter) {
    #[cfg(target_family = "wasm")]
    let adapter = select_adapter::<G>(device).await;

    #[cfg(not(target_family = "wasm"))]
    let adapter = select_adapter::<G>(device);

    let limits = adapter.limits();

    let (device, queue) = adapter
        .request_device(
            &DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits,
            },
            None,
        )
        .await
        .map_err(|err| {
            format!(
                "Unable to request the device with the adapter {:?}, err {:?}",
                adapter.get_info(),
                err
            )
        })
        .unwrap();

    (device, queue, adapter)
}

fn tuner_device_id(info: AdapterInfo) -> String {
    format!("wgpu-{}-{}", info.device, info.backend.to_str())
}

#[cfg(target_family = "wasm")]
async fn select_adapter<G: GraphicsApi>(_device: &WgpuDevice) -> wgpu::Adapter {
    let instance = wgpu::Instance::default();

    instance
        .request_adapter(&wgpu::RequestAdapterOptionsBase::default())
        .await
        .unwrap()
}

#[cfg(not(target_family = "wasm"))]
fn select_adapter<G: GraphicsApi>(device: &WgpuDevice) -> wgpu::Adapter {
    use wgpu::DeviceType;

    let instance = wgpu::Instance::default();
    let mut adapters_other = Vec::new();
    let mut adapters = Vec::new();

    if matches!(device, WgpuDevice::Existing(_)) {
        panic!("Cannot automatically create a client for an existing device! Please use init_existing_device instead.")
    }

    instance
        .enumerate_adapters(G::backend().into())
        .for_each(|adapter| {
            let device_type = adapter.get_info().device_type;

            if let DeviceType::Other = device_type {
                adapters_other.push(adapter);
                return;
            }

            let is_same_type = match device {
                WgpuDevice::DiscreteGpu(_) => device_type == DeviceType::DiscreteGpu,
                WgpuDevice::IntegratedGpu(_) => device_type == DeviceType::IntegratedGpu,
                WgpuDevice::VirtualGpu(_) => device_type == DeviceType::VirtualGpu,
                WgpuDevice::Cpu => device_type == DeviceType::Cpu,
                WgpuDevice::BestAvailable => true,
                WgpuDevice::Existing(_) => {
                    panic!("Cannot automatically create a client for an existing device! Please use init_existing_device instead.")
                }
            };

            if is_same_type {
                adapters.push(adapter);
            }
        });

    fn select(
        num: usize,
        error: &str,
        mut adapters: Vec<wgpu::Adapter>,
        mut adapters_other: Vec<wgpu::Adapter>,
    ) -> wgpu::Adapter {
        if adapters.len() <= num {
            if adapters_other.len() <= num {
                panic!(
                    "{}, adapters {:?}, other adapters {:?}",
                    error,
                    adapters
                        .into_iter()
                        .map(|adapter| adapter.get_info())
                        .collect::<Vec<_>>(),
                    adapters_other
                        .into_iter()
                        .map(|adapter| adapter.get_info())
                        .collect::<Vec<_>>(),
                );
            }

            return adapters_other.remove(num);
        }

        adapters.remove(num)
    }

    let adapter = match device {
        WgpuDevice::DiscreteGpu(num) => select(
            *num,
            "No Discrete GPU device found",
            adapters,
            adapters_other,
        ),
        WgpuDevice::IntegratedGpu(num) => select(
            *num,
            "No Integrated GPU device found",
            adapters,
            adapters_other,
        ),
        WgpuDevice::VirtualGpu(num) => select(
            *num,
            "No Virtual GPU device found",
            adapters,
            adapters_other,
        ),
        WgpuDevice::Cpu => select(0, "No CPU device found", adapters, adapters_other),
        WgpuDevice::BestAvailable => {
            let mut most_performant_adapter = None;
            let mut current_score = -1;

            adapters
                .into_iter()
                .chain(adapters_other)
                .for_each(|adapter| {
                    let info = adapter.get_info();
                    let score = match info.device_type {
                        DeviceType::DiscreteGpu => 5,
                        DeviceType::Other => 4, // Let's be optimistic with the Other device, it's
                        // often a Discrete Gpu.
                        DeviceType::IntegratedGpu => 3,
                        DeviceType::VirtualGpu => 2,
                        DeviceType::Cpu => 1,
                    };

                    if score > current_score {
                        most_performant_adapter = Some(adapter);
                        current_score = score;
                    }
                });

            if let Some(adapter) = most_performant_adapter {
                adapter
            } else {
                panic!("No adapter found for graphics API {:?}", G::default());
            }
        }
        WgpuDevice::Existing(_) => unreachable!(),
    };

    log::info!("Using adapter {:?}", adapter.get_info());

    adapter
}
