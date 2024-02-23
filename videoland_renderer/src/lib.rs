#![allow(clippy::new_without_default)]
#![allow(dead_code)]

use std::ffi::CStr;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use anyhow::{anyhow, Error};
use ash::extensions::{ext, khr};
use ash::vk;
use bitflags::bitflags;
use hassle_rs::{Dxc, DxcCompiler, DxcIncludeHandler, DxcLibrary};

struct Instance {
    messenger: vk::DebugUtilsMessengerEXT,
    debug_utils: ext::DebugUtils,
    instance: ash::Instance,
    _entry: ash::Entry,
}

impl Drop for Instance {
    fn drop(&mut self) {
        println!("dropping instance");
        unsafe {
            self.debug_utils
                .destroy_debug_utils_messenger(self.messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}

impl Instance {
    pub unsafe fn new() -> Self {
        let entry = ash::Entry::load().unwrap();

        let khronos_validation =
            CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0").unwrap();
        let instance_layers = vec![khronos_validation.as_ptr()];
        let instance_extensions = vec![ext::DebugUtils::name().as_ptr()];

        let app_info = vk::ApplicationInfo::builder().api_version(vk::API_VERSION_1_3);
        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&instance_extensions)
            .enabled_layer_names(&instance_layers);

        let instance = entry.create_instance(&create_info, None).unwrap();

        let debug_utils = ext::DebugUtils::new(&entry, &instance);

        let severity = vk::DebugUtilsMessageSeverityFlagsEXT::empty()
            | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
            | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
            | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
            | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR;

        let ty = vk::DebugUtilsMessageTypeFlagsEXT::empty()
            | vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE;

        let create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(severity)
            .message_type(ty)
            .pfn_user_callback(Some(vulkan_debug_callback));

        let messenger = debug_utils
            .create_debug_utils_messenger(&create_info, None)
            .unwrap();

        Self {
            messenger,
            debug_utils,
            instance,
            _entry: entry,
        }
    }

    unsafe fn get_physical_devices(&self) -> Vec<PhysicalDevice> {
        let devices = self.instance.enumerate_physical_devices().unwrap();

        devices
            .iter()
            .filter_map(|device| {
                let properties = self.instance.get_physical_device_properties(*device);
                let name =
                    CStr::from_bytes_until_nul(bytemuck::cast_slice(&properties.device_name))
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .to_owned();

                let queue_properties = self
                    .instance
                    .get_physical_device_queue_family_properties(*device);

                let graphics_queue_family =
                    queue_properties
                        .iter()
                        .enumerate()
                        .find_map(|(index, family)| {
                            let index = index as u32;

                            let is_graphics = family.queue_flags.contains(vk::QueueFlags::GRAPHICS);

                            is_graphics.then_some(index)
                        });

                graphics_queue_family.map(|graphics_queue_family| PhysicalDevice {
                    device: *device,
                    graphics_queue_family,
                    name,
                })
            })
            .collect()
    }

    fn raw(&self) -> &ash::Instance {
        &self.instance
    }
}

#[derive(Clone)]
struct PhysicalDevice {
    device: vk::PhysicalDevice,
    graphics_queue_family: u32,
    name: String,
}

pub type GpuAllocator = Arc<RwLock<gpu_alloc::GpuAllocator<vk::DeviceMemory>>>;

struct Device {
    instance: Arc<Instance>,

    physical_device: PhysicalDevice,
    device: ash::Device,

    dynamic_rendering_ext: khr::DynamicRendering,
    timeline_semaphore_ext: khr::TimelineSemaphore,

    timeline_semaphore: vk::Semaphore,
    sync: AtomicU64,

    queue: vk::Queue,

    allocator: GpuAllocator,
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_semaphore(self.timeline_semaphore, None);
            self.device.destroy_device(None);
        }
    }
}

impl Device {
    unsafe fn new(instance: Arc<Instance>, physical_device: PhysicalDevice) -> Self {
        let create_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(physical_device.graphics_queue_family)
            .queue_priorities(&[1.0])
            .build();

        let queue_create_infos = &[create_info];

        let extensions = vec![
            vk::KhrDynamicRenderingFn::name().as_ptr(),
            vk::KhrTimelineSemaphoreFn::name().as_ptr(),
            vk::KhrBufferDeviceAddressFn::name().as_ptr(),
        ];

        let mut buffer_device_address = vk::PhysicalDeviceBufferDeviceAddressFeatures::builder()
            .buffer_device_address(true)
            .build();

        let mut indexing_features = vk::PhysicalDeviceDescriptorIndexingFeatures::builder()
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_uniform_buffer_update_after_bind(true)
            .build();

        let mut physical_device_features = vk::PhysicalDeviceFeatures2::builder()
            .push_next(&mut buffer_device_address)
            .push_next(&mut indexing_features)
            .build();

        let mut khr_dynamic_rendering =
            vk::PhysicalDeviceDynamicRenderingFeaturesKHR::builder().dynamic_rendering(true);
        let mut khr_timeline_semaphore =
            vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR::builder().timeline_semaphore(true);
        let create_info = vk::DeviceCreateInfo::builder()
            .enabled_extension_names(&extensions)
            .queue_create_infos(queue_create_infos)
            .push_next(&mut khr_dynamic_rendering)
            .push_next(&mut khr_timeline_semaphore)
            .push_next(&mut physical_device_features);

        let device = instance
            .raw()
            .create_device(physical_device.device, &create_info, None)
            .unwrap();

        let khr_dynamic_rendering_ext = khr::DynamicRendering::new(instance.raw(), &device);
        let khr_timeline_semaphore_ext = khr::TimelineSemaphore::new(instance.raw(), &device);

        let queue = device.get_device_queue(physical_device.graphics_queue_family, 0);

        let mut semaphore_type_create_info = vk::SemaphoreTypeCreateInfo::builder()
            .semaphore_type(vk::SemaphoreType::TIMELINE_KHR)
            .initial_value(0);
        let create_info =
            vk::SemaphoreCreateInfo::builder().push_next(&mut semaphore_type_create_info);
        let timeline_semaphore = device.create_semaphore(&create_info, None).unwrap();

        let properties = unsafe {
            gpu_alloc_ash::device_properties(
                instance.raw(),
                vk::API_VERSION_1_3,
                physical_device.device,
            )
            .unwrap()
        };

        let allocator = Arc::new(RwLock::new(gpu_alloc::GpuAllocator::new(
            gpu_alloc::Config::i_am_prototyping(),
            properties,
        )));

        Device {
            instance,

            physical_device,
            device,

            dynamic_rendering_ext: khr_dynamic_rendering_ext,
            timeline_semaphore_ext: khr_timeline_semaphore_ext,

            timeline_semaphore,
            sync: AtomicU64::new(0),

            queue,

            allocator,
        }
    }

    unsafe fn barrier(
        &self,
        cmd: &CommandBuffer,
        texture: &Texture,
        from: vk::ImageLayout,
        to: vk::ImageLayout,
    ) {
        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(from)
            .new_layout(to)
            .image(texture.raw())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .build();

        self.device.cmd_pipeline_barrier(
            cmd.raw(),
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }

    fn raw(&self) -> &ash::Device {
        &self.device
    }

    fn allocator(&self) -> GpuAllocator {
        Arc::clone(&self.allocator)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Extent3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

pub struct TextureDesc {
    pub extent: Extent3D,
}

pub struct TextureViewDesc {
    pub extent: Extent3D,
}

#[derive(Clone, Copy)]
pub enum TextureLayout {
    Undefined,
    General,
    Color,
    DepthStencil,
    TransferSrc,
    TransferDst,
}

pub struct Texture {
    device: Arc<Device>,
    allocator: GpuAllocator,
    allocation: Option<gpu_alloc::MemoryBlock<vk::DeviceMemory>>,
    image: vk::Image,
}

impl Drop for Texture {
    fn drop(&mut self) {
        unsafe {
            self.allocator.write().unwrap().dealloc(
                gpu_alloc_ash::AshMemoryDevice::wrap(self.device.raw()),
                self.allocation.take().unwrap(),
            );

            self.device.raw().destroy_image(self.image, None);
        }
    }
}

impl Texture {
    unsafe fn new(device: Arc<Device>, desc: &TextureDesc) -> Result<Self, Error> {
        let create_info = vk::ImageCreateInfo::builder()
            .array_layers(1)
            .extent(vk::Extent3D {
                width: desc.extent.width,
                height: desc.extent.height,
                depth: desc.extent.depth,
            })
            .format(vk::Format::R8G8B8A8_UNORM)
            .image_type(vk::ImageType::TYPE_2D)
            .mip_levels(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC);

        let image = device.raw().create_image(&create_info, None)?;
        let requirements = device.raw().get_image_memory_requirements(image);

        let allocator = device.allocator();

        let allocation = allocator.write().unwrap().alloc(
            gpu_alloc_ash::AshMemoryDevice::wrap(device.raw()),
            gpu_alloc::Request {
                size: requirements.size,
                align_mask: requirements.alignment,
                usage: gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
                memory_types: requirements.memory_type_bits,
            },
        )?;

        device
            .raw()
            .bind_image_memory(image, *allocation.memory(), allocation.offset())?;

        Ok(Self {
            device,
            allocator,
            allocation: Some(allocation),
            image,
        })
    }

    fn raw(&self) -> vk::Image {
        self.image
    }
}

#[derive(Clone)]
pub struct TextureView {
    is_managed: bool,
    device: Arc<Device>,
    image_view: vk::ImageView,
    width: u32,
    height: u32,
}

impl Drop for TextureView {
    fn drop(&mut self) {
        unsafe {
            if !self.is_managed {
                self.device.raw().destroy_image_view(self.image_view, None);
            }
        }
    }
}

impl TextureView {
    unsafe fn new(
        device: Arc<Device>,
        texture: &Texture,
        desc: &crate::TextureViewDesc,
    ) -> Result<Self, Error> {
        let subresource_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
            .layer_count(vk::REMAINING_ARRAY_LAYERS)
            .base_array_layer(0)
            .level_count(vk::REMAINING_MIP_LEVELS)
            .base_mip_level(0);

        let create_info = vk::ImageViewCreateInfo::builder()
            .components(vk::ComponentMapping::default())
            .format(vk::Format::D24_UNORM_S8_UINT)
            .image(texture.image)
            .subresource_range(subresource_range.build())
            .view_type(vk::ImageViewType::TYPE_2D);

        let image_view = device.raw().create_image_view(&create_info, None)?;

        Ok(Self {
            is_managed: false,
            device,
            image_view,
            width: desc.extent.width,
            height: desc.extent.height,
        })
    }

    unsafe fn from_managed(
        device: Arc<Device>,
        image_view: vk::ImageView,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            is_managed: true,
            device,
            image_view,
            width,
            height,
        }
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn raw(&self) -> vk::ImageView {
        self.image_view
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BufferUsage(u32);

bitflags! {
    impl BufferUsage: u32 {
        const VERTEX = 1 << 0;
        const INDEX = 1 << 1;
        const TRANSFER_SRC = 1 << 2;
        const TRANSFER_DST = 1 << 3;
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BufferLocation {
    Cpu,
    Gpu,
}

#[derive(Debug, Clone, Copy)]
pub struct BufferAllocation {
    pub usage: BufferUsage,
    pub location: BufferLocation,
    pub size: u64,
}

pub struct Buffer {
    device: Arc<Device>,
    allocator: GpuAllocator,
    allocation: Option<gpu_alloc::MemoryBlock<vk::DeviceMemory>>,
    buffer: vk::Buffer,
    len: u64,
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.allocator.write().unwrap().dealloc(
                gpu_alloc_ash::AshMemoryDevice::wrap(self.device.raw()),
                self.allocation.take().unwrap(),
            );

            self.device.raw().destroy_buffer(self.buffer, None);
        }
    }
}

impl Buffer {
    unsafe fn new(device: Arc<Device>, allocation: BufferAllocation) -> Result<Self, Error> {
        let len = allocation.size;

        let create_info = vk::BufferCreateInfo::builder()
            .size(allocation.size)
            .usage(buffer_usage_to_vk(allocation.usage));

        let buffer = device.raw().create_buffer(&create_info, None)?;
        let requirements = device.raw().get_buffer_memory_requirements(buffer);

        let allocator = device.allocator();

        let allocation = allocator.write().unwrap().alloc(
            gpu_alloc_ash::AshMemoryDevice::wrap(device.raw()),
            gpu_alloc::Request {
                size: requirements.size,
                align_mask: requirements.alignment,
                usage: match allocation.location {
                    crate::BufferLocation::Cpu => gpu_alloc::UsageFlags::UPLOAD,
                    crate::BufferLocation::Gpu => gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
                },
                memory_types: requirements.memory_type_bits,
            },
        )?;

        device
            .raw()
            .bind_buffer_memory(buffer, *allocation.memory(), allocation.offset())?;

        Ok(Self {
            device,
            allocator,
            buffer,
            allocation: Some(allocation),
            len,
        })
    }

    pub fn write_data(&mut self, offset: u64, data: &[u8]) {
        unsafe {
            self.allocation
                .as_mut()
                .unwrap()
                .write_bytes(
                    gpu_alloc_ash::AshMemoryDevice::wrap(self.device.raw()),
                    offset,
                    data,
                )
                .unwrap();
        }
    }

    #[allow(clippy::len_without_is_empty)]
    fn len(&self) -> u64 {
        self.len
    }

    fn raw(&self) -> vk::Buffer {
        self.buffer
    }
}

fn buffer_usage_to_vk(usage: crate::BufferUsage) -> vk::BufferUsageFlags {
    let mut vk_usage = vk::BufferUsageFlags::empty();

    if usage.contains(crate::BufferUsage::VERTEX) {
        vk_usage |= vk::BufferUsageFlags::VERTEX_BUFFER;
    }

    if usage.contains(crate::BufferUsage::INDEX) {
        vk_usage |= vk::BufferUsageFlags::INDEX_BUFFER;
    }

    if usage.contains(crate::BufferUsage::TRANSFER_SRC) {
        vk_usage |= vk::BufferUsageFlags::TRANSFER_SRC;
    }

    if usage.contains(crate::BufferUsage::TRANSFER_DST) {
        vk_usage |= vk::BufferUsageFlags::TRANSFER_DST;
    }

    vk_usage
}

struct ShaderModule {
    device: Arc<Device>,

    module: vk::ShaderModule,
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        unsafe {
            self.device.raw().destroy_shader_module(self.module, None);
        }
    }
}

impl ShaderModule {
    unsafe fn new(device: Arc<Device>, code: &[u32]) -> Self {
        let create_info = vk::ShaderModuleCreateInfo::builder().code(code);

        let module = device
            .raw()
            .create_shader_module(&create_info, None)
            .unwrap();

        Self { device, module }
    }

    fn raw(&self) -> vk::ShaderModule {
        self.module
    }
}

struct ComputePipelineDesc<'a> {
    module: &'a ShaderModule,
}

struct ComputePipeline {
    device: Arc<Device>,

    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.raw().destroy_pipeline(self.pipeline, None);
            self.device.raw().destroy_pipeline_layout(self.layout, None);
        }
    }
}

impl ComputePipeline {
    unsafe fn new(device: Arc<Device>, desc: &ComputePipelineDesc) -> Self {
        let push_constant_ranges = &[vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(256)
            .build()];

        let create_info = vk::PipelineLayoutCreateInfo::builder()
            .push_constant_ranges(push_constant_ranges)
            .build();

        let layout = device
            .raw()
            .create_pipeline_layout(&create_info, None)
            .unwrap();

        let entry_point = CStr::from_bytes_until_nul(b"cs_main\0").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(desc.module.raw())
            .name(entry_point)
            .build();

        let create_info = vk::ComputePipelineCreateInfo::builder()
            .layout(layout)
            .stage(stage)
            .build();

        let pipeline = device
            .raw()
            .create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None)
            .unwrap()[0];

        Self {
            device,

            layout,
            pipeline,
        }
    }

    fn raw(&self) -> vk::Pipeline {
        self.pipeline
    }
}

struct CommandPool {
    device: Arc<Device>,

    command_pool: vk::CommandPool,
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe {
            self.device
                .raw()
                .destroy_command_pool(self.command_pool, None);
        }
    }
}

impl CommandPool {
    unsafe fn new(device: Arc<Device>, queue_family_index: u32) -> Self {
        let create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = device
            .raw()
            .create_command_pool(&create_info, None)
            .unwrap();

        Self {
            device,
            command_pool,
        }
    }

    fn raw(&self) -> vk::CommandPool {
        self.command_pool
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

struct CommandBuffer {
    command_pool: Arc<CommandPool>,

    buf: vk::CommandBuffer,
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        unsafe {
            self.command_pool
                .device()
                .raw()
                .free_command_buffers(self.command_pool.raw(), &[self.buf]);
        }
    }
}

impl CommandBuffer {
    unsafe fn new(command_pool: Arc<CommandPool>) -> Self {
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_buffer_count(1)
            .command_pool(command_pool.raw())
            .level(vk::CommandBufferLevel::PRIMARY);

        let buf = command_pool
            .device()
            .raw()
            .allocate_command_buffers(&allocate_info)
            .unwrap()[0];

        Self { command_pool, buf }
    }

    fn raw(&self) -> vk::CommandBuffer {
        self.buf
    }
}

pub struct Renderer {
    instance: Arc<Instance>,
    device: Arc<Device>,
    command_pool: Arc<CommandPool>,

    output: Texture,
    output_buffer: Buffer,
    pipeline: ComputePipeline,
}

impl Renderer {
    pub fn new(kernel: &[u32]) -> Self {
        unsafe {
            let instance = Arc::new(Instance::new());
            let mut physical_devices = instance.get_physical_devices();

            for (i, device) in physical_devices.iter().enumerate() {
                println!("{}: {}", i, device.name);
            }

            let physical_device = physical_devices.swap_remove(0);

            let device = Arc::new(Device::new(Arc::clone(&instance), physical_device.clone()));

            let command_pool = Arc::new(CommandPool::new(
                Arc::clone(&device),
                physical_device.graphics_queue_family,
            ));

            let output = Texture::new(
                Arc::clone(&device),
                &TextureDesc {
                    extent: Extent3D {
                        width: 1024,
                        height: 1024,
                        depth: 1,
                    },
                },
            )
            .unwrap();

            let module = ShaderModule::new(Arc::clone(&device), kernel);

            let pipeline = ComputePipeline::new(
                Arc::clone(&device),
                &ComputePipelineDesc { module: &module },
            );

            let output_buffer = Buffer::new(
                Arc::clone(&device),
                BufferAllocation {
                    location: BufferLocation::Cpu,
                    size: 1024 * 1024 * 4 * 4,
                    usage: BufferUsage::TRANSFER_DST,
                },
            )
            .unwrap();

            Self {
                instance,
                device,
                command_pool,

                output,
                output_buffer,
                pipeline,
            }
        }
    }

    pub fn render(&self) {
        unsafe {
            let cmd = CommandBuffer::new(Arc::clone(&self.command_pool));

            let begin_info = vk::CommandBufferBeginInfo::builder();
            self.device
                .raw()
                .begin_command_buffer(cmd.raw(), &begin_info)
                .unwrap();

            self.device.barrier(
                &cmd,
                &self.output,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );

            self.device.raw().cmd_bind_pipeline(
                cmd.raw(),
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline.raw(),
            );
            self.device.raw().cmd_dispatch(cmd.raw(), 10, 10, 1);

            self.device.barrier(
                &cmd,
                &self.output,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            let regions = &[vk::BufferImageCopy::builder()
                .image_subresource(
                    vk::ImageSubresourceLayers::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1)
                        .build(),
                )
                .buffer_offset(0)
                .image_extent(vk::Extent3D {
                    width: 1024,
                    height: 1024,
                    depth: 1,
                })
                .build()];

            self.device.raw().cmd_copy_image_to_buffer(
                cmd.raw(),
                self.output.raw(),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.output_buffer.raw(),
                regions,
            );

            self.device.raw().end_command_buffer(cmd.raw()).unwrap();

            self.device.sync.fetch_add(1, Ordering::SeqCst);

            let wait_values = &[];
            let signal_values = &[self.device.sync.load(Ordering::SeqCst)];

            let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::builder()
                .wait_semaphore_values(wait_values)
                .signal_semaphore_values(signal_values);

            let submit_command_buffers = &[cmd.raw()];
            let signal_semaphores = &[self.device.timeline_semaphore];

            let submit_info = vk::SubmitInfo::builder()
                .command_buffers(submit_command_buffers)
                .signal_semaphores(signal_semaphores)
                .push_next(&mut timeline_info)
                .build();

            self.device
                .raw()
                .queue_submit(self.device.queue, &[submit_info], vk::Fence::null())
                .unwrap();

            self.device.raw().device_wait_idle().unwrap();
        }
    }
}

fn read_shader_source(path: &str) -> Result<String, Error> {
    Ok(std::fs::read_to_string(path)?)
}

struct IncludeHandler {}

impl IncludeHandler {
    pub fn new() -> Self {
        Self {}
    }
}

impl DxcIncludeHandler for IncludeHandler {
    fn load_source(&mut self, path: String) -> Option<String> {
        read_shader_source(&path).ok()
    }
}

#[derive(Clone, Copy)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

#[allow(dead_code)]
pub struct ShaderCompiler {
    library: DxcLibrary,
    compiler: DxcCompiler,
    dxc: Dxc,
}

fn shader_profile_name(stage: ShaderStage) -> &'static str {
    match stage {
        ShaderStage::Vertex => "vs_6_0",
        ShaderStage::Fragment => "ps_6_0",
        ShaderStage::Compute => "cs_6_0",
    }
}

fn shader_entry_point(stage: ShaderStage) -> &'static str {
    match stage {
        ShaderStage::Vertex => "vs_main",
        ShaderStage::Fragment => "fs_main",
        ShaderStage::Compute => "cs_main",
    }
}

impl ShaderCompiler {
    pub fn new(compiler_path: Option<PathBuf>) -> Self {
        let dxc = Dxc::new(compiler_path).unwrap();
        let compiler = dxc.create_compiler().unwrap();
        let library = dxc.create_library().unwrap();

        Self {
            dxc,
            compiler,
            library,
        }
    }

    pub fn compile_hlsl(&self, path: &str, stage: ShaderStage) -> Result<Vec<u8>, Error> {
        let source = read_shader_source(path)?;

        let blob = self
            .library
            .create_blob_with_encoding_from_str(&source)
            .unwrap();

        let profile = shader_profile_name(stage);
        let entry_point = shader_entry_point(stage);
        let args = ["-HV 2021", "-I /", "-spirv"].as_slice();
        let mut include_handler = IncludeHandler::new();
        let defines = &[];
        let result = self.compiler.compile(
            &blob,
            path,
            entry_point,
            profile,
            args,
            Some(&mut include_handler),
            defines,
        );

        match result {
            Ok(v) => {
                let data = v.get_result().unwrap().to_vec();

                Ok(data)
            }
            Err(err) => {
                let message = self
                    .library
                    .get_blob_as_string(&err.0.get_error_buffer().unwrap().into())?;

                Err(anyhow!(message))
            }
        }
    }
}

pub struct LdrImage {
    data: Vec<u8>,
    width: u32,
    height: u32,
}

impl LdrImage {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            data: vec![0; 4 * width as usize * height as usize],
            width,
            height,
        }
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) {
        let file = File::create(path).unwrap();
        let mut writer = BufWriter::new(file);

        let mut encoder = png::Encoder::new(&mut writer, self.width, self.height);

        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);

        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&self.data).unwrap();
    }
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;

    let message = callback_data
        .p_message
        .is_null()
        .then(Default::default)
        .unwrap_or_else(|| CStr::from_ptr(callback_data.p_message).to_string_lossy());

    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => println!("Validation ERROR:   {message}"),
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => println!("Validation WARNING: {message}"),
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => println!("Validation INFO:    {message}"),
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => println!("Validation VERBOSE: {message}"),
        _ => println!("(unknown level) {message}"),
    };

    vk::FALSE
}
