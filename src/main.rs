#![allow(clippy::new_without_default)]
#![allow(dead_code)]

use std::ffi::CStr;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, RwLock};

use anyhow::{Error, anyhow};
use ash::extensions::{ext, khr};
use ash::vk;
use hassle_rs::{Dxc, DxcCompiler, DxcIncludeHandler, DxcLibrary};

struct Instance {
    messenger: vk::DebugUtilsMessengerEXT,
    debug_utils: ext::DebugUtils,
    instance: ash::Instance,
    _entry: ash::Entry,
}

impl Drop for Instance {
    fn drop(&mut self) {
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
}

struct Renderer {
    instance: Arc<Instance>,
    device: Arc<Device>,
}

impl Renderer {
    fn new() -> Self {
        unsafe {
            let instance = Arc::new(Instance::new());
            let mut physical_devices = instance.get_physical_devices();

            for (i, device) in physical_devices.iter().enumerate() {
                println!("{}: {}", i, device.name);
            }

            let device = Arc::new(Device::new(
                Arc::clone(&instance),
                physical_devices.swap_remove(0),
            ));

            Self { instance, device }
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
enum ShaderStage {
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
    pub fn new() -> Self {
        let dxc = Dxc::new(Some(PathBuf::from("bin"))).unwrap();
        let compiler = dxc.create_compiler().unwrap();
        let library = dxc.create_library().unwrap();

        Self {
            dxc,
            compiler,
            library,
        }
    }

    fn compile_hlsl(&self, path: &str, stage: ShaderStage) -> Result<Vec<u8>, Error> {
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

fn main() {
    let renderer = Renderer::new();
    let compiler = ShaderCompiler::new();
    let kernel = compiler.compile_hlsl("kernel/kernel.hlsl", ShaderStage::Compute).unwrap();

    let image = LdrImage::new(1024, 1024);
    image.save("test.png");

    println!("Hello, world!");
}

struct LdrImage {
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
