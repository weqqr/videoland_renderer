use std::ffi::CStr;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::Arc;

use ash::extensions::ext;
use ash::vk;

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
            .map(|device| {
                let properties = self.instance.get_physical_device_properties(*device);
                let name =
                    CStr::from_bytes_until_nul(bytemuck::cast_slice(&properties.device_name))
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .to_owned();

                PhysicalDevice { name }
            })
            .collect()
    }
}

struct PhysicalDevice {
    name: String,
}

struct Renderer {
    instance: Arc<Instance>,
}

impl Renderer {
    fn new() -> Self {
        unsafe {
            let instance = Arc::new(Instance::new());
            let physical_devices = instance.get_physical_devices();

            for (i, device) in physical_devices.iter().enumerate() {
                println!("{}: {}", i, device.name);
            }

            Self { instance }
        }
    }
}

fn main() {
    let renderer = Renderer::new();

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
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => println!("{message}"),
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => println!("{message}"),
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => println!("{message}"),
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => println!("{message}"),
        _ => println!("(unknown level) {message}"),
    };

    vk::FALSE
}
