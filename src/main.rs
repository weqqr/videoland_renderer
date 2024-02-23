use std::path::PathBuf;

use videoland_renderer::{LdrImage, Renderer, ShaderCompiler, ShaderStage};

fn main() {
    let compiler = ShaderCompiler::new(Some(PathBuf::from("bin")));
    let kernel = compiler
        .compile_hlsl("kernel/kernel.hlsl", ShaderStage::Compute)
        .unwrap();

    let renderer = Renderer::new(bytemuck::cast_slice(&kernel));

    renderer.render();

    let image = LdrImage::new(1024, 1024);
    image.save("test.png");

    println!("Hello, world!");
}
