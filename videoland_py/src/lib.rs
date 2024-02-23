use pyo3::prelude::*;
use videoland_renderer::{ShaderCompiler, ShaderStage};

#[pyclass]
pub struct Renderer {
    renderer: videoland_renderer::Renderer,
}

#[pymethods]
impl Renderer {
    #[new]
    fn new() -> PyResult<Self> {
        let compiler = ShaderCompiler::new(None);
        let kernel = compiler
            .compile_hlsl("kernel/kernel.hlsl", ShaderStage::Compute)
            .unwrap();

        let renderer = videoland_renderer::Renderer::new(bytemuck::cast_slice(&kernel));

        Ok(Self {
            renderer,
        })
    }

    fn render(&self) {
        self.renderer.render();
    }
}

#[pymodule]
fn videoland(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Renderer>().unwrap();
    Ok(())
}
