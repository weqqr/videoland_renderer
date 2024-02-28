use pyo3::prelude::*;
use videoland_renderer::{HdrImage, ShaderCompiler, ShaderStage};

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

        Ok(Self { renderer })
    }

    fn render<'a>(&self, width: i32, height: i32, py: Python<'a>) -> PyResult<&'a PyAny> {
        self.renderer.render();
        let image = HdrImage::new(width as u32, height as u32);

        let array = PyModule::import(py, "array")?;
        let data = array
            .getattr("array")?
            .call1(("f", image.into_values()));

        data
    }
}

#[pymodule]
fn videoland(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Renderer>().unwrap();
    Ok(())
}
