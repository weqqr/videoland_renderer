use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::Command;

type TaskResult = Result<(), Box<dyn Error>>;

fn main() {
    let command = std::env::args().nth(1);
    match command {
        Some(command) => match execute(&command) {
            Ok(_) => {}
            Err(err) => {
                println!("error: {err}");
                std::process::exit(1);
            }
        },
        None => {
            help();
        }
    }
}

fn execute(command: &str) -> TaskResult {
    match command {
        "bundle" => bundle(),
        _ => Err("unknown")?,
    }
}

fn help() {
    println!("available tasks: bundle");
}

fn bundle() -> TaskResult {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());

    let status = Command::new(cargo)
        .current_dir(project_dir())
        .args(["build", "-p", "videoland_py", "--release"])
        .status()?;

    if !status.success() {
        Err("failed to bundle addon")?;
    }

    copy(
        "target/release/videoland.dll",
        "addons/videoland/videoland.pyd",
    )?;
    copy("bin/dxcompiler.dll", "addons/videoland/dxcompiler.dll")?;
    copy_dir("kernel", "addons/videoland/kernel")?;

    Ok(())
}

fn copy<F: AsRef<Path>, T: AsRef<Path>>(from: F, to: T) -> TaskResult {
    let from = project_dir().join(from.as_ref());
    let to = project_dir().join(to.as_ref());

    println!(
        "Copying {} to {}",
        from.to_string_lossy(),
        to.to_string_lossy()
    );

    std::fs::copy(from, to)?;

    Ok(())
}

fn copy_dir<F: AsRef<Path>, T: AsRef<Path>>(from: F, to: T) -> TaskResult {
    let from = project_dir().join(from.as_ref());
    let to = project_dir().join(to.as_ref());

    println!(
        "Copying directory {} to {}",
        from.to_string_lossy(),
        to.to_string_lossy()
    );

    let _ = std::fs::remove_dir_all(&to);

    copy_dir_recursive(&from, &to)?;

    Ok(())
}

fn copy_dir_recursive(from: &Path, to: &Path) -> TaskResult {
    std::fs::create_dir_all(to)?;

    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let path = entry.path();

        if entry.file_type()?.is_dir() {
            copy_dir_recursive(&path, to)?;
        } else {
            copy(path, to.join(entry.file_name()))?;
        }
    }

    Ok(())
}

fn project_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(1)
        .unwrap()
        .into()
}
