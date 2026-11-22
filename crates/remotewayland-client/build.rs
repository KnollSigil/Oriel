fn main() {
    println!("cargo:rerun-if-changed=src/qt_icon.cpp");
    println!("cargo:rerun-if-env-changed=QMAKE");

    #[cfg(feature = "qt")]
    compile_qt_icon_helper();
}

#[cfg(feature = "qt")]
fn compile_qt_icon_helper() {
    use std::path::PathBuf;
    use std::process::Command;

    let qmake_candidates: Vec<String> = std::env::var("QMAKE")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .into_iter()
        .chain(["qmake6".to_string(), "qmake".to_string()])
        .collect();

    let qt_headers = qmake_candidates
        .iter()
        .find_map(|qmake| {
            let out = Command::new(qmake)
                .args(["-query", "QT_INSTALL_HEADERS"])
                .output()
                .ok()?;
            if !out.status.success() {
                return None;
            }
            let dir = String::from_utf8(out.stdout).ok()?.trim().to_string();
            let path = PathBuf::from(&dir);
            if dir.is_empty() || !path.is_dir() {
                None
            } else {
                Some(path)
            }
        })
        .expect("failed to query QT_INSTALL_HEADERS from qmake");

    cc::Build::new()
        .cpp(true)
        .flag_if_supported("-std=c++17")
        .include(&qt_headers)
        .include(qt_headers.join("QtCore"))
        .include(qt_headers.join("QtGui"))
        .file("src/qt_icon.cpp")
        .compile("rw_qt_icon");
}
