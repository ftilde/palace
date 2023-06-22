fn main() {
    let glsl_dir = std::path::Path::new("src/glsl/");
    let glsl_dir = std::fs::canonicalize(glsl_dir).unwrap();

    // Specifying one instance of rerun-if-changed disables cargo's conservative approach of
    // rebuilding when anything in the directory changes. This is helpful because we don't care if
    // shaders changed since these are compiled at run-time.
    println!("cargo:rerun-if-changed=build.rs");

    //TODO This is somewhat hacky greatly hinders deployment of the generated binary to other
    //systems. Potential solutions are:
    //  1) Embed included glsl files into binary and possibly use glsl_include crate to include
    //     them. This has the disadvantage of bad line information on compile errors (which may be
    //     fine for desployment).
    //  2) Deploy the glsl files into some kind of ressource folder that is shipped with the
    //     binary.
    println!(
        "cargo:rustc-env=GLSL_INCLUDE_DIR={}",
        glsl_dir.to_str().unwrap()
    );
}
