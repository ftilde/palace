fn main() {
    let glsl_dir = std::path::Path::new("src/glsl/");
    let glsl_dir = std::fs::canonicalize(glsl_dir).unwrap();

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
