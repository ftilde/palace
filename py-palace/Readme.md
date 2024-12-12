# How to use

1. Install [rust](https://www.rust-lang.org) (preferably using your operating system's package manager)
2. Install [maturin](https://github.com/PyO3/maturin) (may even be in your package manager's repo)
3. Set up a virtual enviroment, for example:
    1. Create it `python -m venv venv_dir` (only once)
    2. Enable it `source venv_dir/bin/<activate_for_your_shell>` (you'll have to do this for new each shell)

4. Build it: `./build.sh --release` (you can also omit --release for a dev build, but this will result in slower code)
5. Run it: `python demo.py`
