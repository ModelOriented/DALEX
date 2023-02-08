def run_setup():
    # fixes warning https://github.com/pypa/setuptools/issues/2230
    from setuptools import find_packages, setup

    setup(
        name="dalex-evaluation",
        install_requires=[
            "setuptools",
            "pyarrow",
        ],
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        python_requires=">=3.7",
    )


if __name__ == "__main__":
    run_setup()
    # pdoc command: pdoc --html dalex --force --template-dir dalex\pdoc_template
