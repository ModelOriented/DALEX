import os
import ast


this_directory = os.path.abspath(os.path.dirname(__file__))

# https://packaging.python.org/guides/single-sourcing-package-version/
def read(rel_path):
    """Read a file relative to the setup.py location."""
    with open(os.path.join(this_directory, rel_path), encoding='utf-8') as fp:
        return fp.read()

readme = read('README.md')
news = read('NEWS.md')

def get_version(rel_path):
    """Extract __version__ from a file without importing it."""
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delimiter = '"' if '"' in line else "'"
            return line.split(delimiter)[1]

def get_optional_dependencies(rel_path):
    """Parse OPTIONAL_DEPENDENCIES dict from a python file securely."""
    # read _global_checks.py and construct a list of optional dependencies
    flag = False
    to_parse = "{"

    for line in read(rel_path).splitlines():
        if flag:
            if line == "}":  # end of dict
                to_parse += line
                break
            to_parse += line.strip()
        if line.startswith('OPTIONAL_DEPENDENCIES'):  # start of dict
            flag = True

    # Use ast.literal_eval instead of eval for safety
    od_dict = ast.literal_eval(to_parse)
    od_list = [f"{k}>={v}" for k, v in od_dict.items()]
    # remove artificial dependency used in test_global.py
    del od_list[0]
    return od_list

def run_setup():
    # fixes warning https://github.com/pypa/setuptools/issues/2230
    from setuptools import setup, find_packages

    full_dependencies = get_optional_dependencies("dalex/_global_checks.py")

    setup(
        name="dalex",
        maintainer="Hubert Baniecki",
        maintainer_email="hbaniecki@gmail.com",
        author="Przemyslaw Biecek",
        author_email="przemyslaw.biecek@gmail.com",
        version=get_version("dalex/__init__.py"),
        description="Responsible Machine Learning in Python",
        long_description="\n\n".join([readme, news]),
        long_description_content_type="text/markdown",
        url="https://dalex.drwhy.ai/",
        project_urls={
            "Documentation": "https://dalex.drwhy.ai/python/",
            "Code": "https://github.com/ModelOriented/DALEX/tree/master/python/dalex",
            "Issue tracker": "https://github.com/ModelOriented/DALEX/issues",
        },
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Python :: 3.13",
            "License :: OSI Approved",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            "Operating System :: OS Independent",
        ],
        install_requires=[
            'setuptools',
            'packaging',
            'pandas>=1.5.3',
            'numpy>=1.23.5',
            'scipy>=1.6.3',
            'plotly>=6.0.0',
            'tqdm>=4.61.2',
        ],
        extras_require={'full': full_dependencies},
        packages=find_packages(include=["dalex", "dalex.*"]),
        python_requires='>=3.9',
        include_package_data=True
    )


if __name__ == "__main__":
    run_setup()
    # pdoc command: pdoc --html dalex --force --template-dir dalex\pdoc_template