import codecs
import os

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


# https://packaging.python.org/guides/single-sourcing-package-version/
def read(rel_path):
    with codecs.open(os.path.join(this_directory, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delimiter = '"' if '"' in line else "'"
            return line.split(delimiter)[1]


def run_setup():
    # fixes warning https://github.com/pypa/setuptools/issues/2230
    from setuptools import setup, find_packages
    from .dalex._global_checks import OPTIONAL_DEPENDENCIES

    test_requirements = []  # input dependencies for test, but not for user
    test_requirements += [k + ">=" + v for k, v in OPTIONAL_DEPENDENCIES.items()]
    del test_requirements[0]  # remove artificial dependency used in test_global.py

    setup(
        name="dalex",
        author="Wojciech Kretowicz, Hubert Baniecki, Przemyslaw Biecek",
        author_email="wojtekkretowicz@gmail.com, hbaniecki@gmail.com",
        version=get_version("dalex/__init__.py"),
        description="DALEX in Python",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/ModelOriented/DALEX",
        classifiers=[
            "Development Status :: 4 - Beta",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "License :: OSI Approved",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            "Operating System :: OS Independent",
        ],
        install_requires=[
            'pandas>=1.1.0',
            'numpy>=1.18.1',
            'plotly>=4.9.0',
            'tqdm>=4.48.2'
        ],
        test_requirements=test_requirements,
        packages=find_packages(include=["dalex", "dalex.*"]),
        python_requires='>=3.6',
        include_package_data=True
    )


if __name__ == "__main__":
    # allows for "from setup import OPTIONAL_DEPENDENCIES" in '_global_checks.py'
    run_setup()
