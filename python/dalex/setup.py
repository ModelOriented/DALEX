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


def get_optional_dependencies(rel_path):
    # read _global_checks.py and construct a list of optional dependencies
    flag = False
    to_parse = "{"

    for line in read(rel_path).splitlines():
        if flag:
            if line == "}":  # end
                to_parse += line
                break
            to_parse += line.strip()
        if line.startswith('OPTIONAL_DEPENDENCIES'):  # start
            flag = True

    od_dict = eval(to_parse)
    od_list = [k + ">=" + v for k, v in od_dict.items()]
    del od_list[0]  # remove artificial dependency used in test_global.py
    return od_list


def run_setup():
    # fixes warning https://github.com/pypa/setuptools/issues/2230
    from setuptools import setup, find_packages

    test_requirements = []  # input dependencies for test, but not for user
    test_requirements += get_optional_dependencies("dalex/_global_checks.py")

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
            'tqdm>=4.48.2',
            'Flask==1.1.2',
            'Flask-Cors==3.0.8',
            'requests==2.24.0'
        ],
        test_requirements=test_requirements,
        packages=find_packages(include=["dalex", "dalex.*"]),
        python_requires='>=3.6',
        include_package_data=True
    )


if __name__ == "__main__":
    run_setup()
