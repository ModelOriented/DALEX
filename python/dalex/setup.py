from os import path

import setuptools

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="dalex",
    version="0.1.8.9000",
    author="Wojciech Kretowicz, Hubert Baniecki, Przemyslaw Biecek",
    author_email="wojtekkretowicz@gmail.com, hbaniecki@gmail.com",
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
        'pandas>=1.0.1',
        'numpy>=1.18.1',
        'plotly>=4.5.2',
        'tqdm>=4.42.1'
    ],
    packages=setuptools.find_packages(include=["dalex", "dalex.*"]),
    python_requires='>=3.6',
    include_package_data=True
)
