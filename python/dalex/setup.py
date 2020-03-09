import setuptools

setuptools.setup(
    name="dalex",
    version="0.1.0",
    author="Wojciech Kretowicz, Hubert Baniecki, Przemysław Biecek",
    author_email="wojtekkretowicz@gmail.com, hbaniecki@gmail.com",
    description="DALEX in Python",
    long_description="moDel Agnostic Language for Exploration and eXplanation",
    long_description_content_type="text/markdown",
    url="https://github.com/ModelOriented/DALEX",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'pandas',
        'numpy',
        'tqdm',
        'plotly>=4.5.2'
    ],
    packages=setuptools.find_packages(include=["dalex", "dalex.*"]),
    python_requires='>=3.6',
    include_package_data=True
)
