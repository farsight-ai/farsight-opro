import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="farsight-opro",  # This is the name of the package
    version="0.2.14",  # The initial release version
    author="Katie Pelton",  # Full name of the author
    description="Farsight LLM optimizer",
    long_description=long_description,  # Long description read from the the readme file
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),  # List of all python modules to be installed
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],  # Information to filter the project on PyPi website
    python_requires=">=3.6",  # Minimum version requirement of the package
    py_modules=[
        "opro",
    ],  # Name of the python package
    package_dir={"": "opro/src"},  # Directory of the source code of the package
    install_requires=[
        "openai",
        "random",
        "tqdm",
    ],  # Install other dependencies if any
)
