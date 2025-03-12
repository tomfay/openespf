from setuptools import setup, find_packages

setup(
    name="openespf",  # Replace with your package name
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy", "scipy", "pyscf", "openmm", "matplotlib"
    ],
    author="Thomas Fay",
    author_email="tom.patrick.fay@gmail.com",
    description="A toolkit for performing polarizable QM/MM simulations.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tomfay/openespf",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)