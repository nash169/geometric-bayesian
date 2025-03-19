#!/usr/bin/env python
# encoding: utf-8

from setuptools import setup, find_namespace_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="geometric-bayesian",
    version="1.0.0",
    author="Bernardo Fichera",
    author_email="bernardo.fichera@gmail.com",
    description="Geometric Bayesian Inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nash169/geometric-bayesian.git",
    packages=find_namespace_packages(),
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "jax",
        "numpy",
        "flax",
        "matplotlib",
        "jaxtyping",
    ],
    extras_require={
        "dev": [
            "pylint",
            "jupyter",
        ]
    },
)
