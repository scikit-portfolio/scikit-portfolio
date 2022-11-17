#!/usr/bin/env python

"""The setup script."""
import os
from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf8") as readme_file:
    readme = readme_file.read()

requirements_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "requirements.txt"
)

with open(requirements_path, "r", encoding="utf8") as f:
    requirements = [r.strip() for r in f.readlines()]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Carlo Nicolini",
    author_email="nicolini.carlo@gmail.com",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Alpha",
        "Intended Audience :: Machine Learning Engineers, Portfolio Managers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description="Portfolio optimization for data scientists",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    # include_package_data=True,
    keywords="scikit-portfolio",
    name="scikit-portfolio",
    packages=find_packages(include=["skportfolio", "skportfolio.*"]),
    test_suite="tests",
    # tests_require=test_requirements,
    url="https://github.com/scikit-portfolio/scikit-portfolio",
    version="0.2.0",
    zip_safe=True,
)
