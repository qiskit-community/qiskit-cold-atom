# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import os
import sys
import setuptools
import inspect

long_description = """Qiskit cold atom is an open-source library to describe cold atomic quantum 
experiments in a gate- and circuit-based framework."""

with open("requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

VERSION_PATH = os.path.join(os.path.dirname(__file__), "qiskit_cold_atom", "VERSION.txt")
with open(VERSION_PATH, "r") as version_file:
    VERSION = version_file.read().strip()

setuptools.setup(
    name="qiskit-cold-atom",
    version=VERSION,
    description="Integration of cold atomic experiments into the Qiskit SDK.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qiskit-community/qiskit-cold-atom",
    author="Qiskit cold atom development team",
    license="Apache 2.0",
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
    ],
    keywords="qiskit sdk quantum cold atoms",
    packages=setuptools.find_packages(include=["qiskit_cold_atom", "qiskit_cold_atom.*"]),
    install_requires=REQUIREMENTS,
    extras_require={"ffsim": ["ffsim==0.0.17"]},
    include_package_data=True,
    python_requires=">=3.8",
    zip_safe=False,
)
