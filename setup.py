import os

from setuptools import setup, find_packages


def read(fname):
    with open(fname, "r", encoding="utf-8") as f:
        return f.read()


setup(
    name="pid3net",
    version="1.0",
    author="Vu Tien-Sinh",
    author_email="sinh.vt@jaist.ac.jp",
    url="https://github.com/sinhvt3421/pid3net",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "tensorflow>=2.10.0,<=2.12.0",
        "scikit-learn",
        "numpy",
        "h5py",
        "pyyaml",
    ],
    include_package_data=True,
    keywords=["materials", "science", "ptychography", "X-ray", "deep", "networks","physics-informed"],
    extras_require={"test": ["pytest", "pytest-datadir", "pytest-benchmark"]},
    license="MIT",
    description="PID3Net - Physics-Informed Deep learning Network for Dynamic Diffraction imaging",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
