"""
Setup script for Energy Optimization ML package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements-docker.txt", "r", encoding="utf-8") as fh:
    requirements = [line.split("#")[0].strip() for line in fh]
    requirements = [r for r in requirements if r and not r.startswith("-r")]

setup(
    name="energy-optimization-ml",
    version="1.0.0",
    author="Etisam Ul Haq",
    description="Advanced ML system for building energy optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/etisamhaq/energy-optimization-ai-ml-project",
    # Code imports itself as `src.<module>`, so `src` is the top-level package
    packages=find_packages(include=["src", "src.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dl": ["tensorflow==2.13.0", "torch==2.0.1", "pytorch-lightning==2.0.6"],
    },
    entry_points={
        "console_scripts": [
            "energy-collect=src.data.collect_data:main",
            "energy-preprocess=src.data.preprocess:main",
            "energy-train=src.models.train:main",
            "energy-predict=src.models.predict:main",
            "energy-api=src.api.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
)