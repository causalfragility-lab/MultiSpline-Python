from setuptools import setup, find_packages

setup(
    name="multispline",
    version="0.1.0",
    author="Subir Hait",
    author_email="haitsubi@msu.edu",
    description="Nonlinear multilevel spline modeling",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/causalfragility-lab/MultiSpline-Python",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Statistics",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "pandas>=1.3",
        "scipy>=1.7",
        "statsmodels>=0.13",
        "patsy>=0.5",
        "matplotlib>=3.4",
    ],
)
