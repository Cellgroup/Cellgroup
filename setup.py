# TODO: replace with pyproject.toml
from setuptools import setup, find_packages

setup(
    name="Cellgroup",
    version="0.0.1",
    author="Guido Putignano",
    author_email="guido.putignano@gmail.com",
    url="https://github.com/guidoputignano/guidoputignano",
    description="A library whose function is to cluster, analyse, and potentially forecast cells in a 2D environment",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["click", "pytz"], #to write once ocmpleted
    entry_points={"console_scripts": ["Cellgroup = src.main:main"]},
)