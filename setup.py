"""
setup.py
========
Install the project as an editable package so `src` and `config` are always
importable regardless of working directory or how scripts are invoked.

ONE-TIME SETUP (after creating venv and installing requirements):
    ipl_venv/bin/pip install -e .

After this, `from src.ingestion.cricsheet_ingest import ...` works from
anywhere — VS Code debugger, terminal, Jupyter notebooks, all consistent.
"""
from setuptools import setup, find_packages

setup(
    name="ipl_predictor",
    version="0.1.0",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.10",
)