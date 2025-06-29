# setup.py
from setuptools import setup, find_packages

setup(
  name="elec_pinn",
  version="0.1",
  packages=find_packages("src"),
  package_dir={"": "src"},
  install_requires=["torch", "numpy", "pandas", "pyyaml", "scikit-learn", "matplotlib", "scipy"],
)
