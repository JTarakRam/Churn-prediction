from pathlib import Path

from setuptools import find_namespace_packages, setup

BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as f:
    required_packages = [ln.strip() for ln in f.readlines()]

setup(
    name='Churn prediction',
    version='0.0.1',
    author='Tarak Ram',
    author_email='jujjurutarakram1818@gmail.com',
    url="https://github.com/JTarakRam/Churn-prediction",
    packages=find_namespace_packages(),
    install_requires=required_packages,
)