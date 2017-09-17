from setuptools import setup
from setuptools import find_packages

setup(
    name="l2a",
    version="1.0.0",
    description="Machine learning sample for integer addition",
    author="Philippe Trempe",
    author_email="ph.trempe@gmail.com",
    url="https://github.com/PhTrempe/l2a",
    license="MIT",
    install_requires=[
        "numpy",
        "scipy",
        "tensorflow",
        "keras",
        "pyyaml",
        "h5py"
    ],
    packages=find_packages()
)
