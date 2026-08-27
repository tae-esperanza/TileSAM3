from setuptools import setup, find_packages

setup(
    name="droplet_detection",
    version="1.0",
    url="https://github.com/yourusername/Droplet-Detection",
    packages=find_packages(),
    install_requires=["segment-anything", "sam3"],
)
