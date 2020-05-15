#!/usr/bin/env python

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages
    
def parse_requirements(filename):
    with open(filename, "r") as file:
        lines = (line.strip() for line in file)
        return [line for line in lines if line and not line.startswith("#")]

reqs = parse_requirements("requirements.txt")
dep_links = [url for url in reqs if "http" in url]
reqs = [req for req in reqs if "http" not in req]
reqs += [url.split("egg=")[-1] for url in dep_links if "egg=" in url]


setup(
    name='bb_tracking',
    version='0.1',
    description='BeesBook tracking',
    author='David Dormagen',
    author_email='david.dormagen@fu-berlin.de',
    url='https://github.com/BioroboticsLab/bb_tracking/',
    install_requires=reqs,
    dependency_links=dep_links,
    packages=find_packages(),
    package_dir={'bb_tracking': 'bb_tracking/'}
)
