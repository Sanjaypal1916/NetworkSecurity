'''
The setup.py file is an essential part of packaging and 
distributing Python projects. It is used by setuptools 
(or distutils in older Python versions) to define the configuration 
of your project, such as its metadata, dependencies, and more
'''

from setuptools import find_packages, setup
from typing import List

def get_requirements() -> List[str]:
    requirements_list : List[str] = []
    try:
        with open("requirements.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement!="-e .":
                    requirements_list.append(requirement)
    except Exception as e:
        print(f"requirements.txt file could note b found : {e}")
        raise e

    return requirements_list

setup(
    name="NetworkSecurity",
    version="0.0.1",
    author="Pal Sanjay",
    author_email="sanjaypal606060@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)


