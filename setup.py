from setuptools import find_packages, setup
from typing import List

dash_e_dot = "-e ."

def packages(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

        if dash_e_dot in requirements:
            requirements.remove(dash_e_dot)

    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='sa',
    packages=find_packages(),
    install_requires=packages('requirements.txt')  # Fixed key name
)
