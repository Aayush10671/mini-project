from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """
    Returns a list of requirements from requirements.txt
    """

    requirements: List[str] = []

    with open(file_path, "r") as file:
        requirements = file.readlines()
        requirements = [req.strip() for req in requirements]

    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name="mini_project",
    version="0.0.1",
    author="Aayush Shah",
    author_email="aayush0131@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)