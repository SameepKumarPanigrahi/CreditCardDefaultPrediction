from typing import List
from setuptools import find_packages, setup

PROJECT_NAME="credit-card-default-predictor"
VERSION="0.0.1"
AUTHOR="Sameep"
DESCRIPTION="We will add it later"
REQUIREMENTS_FILE_NAME="requirements.txt"

def get_requirements_list()->List[str]:
    """
    Description: This function is going to return list of requirement 
    mention in requirements.txt file
    return This function is going to return a list which contain name 
    of libraries mentioned in requirements.txt file
    # find_packages() method will return all the folder where __init__.py exists
    """
    with open(REQUIREMENTS_FILE_NAME) as requirement_file:
        return requirement_file.readlines().remove("-e .")

setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR,
    description=DESCRIPTION,
    packages= find_packages(),
    install_requires=get_requirements_list()
)