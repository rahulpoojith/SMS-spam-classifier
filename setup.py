from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    '''
    This function reads the requirements.txt file and returns a list of dependencies.
    '''
    try:
        with open(file_path, 'r') as file:
            requirements = file.readlines()
            requirements = [req.strip() for req in requirements]  # Remove newline characters
            if HYPHEN_E_DOT in requirements:
                requirements.remove(HYPHEN_E_DOT)
        return requirements
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return []

setup(
    name='SMS-SPAM-CLASSIFICATION PROJECT',
    version='0.0.1',
    author='Rahul',
    author_email='poojith.p.rahul@gmail.com',
    packages=find_packages(),  # Automatically finds all packages in the directory
    install_requires=get_requirements('requirements.txt')  # Reads dependencies
)