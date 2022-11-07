from setuptools import find_packages,setup
from typing import List

def get_requirements() -> List[str]:
    '''
    This function will return list of requirements
    '''
    requirements_list:List[str]=[]
    with open('requirements.txt') as requirements:
        requirements_list= requirements.readlines().remove('-e .')
    return requirements_list

setup(

    name = "order",
    version = '0.0.1',
    author = 'Vishal',
    author_email ='vishal.choudhary2404@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements()
)