from setuptools import find_packages
from setuptools import setup


INSTALL_REQUIRES = ['tensorflow==1.3.0']


setup(
    name='trainer',
    version='0.2',
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    include_package_data=True,
)
