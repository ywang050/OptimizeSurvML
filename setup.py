from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='OptimizeSurvML',
    version=0.1,

    url='https://github.com/ywang050/OptimizeSurvML',
    author='Yifan Wang',
    author_email='ywang050@uottawa.ca',
    license='MIT',
    description='Tools to help automate survival ML model selection & optimization',
    long_description=open('README.md').read(),

    packages = find_packages(),

    install_requires = requirements

)
