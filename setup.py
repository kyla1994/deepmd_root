from setuptools import setup,find_packages
setup(name='deepmd',
version='1.2',
author='Deepmd Group',
packages=find_packages(),
package_data={
    '':['*'],
},
install_requires = ["msgpack"]
)