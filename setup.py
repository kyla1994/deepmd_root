from setuptools import setup,find_packages
setup(name='deepmd',
version='1.2',
author='Deepmd Group',
packages=find_packages(),
package_data={
    '':['*.txt','*.so', 'dp_*'],
},
entry_points = {
    'console_scripts':[
        'dp_train = bin.dp_train:_main',
        'dp_test = bin.dp_test:_main',
        'dp_frz = bin.dp_frz:_main', 
    ]
}
)
