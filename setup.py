import setuptools

setuptools.setup(
    name='treeflow',
    version='0.0.1',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'tf-nightly',
        'tfp-nightly',
        'jupyter',
        'ete3',
        'pytest'
    ]
)
