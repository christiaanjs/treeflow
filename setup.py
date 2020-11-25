import setuptools

setuptools.setup(
    name='treeflow',
    version='0.0.1',
    packages=setuptools.find_packages(),
    install_requires=[
        'libsbn',
        'biopython',
        'ete3',
        'numdifftools' # Test dependency
    ]
)
