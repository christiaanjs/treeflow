import setuptools

setuptools.setup(
    name="treeflow",
    version="0.0.1",
    packages=setuptools.find_packages(),
    install_requires=[
        "biopython==1.76",
        "ete3",
        "numdifftools",  # Test dependency
        "pytest",
    ],
    extras_require=dict(accelerated=["bito"]),
)
