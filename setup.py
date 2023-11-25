import setuptools

setuptools.setup(
    name='cqarchaeo',
    version='0.0.5',
    author='Lorenzo Cardarelli',
    author_email='lorenzocardarelli2@gmail.com',
    scripts=[],
    description='A simple package to analyze archaeological data using Quantogram analysis',
    package_dir={"": "src"},
    project_urls = {
        "to add": "ulr to add",
    },
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9",
    #
    install_requires=[
        "pandas", "matplotlib", "numpy", "seaborn", "tqdm", "openpyxl"
    ]
    #
)