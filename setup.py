from setuptools import setup, find_packages

setup(
    name="search-rescue-simulation",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "scipy>=1.5.0",
        "tqdm>=4.50.0",
    ],
    author="William Smith",
    description="Multi-agent coordination for search and rescue operations",
    python_requires=">=3.7",
)