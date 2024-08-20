from setuptools import setup, find_packages

setup(
    name="experiments_utils",  # Replace with your package name
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "wandb >= 0.1",
        "torch >= 2.0",
        "numpy >= 1.18.5",
    ],
    url="https://github.com/jvasso/experiments_utils",
    author="Jean Vassoyan",
    author_email="",
    python_requires=">=3.6",
)
