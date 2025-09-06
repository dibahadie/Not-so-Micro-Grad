from setuptools import setup, find_packages

setup(
    name="my_autograd",
    version="0.1.0",
    description="A not so simple autograd engine",
    author="Diba Hadi Esfangereh",
    author_email="diba.hadie@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
    ],
    python_requires=">=3.12",
)