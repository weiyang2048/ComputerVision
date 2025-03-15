from setuptools import setup, find_packages

setup(
    name="ComputerVision",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "pillow",
        "matplotlib",
        "pandas",
        "scikit-learn",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "mypy",
        ],
    },
    author="Wei Yang",
    author_email="weiyang2048@gmail.com",
    description="A computer vision package for various vision-related tasks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/weiyang2048/ComputerVision",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
