"""
Setup script untuk Legonizer4
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="legonizer4",
    version="1.0.0",
    author="Legonizer4 Team",
    author_email="team@legonizer4.com",
    description="Sistem identifikasi LEGO brick menggunakan machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yohannes1097/legonizer4",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
    },
    entry_points={
        "console_scripts": [
            "legonizer4-api=src.api.main:main",
            "legonizer4-train=src.models.trainer:main",
            "legonizer4-preprocess=src.preprocessing.processor:main",
        ],
    },
    include_package_data=True,
    package_data={
        "legonizer4": ["data/*", "data/**/*"],
    },
)
