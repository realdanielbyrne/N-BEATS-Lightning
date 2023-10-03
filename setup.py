from pathlib import Path
from setuptools import setup, find_packages


def read_requirements(path):
    return list(Path(path).read_text().splitlines())
  
base_reqs = read_requirements("requirements.txt")

with open("README.md") as fh:
    LONG_DESCRIPTION = fh.read()

URL = 'https://github.com/realdanielbyrne/N-BEATS-Lightning'

setup(
    name='nbeats_lightning',
    version='0.1.0',
    url=URL,
    author='Daniel Byrne',
    author_email='realdanielbyrne@icloud.com',
    description='A Pytorch Lignthing implementation of N-BEATS.',
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(),    
    python_requires=">=3.9",
    zip_safe=False,        
    install_requires=base_reqs, 
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",        
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    keywords="time series forecasting",
)
