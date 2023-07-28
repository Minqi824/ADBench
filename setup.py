from setuptools import setup, find_packages
import os

# read the contents of requirements.txt
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().splitlines()

# 配置
setup(
    name="adbench",
    version='0.1.10',
    author="Minqi Jiang",
    author_email="<jiangmq95@163.com>",
    url='https://github.com/Minqi824/ADBench',
    description='Python package of ADBench',
    long_description='Python package of ADBench: Anomaly detection benchmark. Fast implementation of the large '
                     'experiments in ADBench and your customized AD algorithm.',
    packages=find_packages(),
    include_package_data=False,
    install_requires=requirements,
    keywords=['anomaly detection', 'outlier detection', 'tabular data', 'benchmark'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ]
)