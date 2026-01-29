# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 18:37:28 2023

@author: Yue
"""

from setuptools import setup

# Get the requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="sleep_scoring",
    version="0.1",
    py_modules=["app", "inference", "make_figure", "model", "utils"],
    include_package_data=False,
    install_requires=requirements,
)
