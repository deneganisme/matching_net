#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

# with open('README.rst') as readme_file:
#     readme = readme_file.read()
readme = ''

# with open('HISTORY.rst') as history_file:
#     history = history_file.read()
history = ''

requirements = [
    # TODO: put package requirements here
]

setup_requirements = [
    # TODO: put setup requirements (distutils extensions, etc.) here
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='matching_net',
    version='0.1.0',
    description="PyTorch implementation of Matching Net",
    long_description=readme + '\n\n' + history,
    author="Dennis Egan",
    author_email='d.james.egan@gmail.com',
    url='',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords='matching_net',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
