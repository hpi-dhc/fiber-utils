#!/usr/bin/env python
# -*- coding: utf-8 -*-

'The setup script.'

from setuptools import find_packages, setup

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.read().split('\n')

setup_requirements = [
    'pytest-runner',
]

test_requirements = [
    'pytest',
]

setup(
    author='Tom Martensen',
    author_email='mail@tommartensen.de',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description=(
        'Utility methods for FIBER that do not belong into the core module.'
        'Like for visualizations, cohort filters, or time series '
        'transformation.'
    ),
    install_requires=requirements,
    license='MIT license',
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='fiberutils',
    name='fiberutils',
    packages=find_packages(include=['fiberutils']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://gitlab.hpi.de/fiber/fiber-utils',
    version='0.0.1',
    zip_safe=False,
)
