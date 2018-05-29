# pylint: skip-file
import os
import sys
from setuptools import setup

runtime_requires = ['blinker==1.4.0']
tests_require=['pytest-runner', 
                  'pytest-pylint',
                  'pytest',
                  'pytest-cov',
                  'pytest-mock',
                  'pylint',
                  'pep8',
                  'pympler',
                  'sphinx',
                  'pydocstyle']


setup(name='bushn',
      version="1.0.10",
      description='There can be only one... per storage.',
      url='https://gitlab.crosscloud.me/CrossCloud/bushn',
      author='CrossCloud GmbH',
      author_email='code+bushn@crosscloud.me',
      packages=['bushn'],
      install_requires=runtime_requires,
      setup_requires=['bumpversion'] + runtime_requires,
      tests_require=tests_require + runtime_requires,
      extras_require={'test': tests_require},
      zip_safe=False)
