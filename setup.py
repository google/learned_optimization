# coding=utf-8
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup the package."""

import os
import setuptools

# https://packaging.python.org/guides/making-a-pypi-friendly-readme/
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

__version__ = '0.0.1'

setuptools.setup(
    name='learned_optimization',
    version=__version__,
    description='Train learned optimizers in Jax.',
    author='learned_optimization team',
    author_email='lmetz@google.com',
    packages=setuptools.find_packages(exclude=['examples']),
    package_data={'learned_optimization': ['py.typed']},
    python_requires='>=3.7',
    setup_requires=['setuptools_scm'],
    include_package_data=True,
    # TODO(lmetz) don't fix many versions! Sadly a number of these libraries
    # don't play nice with newer versions of other libraries.
    # TODO(lmetz) add oryx to this!
    install_requires=[
        'absl-py==0.12.0',
        'numpy>=1.18',
        'jax>=0.2.6',
        'jaxlib>=0.1.68',
        'nose',
        'dm-launchpad-nightly',
        'tqdm>=4.62.3',
        'flax',
        'dm-haiku==0.0.5',
        'optax>=0.0.9',
        'tensorflow>=2.7.0',
        'tensorflow-datasets>=4.4.0',
        'tensorflow-metadata==1.5.0',
        'tensorboard>=2.7.0',
        'gin-config>=0.5.0',
        'oryx',
    ],
    url='https://github.com/google/learned_optimization',
    license='Apache-2.0',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    zip_safe=False,
)
