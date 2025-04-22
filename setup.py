from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='mdds',
    packages=find_packages(exclude=['tests*']),
    version='0.0.1',
    package_data = {
        '': ['*.ttf'],
    },

    description='Package to perform MDDS',
    long_description=long_description,
    long_description_content_type='text/markdown',

    url='https://github.com/arthur-pe/mdds',
    author='Arthur Pellegrino',
    license='MIT',
    install_requires=['jax==0.4.35',
                      'equinox',
                      'quax',
                      'diffrax',
                      'tqdm',
                      'scipy',
                      'matplotlib',
                      'optax',
                      'PyYAML',
                      'finitediffx'
                      ],
    python_requires='>=3.10',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
)