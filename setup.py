from setuptools import setup, find_packages
from pathlib import Path
# Read the long description from the README file
this_directory = Path(__file__).parent.resolve()
long_description = (this_directory / "README.md").read_text(encoding='utf-8')


setup(
    name='ferepack',
    version='0.1.0',
    description='A FEniCS-based package for frequency-domain analysis and optimal control of fluid dynamics.',
    keywords='fenics, fluid dynamics, frequency analysis, optimal control',
    long_description=long_description,
    long_description_content_type='text/markdown',  # Specify the format of the long description
    author='Bo Jin',
    author_email='jinbo199188@gmail.com',
    url='https://kingboxing@bitbucket.org/kingboxing/ferepack.git',  # Optional
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'scipy==1.8.0',
        'numpy==1.24.0',
        'matplotlib==3.5.1',
        'scikit-sparse==0.4.15',
        'scikit-umfpack==0.3.3',
        'pymess',
        'petsc4py',
        'psutil',
        'joblib',
        'h5py',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',  # Project maturity
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',  # Define minimum Python version
)