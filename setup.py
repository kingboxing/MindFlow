from setuptools import setup, find_packages

setup(
    name='ferepack',  # This is the custom name for the package
    version='0.1',
    packages=find_packages(),  # This will find all subpackages under src/
    package_dir={'': 'src'},  # Define 'src' as the root directory for modules
    include_package_data=True,
    install_requires=[
        'python>=3.9.0'
        'FEniCS==2019.1.0',
        'scipy',
        'numpy',
        'matplotlib',
        'scikit-sparse',
        'scikit-umfpack',
        'pymess',
        'matlab.engine'
        'psutil',
        'time',
        'joblib',
        'copy',
        'multiprocessing',
        'petsc4py',
        'gc',
        'os',
        'sys',
        # List your dependencies here (e.g., fenics, numpy)
    ],
    description='FERePack: FEniCS-based package for frequency-based analysis and control problems.',
    author='Bo Jin',
    author_email='jinbo199188@gmail.com',
    url='xxxxx',  # Your repository URL
)