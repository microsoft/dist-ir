import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='dist-ir',
    version='0.0.1',
    description='An IR for representing distributed DNNs',
    long_description=long_description,
    packages=setuptools.find_packages(),
    python_requires='>=3.8',
)
