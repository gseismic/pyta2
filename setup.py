from setuptools import setup, find_packages


setup(
    name='pyta2', 
    version='0.0.1',
    packages=find_packages(),
    description='Technical Indicator Lib for stream financial data',
    install_requires = ['numpy', 'polars', 'matplotlib'],
    scripts=[],
    python_requires = '>=3',
    include_package_data=True,
    author='Liu Shengli',
    url='http://github.com/gseismic/pyta2',
    zip_safe=False,
    author_email='liushengli203@163.com'
)
