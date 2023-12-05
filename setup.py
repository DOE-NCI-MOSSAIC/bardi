from distutils.core import setup

setup(
    name='bardi',
    version='0.2.1',
    author='Oak Ridge National Laboratory',
    packages=['bardi',
              'bardi.nlp_engineering',
              'bardi.data',
              'bardi.data.utils',
              'bardi.nlp_engineering.utils'],
    python_requires='>=3.8.0'
)
