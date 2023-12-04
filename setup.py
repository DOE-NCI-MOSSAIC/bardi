from distutils.core import setup

setup(
    name='gaudi',
    version='0.2.1',
    author='Oak Ridge National Laboratory',
    packages=['gaudi',
              'gaudi.nlp_engineering',
              'gaudi.data',
              'gaudi.data.utils',
              'gaudi.nlp_engineering.utils'],
    python_requires='>=3.8.0'
)
