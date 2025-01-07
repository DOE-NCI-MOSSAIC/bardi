from distutils.core import setup

setup(
    name='bardi',
    version='0.5.0',
    author='Oak Ridge National Laboratory',
    packages=['bardi',
              'bardi.nlp_engineering',
              'bardi.data',
              'bardi.data.utils',
              'bardi.nlp_engineering.regex_library',
              'bardi.nlp_engineering.utils'],
    python_requires='>=3.9.0'
)
