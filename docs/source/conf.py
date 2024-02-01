# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

print(os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'bardi'
copyright = '2024, Dakota Murdock, Patrycja Krawczuk - Oak Ridge National Laboratory'
author = 'Dakota Murdock, Patrycja Krawczuk - Oak Ridge National Laboratory'
release = '0.3.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
   'sphinx.ext.todo',
   'sphinx.ext.viewcode',
   'sphinx.ext.autodoc',
   'sphinx.ext.napoleon'
]
autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
