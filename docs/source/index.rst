.. GAuDI documentation master file, created by
   sphinx-quickstart on Tue Oct 24 15:51:54 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GAuDI's documentation!
=================================
**GAuDI** - Generalized Automatic Data Ingestion pipeline is a data pre-processing toolkit for the NCI project.
With GAuDI you are able to swiftly construct personalized data preprocessing pipeline tailored to your needs.



Installation
==================
At this stage GAuDI can be installed via pip into your favorite conda from the wheel file in a shared location:

.. code-block:: bash

   pip install /mnt/nci/scratch/packages/gaudi-0.2.0-py3-none-any.whl


The package and all its dependencies are also available in the shared conda environment:

.. code-block:: bash

   source /mnt/nci/scratch/data_engineering_conda/etc/profile.d/conda.sh
   conda activate /mnt/nci/scratch/data_engineering_conda/envs/test-gaudi/

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   modules


Key Features
============

* **Efficiency** - with power of Polars and multithreading, GAuDI ensures rapid transformation of even large datasets.
* **Modularity** - designed with component-based architecture, users can integrate individual modules based on their specific needs. Each module (normalizer, pre-tokenizer, splitter etc.) operates as individual unit.
* **Extendibility** - GAuDI's design allows for seamless integration of new modules and methods.

Tutorial and Example Scripts
============================

* `GAuDi Run Scripts Repo <https://ncigitlab01.repd.ornlkdi.org/nci/gaudi-run-scripts>`_
* `GAuDI GitLab <https://ncigitlab01.repd.ornlkdi.org/nci/gaudi>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
