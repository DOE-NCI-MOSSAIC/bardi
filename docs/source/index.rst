.. bardi documentation master file, created by
   sphinx-quickstart on Sun Dec 10 16:37:32 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to bardi's documentation!
=================================

bardi (Batch-processing Abstraction for Raw Data Integration) is a framework for building reproducible data pre-processing pipelines for both machine learning model training and inference workflows.

Our initial release version is written for efficient and reproducible pre-processing of data on a single node utilizing the CPU for computation. Future development goals aim to provide the same functionality while utilizing different hardware such as, distributed computation across multiple nodes, computation on a Spark cluster, and computation utilizing available GPUs.



Installation
==================
bardi can be installed via pip:

.. code-block:: bash

   pip install bardi



============
Key Features
============
* **Abstraction** - common data pre-processing steps are abstracted into modular steps
* **Efficiency** - data is held in Apache Arrow's columnar memory model and computation is implemented with Polars, taking advantage of multithreading utilizing available CPU cores
* **Modularity** - designed with component-based architecture, users can integrate individual modules based on their specific needs. Each module can operate as an individual unit just as well as within a pipeline
* **Extensibility** - bardi's design allows for straightforward extension of capabilities to create new custom steps that haven't been created yet

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Tutorial and Example Scripts
============================

* `bardi's Github Page  <https://github.com/DOE-NCI-MOSSAIC/bardi>`_


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
