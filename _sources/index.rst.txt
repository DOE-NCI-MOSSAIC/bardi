.. bardi documentation master file, created by
   sphinx-quickstart on Mon Jan 29 14:56:15 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to bardi's documentation!
=================================

BARDI (Batch-processing Abstraction for Raw Data Integration), is a specialized framework 
engineered to facilitate the development of reproducible data pre-processing pipelines within machine learning workflows. 

It emphasizes the following key aspects:
   *   **Abstraction:** By transforming common data pre-processing operations into modular components, Bardi simplifies both the 
       development and upkeep of complex data pipelines.
   *   **Efficiency:** Utilizing Apache Arrow's columnar memory model for data storage and Polars for computations, Bardi enhances 
       processing speed through multithreading, optimizing the use of available CPU resources.
   *   **Modularity:** Bardi's design is based on a component-driven architecture, offering users the flexibility to incorporate 
       specific modules tailored to their unique requirements. These modules are crafted to operate seamlessly both as standalone 
       units and within the context of a comprehensive pipeline.
   *   **Extensibility:** Designed with future growth in mind, Bardi allows for the straightforward addition of new custom steps, 
       thereby broadening its functionality to encompass unaddressed demands and evolving data processing needs.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   installation
   basic
   advanced
   bardi.pipeline
   bardi.data
   bardi.nlp_engineering


Indices and tables
==================

* :ref:`modindex`
