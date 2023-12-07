
# bardi: reproducible data pre-processing pipelines
bardi (Batch-processing Abstraction for Raw Data Integration) is a framework for building reproducible data pre-processing pipelines for both machine learning model training and inference workflows.

Our initial release version is written for efficient and reproducible pre-processing of data on a single node utilizing the CPU for computation. Future development goals aim to provide the same functionality while utilizing different hardware such as, distributed computation across multiple nodes, computation on a Spark cluster, and computation utilizing available GPUs.

Installation
==================

``pip install bardi``

Documentation
============================

* Coming Soon

Tutorial and Example Scripts
============================

```
test
```

Key Features
============
* **Efficiency** - with power of Polars and multithreading, bardi ensures rapid transformation of even large datasets.
* **Modularity** - designed with component-based architecture, users can integrate individual modules based on their specific needs. Each module (normalizer, pre-tokenizer, splitter etc.) operates as individual unit.
* **Extendibility** - bardi's design allows for seamless integration of new modules and methods.
