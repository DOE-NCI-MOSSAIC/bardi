
# BARDI (Batch-processing Abstraction for Raw Data Integration)
BARDI is a specialized framework engineered to facilitate the development of reproducible data pre-processing pipelines within machine learning workflows. 

It emphasizes the following key aspects:
    - Abstraction: By transforming common data pre-processing operations into modular components, Bardi simplifies both the development and upkeep of complex data pipelines.
    - Efficiency: Utilizing Apache Arrow's columnar memory model for data storage and Polars for computations, Bardi enhances processing speed through multithreading, optimizing the use of available CPU resources.
    - Modularity: Bardi's design is based on a component-driven architecture, offering users the flexibility to incorporate specific modules tailored to their unique requirements. These modules are crafted to operate seamlessly both as standalone units and within the context of a comprehensive pipeline.
    - Extensibility: Designed with future growth in mind, Bardi allows for the straightforward addition of new custom steps, thereby broadening its functionality to encompass unaddressed demands and evolving data processing needs.

Installation
============

``pip install bardi``

_currently not supported on Windows_

Documentation
=============

*  [bardi's docs](https://doe-nci-mossaic.github.io/bardi/)

Tutorial
========

## Preparing a sample set of data

bardi offers several ways of loading data, but to keep this simple for now we are going to create a pandas DataFrame from some example data and show some of the basic pipeline functionality.

```python
import pandas as pd

# create some sample data
df = pd.DataFrame([
    {
        "patient_id_number": 1,
        "text": "The patient presented with notable changes in behavior, exhibiting increased aggression, impulsivity, and a distinct deviation from the Jedi Code. Preliminary examinations reveal a heightened midichlorian count and an unsettling connection to the dark side of the Force. Further analysis is warranted to explore the extent of exposure to Sith teachings. It is imperative to monitor the individual closely for any worsening symptoms and to engage in therapeutic interventions aimed at preventing further descent into the dark side. Follow-up assessments will be crucial in determining the efficacy of intervention strategies and the overall trajectory of the individual's alignment with the Force.",
        "dark_side_dx": "positive",
    },
    {
        "patient_id_number": 2,
        "text": "Patient exhibits no signs of succumbing to the dark side. Preliminary assessments indicate a stable midichlorian count and a continued commitment to Jedi teachings. No deviations from the Jedi Code or indicators of dark side influence were observed. Regular check-ins with the Jedi Council will ensure the sustained well-being and alignment of the individual within the Jedi Order.",
        "dark_side_dx": "negative",
    },
    {
        "patient_id_number": 3,
        "text": "The individual manifested heightened aggression, impulsivity, and a palpable deviation from established ethical codes. Initial examinations disclosed an elevated midichlorian count and an unmistakable connection to the dark side of the Force. Further investigation is imperative to ascertain the depth of exposure to Sith doctrines. Close monitoring is essential to track any exacerbation of symptoms, and therapeutic interventions are advised to forestall a deeper embrace of the dark side. Subsequent evaluations will be pivotal in gauging the effectiveness of interventions and the overall trajectory of the individual's allegiance to the Force.",
        "dark_side_dx": "positive",
    }
])
```

## Register the sample data as a bardi dataset

Now that we have some sample data in a DataFrame we will register it as a bardi dataset.

```python
from bardi.data import data_handlers

# register a dataset
dataset = data_handlers.from_pandas(df)
```

When data is registered as a bardi dataset, the data is converted into a PyArrow Table. This is required to use the steps we have built.

## Initialize a pre-processing pipeline

Now that we have the data registered, let's set up a pipeline to pre-process the data.

```python
from bardi.pipeline import Pipeline

# initialize a pipeline
pipeline = Pipeline(dataset=dataset, write_outputs=False)
```

In this example we set write_outputs to False, however if you wanted to save the pipeline results to a file you would handle that here at the pipeline creation step (reference the documentation linked above).

So, now we have a pipeline initialized with a dataset, but the pipeline doesn't have any steps in it. Let's look at how we could add some steps.

A common pipeline could involve:
* cleaning/normalizing the text (Normalizer)
* splitting text into a list of tokens (PreTokenizer)
* generating a vocab and training word embeddings with Word2Vec (EmbeddingGenerator)
* mapping the list of tokens to a list of ints with the generated vocab (PostProcessor)
* mapping labels to ints (LabelProcessor)
* splitting the dataset into train/test/val splits (Splitter)

Please note, you do not need to add all of these steps if your use case does not require them. 

## Adding a Normalizer to our pipeline

The normalizer's key functionality is applying a set of regular expression substitutions to text. The normalizer also handles lowercasing text if desired. We need to specify the "fields" (AKA column names in the data) that we want the regular expression substitutions to be applied to. Then, we need to supply a set of regular expression substitutions to be performed. For this example we will supply a pre-built set of regular expressions, however a custom set could be created and supplied as well. Finally, we need to specify if we want the text to be lowercased.

```python
from bardi import nlp_engineering
from bardi.nlp_engineering.regex_library.pathology_report import PathologyReportRegexSet

# grabbing a pre-made regex set for normalizing pathology reports
path_report_regex_set = PathologyReportRegexSet().get_regex_set()

# adding the normalizer step to the pipeline
pipeline.add_step(nlp_engineering.CPUNormalizer(fields=['text'],
                                                regex_set=pathology_regex_set,
                                                lowercase=True))
```

## Adding a PreTokenizer to our pipeline

The pre-tokenizer is a pretty simple operation. We just need to specify the fields to apply the pre-tokenization operation to in addition to the pattern to split on. 

```python
# adding the pre-tokenizer step to the pipeline
pipeline.add_step(nlp_engineering.CPUPreTokenizer(fields=['text'],
                                                  split_pattern=' '))
```

## Adding an EmbeddingGenerator to our pipeline

Fair Warning: The embedding generator is by far the slowest part of the pipeline. It routinely accounts for about 95%+ of the total computation time. This is out of our control as we are just implementing Word2Vec. 

Many aspects of the Word2Vec implementation can be customized here, but in this example we are only changing the min_word_count (simply because our sample data in this tutorial is so small). Reference the documentation for a full list of customizations available in the CPUEmbeddingGenerator.

```python
# adding the embedding generator step to the pipeline
pipeline.add_step(nlp_engineering.CPUEmbeddingGenerator(fields=['text'],
                                                        min_word_count=2))
```

## Adding a PostProcessor to our pipeline

This step is a pretty simple one to add. There are more customizations possible if you are working with multiple text fields, but in this example we just have a single one. Reference the documentation if working with multiple text fields.

A key note is that there is an automatic renaming of the text field to 'X'. If you don't desire this behavior, you can set field_rename to a str of your desired column name. 

```python
# adding the post processor step to the pipeline
pipeline.add_step(nlp_engineering.CPUPostProcessor(fields=['text']))
```

## Adding a LabelProcessor to our pipeline

Again, a pretty straight-forward step. 

```python
# adding the label processor step to the pipeline
pipeline.add_step(nlp_engineering.CPULabelProcessor(fields=['dark_side_dx']))
```

## Running the pipeline

Now that we have added all of the steps, let's actually run the pipeline.

```python
# run the pipeline
pipeline.run_pipeline()
```

Since we set write_outputs to False at the initialization of the pipeline, we will need to grab our results at the end, too. If we had set it to True, then artifacts and data produced by the pipeline would just be saved in a file where we specified.

```python
# grabbing the data
final_data = pipeline.processed_data.to_pandas()

# grabbing the artifacts
vocab = pipeline.artifacts['id_to_token']
label_map = pipeline.artifacts['id_to_label']
word_embeddings = pipeline.artifacts['embedding_matrix']
```

## Results

Data:

| patient_id_number | X | dark_side_dx |
|---| --- | --- |
| 1 | [39, 33, 45, 44, 45, 45, 23, 45, 45, 45, 2, 22... | 1 |
| 2 | [33, 45, 30, 45, 31, 45, 41, 39, 12, 35, 34, 7... | 0 |
| 3 | [39, 24, 45, 20, 2, 22, 5, 1, 45, 13, 18, 45, ... | 1 |


Vocab:
```python
{0: '<pad>', 1: 'a', 2: 'aggression', 3: 'alignment', 4: 'an', 5: 'and', 6: 'any', 7: 'assessments', 8: 'be', 9: 'code', 10: 'connection', 11: 'count', 12: 'dark', 13: 'deviation', 14: 'examinations', 15: 'exposure', 16: 'force', 17: 'force.', 18: 'from', 19: 'further', 20: 'heightened', 21: 'imperative', 22: 'impulsivity', 23: 'in', 24: 'individual', 25: 'individuals', 26: 'interventions', 27: 'is', 28: 'jedi', 29: 'midichlorian', 30: 'no', 31: 'of', 32: 'overall', 33: 'patient', 34: 'preliminary', 35: 'side', 36: 'sith', 37: 'symptoms', 38: 'teachings', 39: 'the', 40: 'therapeutic', 41: 'to', 42: 'trajectory', 43: 'will', 44: 'with', 45: '<unk>'}
```

Label Map:
```python
{'dark_side_dx': {'0': 'negative', '1': 'positive'}}
```

Embedding Matrix:

```python
[[ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
   0.00000000e+00  0.00000000e+00]
 [ 1.77135365e-03 -5.86092880e-04  1.89334818e-03 ...  2.73368554e-03
   8.46754061e-04  3.34021775e-03]
 [-3.38128232e-03  1.09578541e-03  1.56378723e-03 ...  3.29070841e-03
  -1.36099930e-03 -8.10196943e-05]
 ...
 [ 1.00287900e-03  1.46343326e-03 -1.30044727e-03 ... -5.16163127e-04
  -1.43721746e-03 -8.17491091e-04]
 [ 2.52751313e-04  3.05728725e-04 -2.67492444e-03 ... -7.12162175e-04
   3.62762087e-03 -8.12349084e-04]
 [ 6.75368562e-03  5.78313626e-03  9.81814841e-05 ...  4.88654257e-03
   2.93711794e-03  4.90082072e-03]]
```

## Collecting metadata

Nothing we have implemented in this pipeline is particularly revolutionary in and of itself. We provide a handful of abstractions for dealing with text in an ML workflow, but a key objective is to provide these features within a reproducible framework. Everything we did above is automatically recorded by the pipeline so that the operations can be tracked and reproduced. Let's observe this behavior below.

```python
# reviewing the collected metadata
metadata = pipeline.get_parameters()

print(metadata)
```

Result:
```python
{
    "dataset": {
        "<class 'bardi.data.data_handlers.Dataset'>": {
            "date": "2023-12-08 16:10:59.173578",
            "data": ["patient_id_number", "text", "dark_side_dx"],
            "origin_query": "None",
            "origin_format": "pandas",
            "origin_row_count": 3,
        }
    },
    "steps": {
        "<class 'bardi.nlp_engineering.normalizer.CPUNormalizer'>": {
            "fields": ["text"],
            "_data_write_config": {
                "data_format": "parquet",
                "data_format_args": {"compression": "snappy", "use_dictionary": False},
            },
            "lowercase": True,
            "regex_set": [
                {"regex_str": "(\\\\x[0-9A-Fa-f]{2,})|\\\\[stepr]", "sub_str": " "},
                {"regex_str": "[\\r\\n\\t]|\\s{2,}", "sub_str": " "},
                {
                    "regex_str": "\\b(http[s]*:\\/\\/)[^\\s]+|\\b(www\\.)[^\\s]+",
                    "sub_str": " URLTOKEN ",
                },
                {
                    "regex_str": "[\\\\\\_,\\(\\);\\[\\]#{}\\*\"\\'\\~\\?!\\|\\^`]",
                    "sub_str": " ",
                },
                {"regex_str": "[\\-\\.:\\/\\_]{2,}", "sub_str": " "},
                {"regex_str": "<(.*?)>", "sub_str": " $1 "},
                {"regex_str": "%", "sub_str": " percent "},
                {"regex_str": "(\\b\\d{1,})([\\-\\.:])([a-z]+)", "sub_str": " $1 $3 "},
                {"regex_str": "(\\s[\\.:\\-\\\\])([^\\s]+)", "sub_str": " $2 "},
                {"regex_str": "([^\\s]+)([\\.:\\-\\\\]\\s)", "sub_str": " $1 "},
                {
                    "regex_str": "([a-z0-9]{2,})([\\-:\\.])([a-z]{2,})",
                    "sub_str": "$1 $3",
                },
                {"regex_str": "([><=+%\\/&:])", "sub_str": " $1 "},
                {"regex_str": "(\\d+[.\\d]*)([x])", "sub_str": "$1 $2 "},
                {"regex_str": "(\\d+)[-]*([cpamt][mlhc])", "sub_str": "$1 $2 "},
                {
                    "regex_str": "(\\d{1,2}[a-z])(-)(\\d{1,2}[a-z])|([a-z]\\d{1,2})(-)([a-z]\\d{1,2})",
                    "sub_str": "$1 $2 $3 ",
                },
                {
                    "regex_str": "( [\\d+]*[\\.:]*\\d+\\s*)(-)(\\s*[\\d+]*[\\.:]*\\d+)",
                    "sub_str": "$1 $2 $3",
                },
                {
                    "regex_str": "([a-z]{1,2})(\\d+\\.\\d+)([a-z]+)",
                    "sub_str": "$1$2 $3",
                },
                {"regex_str": "(\\b[a-z]+)(\\s+)([s]\\s)", "sub_str": "$1$3"},
                {"regex_str": "(\\s\\d{1,})([a-z]{2,}\\s)", "sub_str": "$1 $2"},
                {
                    "regex_str": "\\(*\\d{3}\\)*[-, ]*\\d{3}[-, ]*\\d{4}",
                    "sub_str": " PHONENUMTOKEN ",
                },
                {
                    "regex_str": "\\d{1,2}\\s*[\\/,-\\.]\\s*\\d{1,2}\\s*[\\/,-\\.]\\s*\\d{2,4}\\s*[at\\s\\-]*[\\d{1,2}\\s*[:\\s*\\d{1,2}]+]*(?:\\s*[pa][m])*|\\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\\s*\\d{1,2}\\s*\\d{2,4}|\\b\\d{1,2}\\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\\s*\\d{2,4}|\\d{1,2}-(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)-\\d{2}\\s*\\d{1,2}[:\\d{1,2}]+(?:\\s*[pa][m])",
                    "sub_str": " DATETOKEN ",
                },
                {
                    "regex_str": "(\\d{1,2}\\s*([:.]\\s*\\d{2}){1,2}\\s*[ap]\\.*[m]\\.*)|\\d{2}\\s*[ap]\\.*[m]\\.*|[0-2][0-9]:[0-5][1-9]",
                    "sub_str": " TIMETOKEN ",
                },
                {
                    "regex_str": "\\d+\\s([0-9a-z.]+[\\s,]+){1,6}[a-z]{2}[./\\s+]*\\d{5}(-\\d{4})*",
                    "sub_str": " ADDRESSTOKEN ",
                },
                {
                    "regex_str": "\\d+\\.*\\d*\\s*x\\s*\\d+\\.*\\d*\\s*x\\s*\\d+\\.*\\d*|\\d+\\.*\\d*\\s*x\\s*\\d+\\.*\\d*",
                    "sub_str": " DIMENSIONTOKEN ",
                },
                {
                    "regex_str": "[a-z]{1,3}[-]*\\d{2}[-]\\d{3,}[-]*",
                    "sub_str": " SPECIMENTOKEN ",
                },
                {
                    "regex_str": "\\d+[\\.\\-]\\d+([\\.\\-]\\d+)+",
                    "sub_str": " DECIMALSEGMENTEDNUMBERTOKEN ",
                },
                {"regex_str": "\\s\\d{3,}\\s", "sub_str": " DIGITSEQUENCETOKEN "},
                {"regex_str": "\\s\\d{2,}\\.\\d{1,}", "sub_str": " LARGEFLOATTOKEN "},
                {"regex_str": "\\s(\\d+)(\\.)(\\d)(\\d+)*\\s", "sub_str": " $1$2$3 "},
                {
                    "regex_str": "\\s\\d{1,2}[\\-]*[a-z]{1,2}\\s|\\b[a-z][\\-]*\\d{1}\\s|\\s[a-z]\\d{1,2}-\\d{1,2}\\s",
                    "sub_str": " CASSETTETOKEN ",
                },
                {
                    "regex_str": " \\d{1,2}d\\d{6,9}[.\\s]*",
                    "sub_str": " DURATIONTOKEN ",
                },
                {
                    "regex_str": "\\b[a-z]\\d{6,10}[.\\s]*",
                    "sub_str": " LETTERDIGITSTOKEN ",
                },
                {"regex_str": "\\s{2,}|\\\\n", "sub_str": " "},
            ],
        },
        "<class 'bardi.nlp_engineering.pre_tokenizer.CPUPreTokenizer'>": {
            "fields": ["text"],
            "split_pattern": " ",
            "_data_write_config": {
                "data_format": "parquet",
                "data_format_args": {"compression": "snappy", "use_dictionary": False},
            },
        },
        "<class 'bardi.nlp_engineering.embedding_generator.CPUEmbeddingGenerator'>": {
            "fields": ["text"],
            "cores": 10,
            "min_word_count": 2,
            "window": 5,
            "vector_size": 300,
            "sample": 6e-05,
            "min_alpha": 0.007,
            "negative": 20,
            "epochs": 30,
            "seed": 42,
            "vocab_exclude_list": [],
            "_data_write_config": {
                "data_format": "parquet",
                "data_format_args": {"compression": "snappy", "use_dictionary": False},
            },
            "_artifacts_write_config": {
                "vocab_format": "json",
                "vocab_format_args": {},
                "embedding_matrix_format": "npy",
                "embedding_matrix_format_args": {},
            },
            "w2v_model": "<class 'gensim.models.word2vec.Word2Vec'>",
            "vocab_size": 46,
        },
        "<class 'bardi.nlp_engineering.post_processor.CPUPostProcessor'>": {
            "fields": ["text"],
            "field_rename": "X",
            "_data_write_config": {
                "data_format": "parquet",
                "data_format_args": {"compression": "snappy", "use_dictionary": False},
            },
            "unk_id": 45,
        },
        "<class 'bardi.nlp_engineering.label_processor.CPULabelProcessor'>": {
            "fields": ["dark_side_dx"],
            "method": "unique",
            "_data_write_config": {
                "data_format": "parquet",
                "data_format_args": {"compression": "snappy", "use_dictionary": False},
            },
            "_artifacts_write_config": {
                "id_to_label_format": "json",
                "id_to_label_format_args": {},
            },
        },
    },
    "performance": {
        "<class 'bardi.nlp_engineering.normalizer.CPUNormalizer'>": {
            "time": "0:00:00.008010",
            "memory (MB)": "0.013305",
        },
        "<class 'bardi.nlp_engineering.pre_tokenizer.CPUPreTokenizer'>": {
            "time": "0:00:00.000863",
            "memory (MB)": "0.003406",
        },
        "<class 'bardi.nlp_engineering.embedding_generator.CPUEmbeddingGenerator'>": {
            "time": "0:00:00.074747",
            "memory (MB)": "0.531624",
        },
        "<class 'bardi.nlp_engineering.post_processor.CPUPostProcessor'>": {
            "time": "0:00:00.003835",
            "memory (MB)": "0.03622",
        },
        "<class 'bardi.nlp_engineering.label_processor.CPULabelProcessor'>": {
            "time": "0:00:00.001360",
            "memory (MB)": "0.008777",
        },
        "<class 'bardi.pipeline.Pipeline'>": "0:00:00.088891",
    },
}
```

## Full tutorial script

```python
import pandas as pd
from bardi.data import data_handlers
from bardi.pipeline import Pipeline
from bardi import nlp_engineering
from bardi.nlp_engineering.splitter import NewSplit
from bardi.nlp_engineering.regex_library.pathology_report import PathologyReportRegexSet

# create some sample data
df = pd.DataFrame([
    {
        "patient_id_number": 1,
        "text": "The patient presented with notable changes in behavior, exhibiting increased aggression, impulsivity, and a distinct deviation from the Jedi Code. Preliminary examinations reveal a heightened midichlorian count and an unsettling connection to the dark side of the Force. Further analysis is warranted to explore the extent of exposure to Sith teachings. It is imperative to monitor the individual closely for any worsening symptoms and to engage in therapeutic interventions aimed at preventing further descent into the dark side. Follow-up assessments will be crucial in determining the efficacy of intervention strategies and the overall trajectory of the individual's alignment with the Force.",
        "dark_side_dx": "positive",
    },
    {
        "patient_id_number": 2,
        "text": "Patient exhibits no signs of succumbing to the dark side. Preliminary assessments indicate a stable midichlorian count and a continued commitment to Jedi teachings. No deviations from the Jedi Code or indicators of dark side influence were observed. Regular check-ins with the Jedi Council will ensure the sustained well-being and alignment of the individual within the Jedi Order.",
        "dark_side_dx": "negative",
    },
    {
        "patient_id_number": 3,
        "text": "The individual manifested heightened aggression, impulsivity, and a palpable deviation from established ethical codes. Initial examinations disclosed an elevated midichlorian count and an unmistakable connection to the dark side of the Force. Further investigation is imperative to ascertain the depth of exposure to Sith doctrines. Close monitoring is essential to track any exacerbation of symptoms, and therapeutic interventions are advised to forestall a deeper embrace of the dark side. Subsequent evaluations will be pivotal in gauging the effectiveness of interventions and the overall trajectory of the individual's allegiance to the Force.",
        "dark_side_dx": "positive",
    }
])

# register a dataset
dataset = data_handlers.from_pandas(df)

# initialize a pipeline
pipeline = Pipeline(dataset=dataset, write_outputs=False)

# grabbing a pre-made regex set for normalizing pathology reports
pathology_regex_set = PathologyReportRegexSet().get_regex_set()

# adding the normalizer step to the pipeline
pipeline.add_step(nlp_engineering.CPUNormalizer(fields=['text'],
                                                regex_set=pathology_regex_set,
                                                lowercase=True))

# adding the pre-tokenizer step to the pipeline
pipeline.add_step(nlp_engineering.CPUPreTokenizer(fields=['text'],
                                                  split_pattern=' '))

# adding the embedding generator step to the pipeline
pipeline.add_step(nlp_engineering.CPUEmbeddingGenerator(fields=['text'],
                                                        min_word_count=2))

# adding the post processor step to the pipeline
pipeline.add_step(nlp_engineering.CPUPostProcessor(fields=['text']))

# adding the label processor step to the pipeline
pipeline.add_step(nlp_engineering.CPULabelProcessor(fields=['dark_side_dx']))

# run the pipeline
pipeline.run_pipeline()

# grabbing the data
final_data = pipeline.processed_data.to_pandas()

# grabbing the artifacts
vocab = pipeline.artifacts['id_to_token']
label_map = pipeline.artifacts['id_to_label']
word_embeddings = pipeline.artifacts['embedding_matrix']

print(final_data)
print(vocab)
print(label_map)
print(word_embeddings)

# reviewing the collected metadata
metadata = pipeline.get_parameters()

print(metadata)
```
