==================
Advanced Tutorials
==================

`Note: all code snippets on this page are based on setup performed in the basic tutorial. As such, code from the snippets may
not run exactly as-is without adding some of the setup steps such as creating a dataset and pipeline.`

Normalizer Customizations
-------------------------

Documentation References:
:mod:`bardi.nlp_engineering.normalizer.CPUNormalizer`
:mod:`bardi.nlp_engineering.regex_library.pathology_report.PathologyReportRegexSet`

The normalizer needs a regular expression set (regex_set) to apply to the text. A regex_set is just a Python list of dictionaries
containing a key for the regular expression, `regex_str`, and a key for the substitution string, `sub_str`. There are a few methods you 
can use to customize what regular expressions are applied to your text.

1. **Tuning a Provided bardi RegexSet**

    An existing RegexSet is tunable by turning individual regular expressions off and on as desired. By default, all are
    turned on, so you would just need to turn off the ones you don't want to use. ::

        from bardi.nlp_engineering import PathologyReportRegexSet

        # grabbing a pre-made regex set for normalizing pathology reports
        # turning off 3 of the regular expression substitutions I don't want to use
        path_report_regex_set = PathologyReportRegexSet().get_regex_set(
            remove_dimensions=False,
            remove_specimen=False,
            remove_decimal_seg_number=False
        )

2. **Append to a Provided bardi RegexSet**

    Suppose you want to use a provided regular expression set, but you have some additional regular expressions 
    you would like to add. Because the regular expression set just returns a Python list, you can just append 
    your additional regular expressions to the list. ::

        from bardi.nlp_engineering import PathologyReportRegexSet

        # grabbing a pre-made regex set for normalizing pathology reports
        # turning off 3 of the regular expression substitutions I don't want to use
        path_report_regex_set = PathologyReportRegexSet().get_regex_set(
            remove_dimensions=False,
            remove_specimen=False,
            remove_decimal_seg_number=False
        )

        # Adding a custom regular expression substitution pair
        path_report_regex_set.append(
            {"regex_str": '$[0-9]+', "sub_str": 'MONEYTOKEN'}
        )

3. **Using Python Built-in Types to Write a Custom Set**

    Nothing is stopping you from writing your own set of regular expressions and providing that to the normalizer. ::

        # Creating custom regular expression set
        custom_regex_set = [
            {"regex_str": '\s', "sub_str": ''}, 
            {"regex_str": 'http[s]?\://\S+', "sub_str": 'URLTOKEN'}
        ]

        # Supplying custom regex set to Normalizer
        pipeline.add_step(
            nlp.CPUNormalizer(
                fields=['text'],
                regex_set=custom_regex_set,
                lowercase=True
            )
        )

4. **Using the bardi RegexSet Class to Write a Custom Set**

    However, if you would like a more organized approach [than the method shown above] to handle your custom regular expression sets, 
    you could follow our lead by creating a new class and inherit from our RegexSet class. The RegexSet class you are inheriting 
    from gives you an attribute `regex_set`, which is a list of your regular expression substitution pairs, and the method to 
    return this attribute, `get_regex_set()` ::

        from bardi.nlp_engineering import RegexSet


        # Creating a custom regex set class inheriting from bardi RegexSet
        class MyCustomRegexSet(RegexSet):
            def __init__(
                self,
                handle_whitespace: bool = True,
                convert_urls: bool = True
            ):
                if handle_whitespace:
                    self.regex_set.append(
                        {"regex_str": '\s', "sub_str": ''}
                    )
                if convert_urls:
                    self.regex_set.append(
                        {"regex_str": 'http[s]?\://\S+', "sub_str": 'URLTOKEN'}
                    )
        

        # Supplying custom regex set to Normalizer
        pipeline.add_step(
            nlp.CPUNormalizer(
                fields=['text'],
                regex_set=MyCustomRegexSet.get_regex_set(),
                lowercase=True
            )
        )

Creating a Custom Pipeline Step
-------------------------------

Documentation References:
:mod:`bardi.pipeline.Step`

We have provided some out-of-the-box pipeline steps, and hope to continue adding more helpful steps, but we don't 
expect to have provided every possible data pre-processing action you could ever need. So, what if we don't have 
something built that you need, but you still would like to use the framework? Create your own custom step! 
By following these guidelines you can create a step that will run within our pipeline, alongside any of the steps 
we provide, and have all of your custom step's metadata captured in the standard pipeline metadata file for
reproducibility. 

Let's look at a full example, and then we'll explain what's happening.::

    from typing import Tuple, Optional

    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from bardi import nlp_engineering as nlp
    from bardi import data as bardi_data
    from bardi import Pipeline, Step

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

    # Register a dataset
    dataset = bardi_data.from_pandas(df)

    # Initialize a pipeline
    pipeline = Pipeline(dataset=dataset, write_outputs=False)

    # Grabbing a pre-made regex set for normalizing pathology reports
    pathology_regex_set = nlp.PathologyReportRegexSet().get_regex_set()

    # Adding a normalizer step to the pipeline
    pipeline.add_step(
        nlp.CPUNormalizer(
            fields=['text'],
            regex_set=pathology_regex_set,
            lowercase=True
        )
    )

    # Adding the pre-tokenizer step to the pipeline
    pipeline.add_step(
        nlp.CPUPreTokenizer(
            fields=['text'],
            split_pattern=' '
        )
    )

    # Creating a custom step to count the token length for each record
    class MyCustomStep(Step):

        def __init__(self, field: str):
            super().__init__()
            self.field = field

        def run(
            self, data: pa.Table, artifacts: Optional[dict] = None
        ) -> Tuple[pa.Table, dict]:

            df = pl.from_arrow(data).with_columns([
                pl.col(self.field).list.lengths().alias('token_count')
            ])

            data = df.to_arrow()

            return data, None

    # Adding the new custom step to the pipeline
    pipeline.add_step(
        MyCustomStep(field='text')
    )

    # Run the pipeline with the provided step and the custom step
    pipeline.run_pipeline()
                                            
In this example we initialized a pipeline and added a couple of the bardi provided Steps (CPUNormalizer and CPUPreTokenizer).
These steps are cleaning the text and splitting on spaces into lists of tokens.

Next, we create a custom Step. We do this by creating a new class and inherit from the bardi Step class. In the constructor
we take a `field` input so we know which column the user of the Step will want to perform the Step's action on. If we reference 
bardi's documentation for the Step class, we will see that we need to create a run() method that will accept two parameters:
a PyArrow.Table and a Python dict. The PyArrow Table is the data being passed from Step to Step in the Pipeline for various
transformations. The dict is the dictionary of artifacts that can be used to pass artifacts from Step to Step in the Pipeline. 
If you don't have the need to work with artifacts in the custom Step, you can just ignore it, but it still needs to be listed in 
run method arguments for the Pipeline to work correctly. 

Similarly, the return from the run method needs to be a Tuple with the first element being data in a PyArrow table and the 
second element being a dict of artifacts produced in the Step. If data isn't altered in your step, you can place None in the 
first position of the return Tuple. If artifacts aren't produced in your step, you can place None in the second position of 
the return Tuple. The example shows that we altered the data and return the new table, but we didn't produce any artifacts so 
we None in the second position of the return Tuple. 

What happens inbetween the inputs and outputs of the run method are totally up to you. We used Polars in the example because it 
is built on Arrow, highly performant, and is what the rest of bardi is built with. However, if you don't know how to use Polars, you 
don't need to use it. If you want to use Pandas, for instance, just create a DataFrame from the PyArrow Table, perform your Pandas 
operations, and then convert your DataFrame back into a PyArrow Table for the return. 

After defining our custom step, we added it to the Pipeline and then ran it. 

The custom Step we created has some methods automatically provided by the Step class including data writing and getting parameters. 
These can be customized if needed, but the base implementation will probably be good enough for most uses. If you want to explore 
this more, refer to the Step documentation.
