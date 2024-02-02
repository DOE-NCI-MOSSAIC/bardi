======================
Advanced Package Usage
======================

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

        from bardi.nlp_engineering.regex_library.pathology_report import PathologyReportRegexSet

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

        from bardi.nlp_engineering.regex_library.pathology_report import PathologyReportRegexSet

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

2. **Using Python Built-in Types to Write a Custom Set**

    Nothing is stopping you from writing your own set of regular expressions and providing that to the normalizer. ::

        # Creating custom regular expression set
        custom_regex_set = [
            {"regex_str": '\s', "sub_str": ''}, 
            {"regex_str": 'http[s]?\://\S+', "sub_str": 'URLTOKEN'}
        ]

        # Supplying custom regex set to Normalizer
        pipeline.add_step(
            nlp_engineering.CPUNormalizer(
                fields=['text'],
                regex_set=custom_regex_set,
                lowercase=True
            )
        )

3. **Using the bardi RegexSet Class to Write a Custom Set**

    However, if you would like a more organized approach [than the method shown above] to handle your custom regular expression sets, 
    you could follow our lead by creating a new class and inherit from our RegexSet class. The RegexSet class you are inheriting 
    from gives you an attribute `regex_set`, which is a list of your regular expression substitution pairs, and the method to 
    return this attribute, `get_regex_set()` ::

        from bardi.nlp_engineering.regex_library.regex_set import RegexSet


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
            nlp_engineering.CPUNormalizer(
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
reproducibility. ::

    from bardi import Step

    # initialize a pipeline
    pipeline = Pipeline(dataset=dataset, write_outputs=False)

    # adding a normalizer step to the pipeline
    pipeline.add_step(
        nlp_engineering.CPUNormalizer(
            fields=['text'],
            regex_set=pathology_regex_set,
            lowercase=True
        )
    )

    # creating a custom step
    class MyCustomStep(Step):

        def run(
            self, data: pa.Table, artifacts: Optional[dict] = None
        ) -> Tuple[pa.Table, dict]:
            
            df = pl.from_arrow(data)

            

            data = df.to_arrow()

            return data

    # adding the custom step to the pipeline
    pipeline.add_step(
        MyCustomStep()
    )

    # run the pipeline with the provided step and the custom step
    pipeline.run_pipeline()
                                            


