"""Define RegexSet and RegexSubPair blueprints"""

from typing import List, TypedDict


class RegexSubPair(TypedDict):
    """Dictionary used for regular expression string substitutions

    Example of a regex sub pair dictionary: ::
    
        {
            "regex_str": r"\\s",
            "sub_str": "WHITESPACE"
        }

    Attributes
    ----------

    regex_str : str 
        regular expression pattern
    sub_str : str
        replacement value for matched string
    """

    regex_str: str
    sub_str: str


class RegexSet:
    """Blueprint for creating a configurable, domain specific regular expression set
    
    Attributes
    ----------

    regex_set : List[RegexSubPair]
        a list of regular expression substitution pairs
    """

    def __init__(self):
        self.regex_set: List[RegexSubPair] = []

    def get_regex_set(self) -> List[RegexSubPair]:
        """Return the ordered set of regular expressions
        
        Returns
        -------

        List[RegexSubPair]
            a list of regular expression substitution pairs
        """
        return self.regex_set
