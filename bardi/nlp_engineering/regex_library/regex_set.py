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

    def get_regex_set(
        self, lowercase_substitution=False, no_substitution=False
    ) -> List[RegexSubPair]:
        """Return the ordered set of regular expressions

        Attributes
        ----------

        lowercase_substitution : Optional[bool]
            It True all the substitution tokeen like DATETOKEN will be returned
            in lowercase `datetoken`. Defaults to False.
        no_substitution : Optionl[bool]
            If True all the regular expression that remove matched pattern will replace it
            with space instead of special token. Defaults to False.

        Returns
        -------

        List[RegexSubPair]
            a list of regular expression substitution pairs
        """
        if no_substitution:
            for regex_pair in self.regex_set:
                if "TOKEN" in regex_pair["sub_str"]:
                    regex_pair["sub_str"] = " "

        elif lowercase_substitution:
            for regex_pair in self.regex_set:
                regex_pair["sub_str"] = regex_pair["sub_str"].lower()

        return self.regex_set
