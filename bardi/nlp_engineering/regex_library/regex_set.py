"""Define RegexSet and RegexSubPair blueprints"""

from typing import List, TypedDict


class RegexSubPair(TypedDict):
    """Dictionary used for regular expression string substitutions

    Attributes:
        regex_str: regular expression pattern
        sub_str: replacement value for matched string

    Example:
        {"regex_str": r"\\s",
         "sub_str": "WHITESPACE"}
    """

    regex_str: str
    sub_str: str


class RegexSet:
    """Blueprint for creating a configurable, domain specific regular expression set"""

    def __init__(self):
        self.regex_set: List[RegexSubPair] = []

    def get_regex_set(self) -> List[RegexSubPair]:
        """Return the ordered set of regular expressions"""
        return self.regex_set
