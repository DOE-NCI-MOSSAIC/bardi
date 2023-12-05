"""Blueprint for creating a configurable, domain specific regular expression set"""

from abc import abstractmethod
from typing import List

from bardi.nlp_engineering.regex_library.regex_lib import RegexSubPair


class RegexSet:
    def __init__(self):
        self.regex_set: List[RegexSubPair] = []

    @abstractmethod
    def get_regex_set(self) -> List[RegexSubPair]:
        pass
