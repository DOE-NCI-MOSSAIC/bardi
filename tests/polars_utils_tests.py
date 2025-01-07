"""Tests for NLP Engineering Utils"""

import unittest

import polars as pl
from polars.testing import assert_frame_equal

from bardi.nlp_engineering.utils import polars_utils


class TestPolarsUtils(unittest.TestCase):
    """Tests the polars utils"""

    def setUp(self):
        """Execute the common setup code needed
        for all of the polars utils tests"""
        # Mock polars DataFrame to test the correct behaviour.
        self.df = pl.DataFrame(
            {
                "text_1": "At 1234 north 500 west provo ca 12345.\n The speciment:"
                "sh-22-0011300 3.89 x4.56cm. Or 4.3 km. ",
                "text_2": "Call  123 456 7890 .This is: 0.8943",
            }
        )

    def test_retain_inputs(self):
        self.df = self.df.pipe(
            polars_utils.retain_inputs,
            retain_input_fields=True,
            fields=["text_1", "text_2"],
            step_name="test",
        )

        self.correct_df = pl.DataFrame(
            {
                "text_1": "At 1234 north 500 west provo ca 12345.\n The speciment:"
                "sh-22-0011300 3.89 x4.56cm. Or 4.3 km. ",
                "text_2": "Call  123 456 7890 .This is: 0.8943",
                "test_input__text_1": "At 1234 north 500 west provo ca 12345.\n The speciment:"
                "sh-22-0011300 3.89 x4.56cm. Or 4.3 km. ",
                "test_input__text_2": "Call  123 456 7890 .This is: 0.8943",
            }
        )

        assert_frame_equal(self.df, self.correct_df)


if __name__ == "__main__":
    unittest.main()
