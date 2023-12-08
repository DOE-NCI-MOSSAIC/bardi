import unittest
import re
import random
from pathlib import Path

import pandas as pd

from bardi.nlp_engineering.regex_library import regex_lib


class TestRegexMultipleExpreesions(unittest.TestCase):
    """Tests the regex library by applying the regex method
    in sequence and printing the results to screen.."""
    def setUp(self):
        # Set up paths
        repo_path = Path().resolve()
        self.data_path = (f'{repo_path}/tests/test_data/'
                          f'recurrence_raw_data_sample.pkl')

        # Get data
        self.data = pd.read_pickle(self.data_path)
        self.data.reset_index(inplace=True)
        self.max_int = self.data.shape[0]

        self.lowercase = True
        self.handle_whitespaces = True  # 1
        self.remove_urls = True  # 2
        self.remove_special_punct = True  # 3
        self.remove_multiple_punct = True  # 4
        self.handle_angle_brackets = True  # 5

        self.replace_percent_sign = True  # 6
        self.handle_leading_digit_punct = True  # 7
        self.remove_leading_punct = True  # 8
        self.remove_trailing_punct = True  # 9
        self.handle_words_with_punct_spacing = True  # 10

        self.handle_math_spacing = True  # 11
        self.handle_dimension_spacing = True  # 12
        self.handle_measure_spacing = True  # 13
        self.handle_cassettes_spacing = True  # 14
        self.handle_dash_digits_spacing = True  # 15

        self.handle_literals_floats_spacing = True  # 16
        self.fix_pluralization = True  # 17
        self.handle_digits_words_spacing = True  # 18
        self.remove_phone_numbers = True  # 19
        self.remove_dates = True  # 20

        self.remove_time = True  # 21
        self.remove_addresses = True  # 22
        self.remove_dimensions = True  # 23
        self.remove_specimen = True  # 24
        self.remove_decimal_seg_numbers = True  # 25
        self.remove_large_digits_seq = True  # 26
        self.remove_large_floats_seq = True  # 27
        self.trunc_decimals = True  # 28
        self.remove_cassette_names = True  # 29

    def test_single(self):
        x = random.randint(0, self.max_int)
        test_text = self.data["text_all"][x].lower()

        regex_sub_pair = regex_lib.get_escape_code_regex()
        pattern = regex_sub_pair["regex_str"]
        replacement = regex_sub_pair["sub_str"]
        print(f'\n Rule 0: Escape Codes -  pattern: {pattern}'
              'replacement: {replacement}\n')
        test_text = re.sub(pattern, replacement, test_text)
        original_text = test_text
        print(test_text)

        if self.handle_whitespaces:
            regex_sub_pair = regex_lib.get_whitespace_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 1 White Spaces: pattern: {pattern} replacement:'
                  f' {replacement} \n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.remove_urls:
            regex_sub_pair = regex_lib.get_urls_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 2 URLs Replacement: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.remove_special_punct:
            regex_sub_pair = regex_lib.get_special_punct_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 3 Chosen Punctuation Removal: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.remove_multiple_punct:
            regex_sub_pair = regex_lib.get_multiple_punct_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 4 Multiple Punctuation: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.handle_angle_brackets:
            regex_sub_pair = regex_lib.get_angle_brackets_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 5 Angle Brackets Removal: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.replace_percent_sign:
            regex_sub_pair = regex_lib.get_percent_sign_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 6 Replace Percent Sign: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.handle_leading_digit_punct:
            regex_sub_pair = regex_lib.get_leading_digit_punctuation_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 7 Leading Digit Punctuation: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.remove_leading_punct:
            regex_sub_pair = regex_lib.get_leading_punctuation_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 8 Leading Punctuation: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.remove_trailing_punct:
            regex_sub_pair = regex_lib.get_trailing_punctuation_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 9 Trailing Punctuation: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.handle_words_with_punct_spacing:
            regex_sub_pair = regex_lib.get_words_with_punct_spacing_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 10 Words with Punctuation: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.handle_math_spacing:
            regex_sub_pair = regex_lib.get_math_spacing_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 11 Math Operator Spacing: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.handle_dimension_spacing:
            regex_sub_pair = regex_lib.get_dimension_spacing_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 12 Dimension spacing: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.handle_measure_spacing:
            regex_sub_pair = regex_lib.get_measure_spacing_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 13 Measure spacing: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.handle_cassettes_spacing:
            regex_sub_pair = regex_lib.get_cassettes_spacing_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 14 Special Specing: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.handle_dash_digits_spacing:
            regex_sub_pair = regex_lib.get_dash_digits_spacing_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 15 Dash Spacing: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.handle_literals_floats_spacing:
            regex_sub_pair = regex_lib.get_literals_floats_spacing_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 16 Literal Floats: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.fix_pluralization:
            regex_sub_pair = regex_lib.get_fix_pluralization_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 17 Plurals Attach: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.handle_digits_words_spacing:
            regex_sub_pair = regex_lib.get_digits_words_spacing_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 18 Digits Words Spacing: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.remove_phone_numbers:
            regex_sub_pair = regex_lib.get_phone_number_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 19 Phone Number Removal: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.remove_dates:
            regex_sub_pair = regex_lib.get_dates_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 20 Dates Removal: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.remove_time:
            regex_sub_pair = regex_lib.get_time_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 21 Time Removal: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.remove_addresses:
            regex_sub_pair = regex_lib.get_address_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 22 Address Removal: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.remove_dimensions:
            regex_sub_pair = regex_lib.get_dimensions_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 23 Dimension Removal: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.remove_specimen:
            regex_sub_pair = regex_lib.get_specimen_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 24 Specimen Removal: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.remove_decimal_seg_numbers:
            regex_sub_pair = regex_lib.get_decimal_segmented_numbers_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 25 Decimal Segmented Numbers: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.remove_large_digits_seq:
            regex_sub_pair = regex_lib.get_large_digits_seq_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 26 Large Digits: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.remove_large_floats_seq:
            regex_sub_pair = regex_lib.get_large_float_seq_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 27 Large Floats: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.trunc_decimals:
            regex_sub_pair = regex_lib.get_trunc_decimals_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 28 Truncate Decimal: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.remove_cassette_names:
            regex_sub_pair = regex_lib.get_cassette_name_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 29 Cassette Names Removal: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        regex_sub_pair = regex_lib.get_spaces_regex()
        pattern = regex_sub_pair["regex_str"]
        replacement = regex_sub_pair["sub_str"]
        print(f'\n Rule 18 Additional Spaces: pattern: {pattern}'
              f'replacement: {replacement}\n')
        test_text = re.sub(pattern, replacement, test_text)
        print(test_text)

        print("********   ORIGINAL TEXT   ********")
        print(original_text)
        print("********   AFTER   ********")
        print(test_text)
        print(f'INDEX : {x}')


if __name__ == '__main__':
    unittest.main()
