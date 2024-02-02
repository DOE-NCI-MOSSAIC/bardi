"""Curated set of regular expressions for cleaning text from pathology reports."""

from typing import List

from bardi.nlp_engineering.regex_library import regex_lib
from bardi.nlp_engineering.regex_library.regex_set import RegexSet, RegexSubPair


class PathologyReportRegexSet(RegexSet):
    """The PathologyReportRegexSet includes set of standard
    regular expression to normalize a pathology report.

    Note
    ----
    
    The set of regular expressions tailored for pathology reports
    was crafted with the understanding that dividing text based
    on punctuation often results in the loss of crucial information.
    E.g. terms like "her-2" should not be split. However, to ensure
    that the number of unique tokens remains manageable we employ a number
    of regular expression to separate some tokens around punctuation.
    E.g. :code:`22-years` becomes :code:`22 years`.
    This consideration is particularly important when 
    employing the word2vec algorithm, as an excessive number of tokens
    can impede the model's effectiveness by diluting the representation of key concepts.

    Attributes
    ----------

    convert_escape_codes: bool
        Removes escape codes such as \\x0d, \\x0a, etc.
    handle_whitespaces: bool
        Removes extra whitespaces: any new line, carriage return tab.
    remove_urls: bool
        Removes URLs found in the text that match the pattern. 
    remove_special_punct: bool 
        Removes special punctuation like (?,$).
    remove_multiple_punct: bool
        Removes duplicated punctuation.
        E.g. :code:`---`
    handle_angle_brackets: bool
        Removes angle brackets.
        E.g. :code:`<title>` becomes :code:`title`.
    replace_percent_sign: bool
        Replaces a percent sign with a 'percent' word.
    handle_leading_digit_punct: bool
        Removes punctuation when digit is attached to word.
        E.g. :code:`22-years` becomes :code:`22 years`.
    remove_leading_punct: bool
        Removes leading punctuation from words.
        E.g. :code:`-result` becomes :code:`result`.    
    remove_trailing_punct: bool
        Removes trailing punctuation from words.
        E.g. :code:`result-` becomes :code:`result`.
    handle_words_with_punct_spacing: bool
        Matches words with hyphen, colon or period and splits them.
    handle_math_spacing: bool 
        Matches "math operators symbols" like ><=%: and adds spaces aroud them.
    handle_dimension_spacing: bool
        Matches digits and x and adds spaces between them.
    handle_measure_spacing: bool
        Matches measurements in mm, cm and ml provides proper spacing between the digits and measure.
    handle_cassettes_spacing: bool
        Matches patterns like 5e-6f and adds spaces around them.
    handle_dash_digit_spacing: bool
        Matches dashes around digits and adds spaces around the dashes.
    handle_literals_floats_spacing: bool
        Matches character followed by a float and a word.
        This is a common formating problem.
        E.g. :code:`r18.0admission` becomes :code:`r18.0 admission`.
    fix_pluralization: bool
        Matches s character after a word and attaches it back to the word.
        This restores plural nouns demages by removed punctuation.
    handle_digits_words_spacing: bool
        Matches digits that are attached to the beginning of a word.
    remove_phone_numbers: bool
        Matches any phone number that consists of 10 digits with delimeters.
    remove_dates: bool
        Removes dates of prespecified format.
    remove_times: bool
        Matches time of format 11:20 am or 1.30pm or 9:52:07AM.
    remove_addresses: bool
        Matches any address of format num (street name) in 1 to 6 words 2-letter state
        and short or long zip code.
    remove_dimensions: bool
        Matches 2D or 3D dimension measurements and adds spaces around them.
    remove_specimen: bool
        Matches marking of a pathology speciman.
    remove_decimal_seg_numbers: bool
        Matches combinations of digits and periods or dashes.
        E.g. :code:` 1.78.9.87`.
    remove_large_digits_seq: bool
        Matches large sequences of digits (3 or more) and replaces it.
    remove_large_floats_seq: bool
        Matches large floats and replace them.
    trunc_decimals: bool = True
        Matches floats and keeps only first decimal.
    remove_cassette_names: bool
        Removes pathology samples' markings.
        E.g. :code:`1-e`.
    remove_duration_time: bool
        Removes duration a speciment was treated.
        E.g. :code:`32d09090301`.
    remove_letter_num_seq: bool
        Removes a character followed directly by 6 to 10 digits.
      
    """
    def __init__(
        self,
        convert_escape_codes: bool = True,  # 0
        handle_whitespaces: bool = True,  # 1
        remove_urls: bool = True,  # 2
        remove_special_punct: bool = True,  # 3
        remove_multiple_punct: bool = True,  # 4
        handle_angle_brackets: bool = True,  # 5
        replace_percent_sign: bool = True,  # 6
        handle_leading_digit_punct: bool = True,  # 7
        remove_leading_punct: bool = True,  # 8
        remove_trailing_punct: bool = True,  # 9
        handle_words_with_punct_spacing: bool = True,  # 10
        handle_math_spacing: bool = True,  # 11
        handle_dimension_spacing: bool = True,  # 12
        handle_measure_spacing: bool = True,  # 13
        handle_cassettes_spacing: bool = True,  # 14
        handle_dash_digit_spacing: bool = True,  # 15
        handle_literals_floats_spacing: bool = True,  # 16
        fix_pluralization: bool = True,  # 17
        handle_digits_words_spacing: bool = True,  # 18
        remove_phone_numbers: bool = True,  # 19
        remove_dates: bool = True,  # 20
        remove_times: bool = True,  # 21
        remove_addresses: bool = True,  # 22
        remove_dimensions: bool = True,  # 23
        remove_specimen: bool = True,  # 24
        remove_decimal_seg_numbers: bool = True,  # 25
        remove_large_digits_seq: bool = True,  # 26
        remove_large_floats_seq: bool = True,  # 27
        trunc_decimals: bool = True,  # 28
        remove_cassette_names: bool = True,  # 29
        remove_duration_time: bool = True,  # 30
        remove_letter_num_seq: bool = True,
    ):
        # === List of regex sub pairs ===
        self.regex_set: List[RegexSubPair] = []

        # === Retrieve regular expression substitution pairs from regex_lib
        #  0 removes escapes codes
        if convert_escape_codes:
            self.regex_sub_escape_codes = regex_lib.get_escape_code_regex()
            self.regex_set.append(self.regex_sub_escape_codes)

        #  1 (regex, sub_str) for any new line, carriage return tab
        # and multiple spaces --> " "
        if handle_whitespaces:
            self.regex_sub_whitespaces = regex_lib.get_whitespace_regex()
            self.regex_set.append(self.regex_sub_whitespaces)

        #  2 removes URLs that start with https http or www
        if remove_urls:
            self.regex_sub_urls = regex_lib.get_urls_regex()
            self.regex_set.append(self.regex_sub_urls)

        #  3 matches a set of special punctuation
        # ,();[]#{}* --> " "
        if remove_special_punct:
            self.regex_sub_special_punct = regex_lib.get_special_punct_regex()
            self.regex_set.append(self.regex_sub_special_punct)

        #  4 matches multiple occurences of symbols like -, .and _
        if remove_multiple_punct:
            self.regex_sub_multiple_punct = regex_lib.get_multiple_punct_regex()
            self.regex_set.append(self.regex_sub_multiple_punct)

        #  5(regex, sub_str) removes angle brackets
        # <THIS IS INSIDE> --> THIS IS INSIDE
        if handle_angle_brackets:
            self.regex_sub_angle_brackets = regex_lib.get_angle_brackets_regex()
            self.regex_set.append(self.regex_sub_angle_brackets)

        #  6 (regex, sub_str) replaces % for a percent word 56% --> 56 PERCENT
        if replace_percent_sign:
            self.regex_sub_percent_sign = regex_lib.get_percent_sign_regex()
            self.regex_set.append(self.regex_sub_percent_sign)

        #  7
        if handle_leading_digit_punct:
            self.regex_sub_leading_digit_punct = (
                regex_lib.get_leading_digit_punctuation_regex()
            )
            self.regex_set.append(self.regex_sub_leading_digit_punct)

        #  8
        if remove_leading_punct:
            self.regex_sub_leading_punct = regex_lib.get_leading_punctuation_regex()
            self.regex_set.append(self.regex_sub_leading_punct)

        #  9
        if remove_trailing_punct:
            self.regex_sub_trailing_punct = regex_lib.get_trailing_punctuation_regex()
            self.regex_set.append(self.regex_sub_trailing_punct)

        #  10
        if handle_words_with_punct_spacing:
            self.regex_sub_words_with_punct_spacing = (
                regex_lib.get_words_with_punct_spacing_regex()
            )
            self.regex_set.append(self.regex_sub_words_with_punct_spacing)

        #  11
        if handle_math_spacing:
            self.regex_sub_math_spacing = regex_lib.get_math_spacing_regex()
            self.regex_set.append(self.regex_sub_math_spacing)

        #  12
        if handle_dimension_spacing:
            self.regex_sub_dimension_spacing = regex_lib.get_dimension_spacing_regex()
            self.regex_set.append(self.regex_sub_dimension_spacing)

        #  13
        if handle_measure_spacing:
            self.regex_sub_measure_spacing = regex_lib.get_measure_spacing_regex()
            self.regex_set.append(self.regex_sub_measure_spacing)

        #  14
        if handle_cassettes_spacing:
            self.regex_sub_cassette_spacing = regex_lib.get_cassettes_spacing_regex()
            self.regex_set.append(self.regex_sub_cassette_spacing)

        #  15
        if handle_dash_digit_spacing:
            self.regex_sub_dash_spacing = regex_lib.get_dash_digits_spacing_regex()
            self.regex_set.append(self.regex_sub_dash_spacing)

        #  16
        if handle_literals_floats_spacing:
            self.regex_sub_literals_floats_spacing = (
                regex_lib.get_literals_floats_spacing_regex()
            )
            self.regex_set.append(self.regex_sub_literals_floats_spacing)

        #  17
        if fix_pluralization:
            self.regex_sub_fix_pluralization = regex_lib.get_fix_pluralization_regex()
            self.regex_set.append(self.regex_sub_fix_pluralization)

        #  18
        if handle_digits_words_spacing:
            self.regex_sub_digits_words_spacing = (
                regex_lib.get_digits_words_spacing_regex()
            )
            self.regex_set.append(self.regex_sub_digits_words_spacing)

        #  19 remove phone numbers
        if remove_phone_numbers:
            self.regex_sub_phone_numbers = regex_lib.get_phone_number_regex()
            self.regex_set.append(self.regex_sub_phone_numbers)

        #  20 remove dates and times
        if remove_dates:
            self.regex_sub_dates = regex_lib.get_dates_regex()
            self.regex_set.append(self.regex_sub_dates)

        #  21 remove time
        if remove_times:
            self.regex_sub_time = regex_lib.get_time_regex()
            self.regex_set.append(self.regex_sub_time)

        #  22 remove addresses
        if remove_addresses:
            self.regex_sub_address = regex_lib.get_address_regex()
            self.regex_set.append(self.regex_sub_address)

        #  23 dimension substitution
        if remove_dimensions:
            self.regex_sub_dimensions = regex_lib.get_dimensions_regex()
            self.regex_set.append(self.regex_sub_dimensions)

        #  24 specimen
        if remove_specimen:
            self.regex_sub_specimen = regex_lib.get_specimen_regex()
            self.regex_set.append(self.regex_sub_specimen)

        #  25
        if remove_decimal_seg_numbers:
            self.regex_sub_decimal_seg_numbers = (
                regex_lib.get_decimal_segmented_numbers_regex()
            )
            self.regex_set.append(self.regex_sub_decimal_seg_numbers)

        #  26
        if remove_large_digits_seq:
            self.regex_sub_large_digits_seq = regex_lib.get_large_digits_seq_regex()
            self.regex_set.append(self.regex_sub_large_digits_seq)

        #  27
        if remove_large_floats_seq:
            self.regex_sub_large_floats_seq = regex_lib.get_large_float_seq_regex()
            self.regex_set.append(self.regex_sub_large_floats_seq)

        #  28
        if trunc_decimals:
            self.regex_sub_trunc_decimals = regex_lib.get_trunc_decimals_regex()
            self.regex_set.append(self.regex_sub_trunc_decimals)

        #  29
        if remove_cassette_names:
            self.regex_sub_cassette_names = regex_lib.get_cassette_name_regex()
            self.regex_set.append(self.regex_sub_cassette_names)

        # 30
        if remove_duration_time:
            self.regex_sub_duration_time = regex_lib.get_duration_regex()
            self.regex_set.append(self.regex_sub_duration_time)

        # 31
        if remove_letter_num_seq:
            self.regex_sub_letter_num_seq = regex_lib.get_letter_num_seq_regex()
            self.regex_set.append(self.regex_sub_letter_num_seq)

        #  LAST condense spacing - not configurable, always executes
        self.regex_sub_spaces = regex_lib.get_spaces_regex()
        self.regex_set.append(self.regex_sub_spaces)
