import re
import unittest
from unittest import TestCase

from bardi.nlp_engineering.regex_library import regex_lib


class TestRegexExpressions(TestCase):
    """Tests the correctness of the functions in bardi's  regex library"""

    # 0
    def test_escape_code_regex(self):
        """Tests the regular expression for the escape codes"""

        input_str = "\\x0dTesting escape codes\\x0d\\x0a\\x0d \\r30  "
        correct_result = " Testing escape codes     30  "

        regex_sub_pair = regex_lib.get_escape_code_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]
        test_str = re.sub(regex_pattern, sub_str, input_str)
        self.assertEqual(test_str, correct_result, "Incorrect escape code substitution result.")

    # 1
    def test_whitespaces_regex(self):
        """Tests the regular expression for the whitespace removal"""

        test_case = "INVASIVE:\nNegative    IN SITU:\nN/A  IN \tThe result \r"
        expected_output = "INVASIVE: Negative IN SITU: N/A IN The result "

        regex_sub_pair = regex_lib.get_whitespace_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]
        output = re.sub(regex_pattern, sub_str, test_case)
        self.assertEqual(output, expected_output, "Incorrect whitespace code substitution result.")

    # 2
    def test_urls_regex(self):
        """Tests the regular expression for urls substitution"""

        remove_urls_test_list = [
            {
                "test": " Source: https://www.merck.com/keytruda_pi.pdf ",
                "expected_output": " Source:  URLTOKEN  ",
            },
            {
                "test": " Libtayo: www.regeneron.com/libtayo_fpi.pdf Patient",
                "expected_output": " Libtayo:  URLTOKEN  Patient",
            },
        ]

        regex_sub_pair = regex_lib.get_urls_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]

        for test_case in remove_urls_test_list:
            test_case["test"] = re.sub(regex_pattern, sub_str, test_case["test"])
            self.assertEqual(
                test_case["test"],
                test_case["expected_output"],
                "Incorrect URLs substitution result.",
            )

    # 3
    def test_remove_special_punct(self):
        """Tests the regular expression for removal
        of the chosen punctuation"""

        chosen_punt_test_list = [
            {
                "test": " wt-1, ck-7 (focal) negative; [sth] ab|cd",
                "expected_output": " wt-1  ck-7  focal  negative   sth  ab cd",
            },
            {
                "test": " h * 1701 oak park blvd * lake charle",
                "expected_output": " h   1701 oak park blvd   lake charle",
            },
        ]

        regex_sub_pair = regex_lib.get_special_punct_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]

        for test_case in chosen_punt_test_list:
            test_case["test"] = re.sub(regex_pattern, sub_str, test_case["test"])
            self.assertEqual(
                test_case["test"],
                test_case["expected_output"],
                "Incorrect chosen punctuation result.",
            )

    # 4
    def test_multiple_punct_regex(self):
        """Tests the regular expression for multiple
        sequntial punctuation removal."""

        test_case = "-----this is report ___ signature"
        expected_output = " this is report   signature"

        regex_sub_pair = regex_lib.get_multiple_punct_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]
        output = re.sub(regex_pattern, sub_str, test_case)
        self.assertEqual(output, expected_output, "Incorrect multiple punctuation removal result.")

    # 5
    def test_angle_brackets_regex(self):
        """Tests the regular expression for angle brackets removal."""

        test_case = "<This should be fixed> But not this >90"
        expected_output = " This should be fixed  But not this >90"

        regex_sub_pair = regex_lib.get_angle_brackets_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]
        output = re.sub(regex_pattern, sub_str, test_case)
        self.assertEqual(output, expected_output, "Incorrect angle brackets removal result.")

    # 6
    def test_percent_sign_regex(self):
        """Tests the regular expression for the percent sign substitution."""

        test_case = "strong intensity >95%"
        expected_output = "strong intensity >95 percent "

        regex_sub_pair = regex_lib.get_percent_sign_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]
        output = re.sub(regex_pattern, sub_str, test_case)
        self.assertEqual(output, expected_output, "Incorrect percent sign substitution result.")

    # 7
    def test_leading_digit_punctuation_regex(self):
        """Tests the regular expression for the leading digit punct removal."""

        test_case = " 13-unremarkable 1-e 22-years "
        expected_output = "  13 unremarkable   1 e   22 years  "

        regex_sub_pair = regex_lib.get_leading_digit_punctuation_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]
        output = re.sub(regex_pattern, sub_str, test_case)
        self.assertEqual(
            output, expected_output, "Incorrect leading digit punctuation removal result."
        )

    # 8
    def test_leading_punctuation_seq_regex(self):
        """Tests the regular expression for the leading punct removal."""

        test_case = " -3a -anterior -result- :cassette "
        expected_output = " 3a  anterior  result-  cassette  "

        regex_sub_pair = regex_lib.get_leading_punctuation_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]
        output = re.sub(regex_pattern, sub_str, test_case)
        self.assertEqual(output, expected_output, "Incorrect leading punctuation removal result.")

    # 9
    def test_trailing_punctuation_seq_regex(self):
        """Tests the regular expression for the trailing punct removal."""

        test_case = " -3a -anterior -result- :cassette "
        expected_output = " -3a -anterior  -result :cassette "

        regex_sub_pair = regex_lib.get_trailing_punctuation_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]
        output = re.sub(regex_pattern, sub_str, test_case)
        self.assertEqual(output, expected_output, "Incorrect trailing punctuation removal result.")

    # 10
    def test_words_with_punct_spacing_regex(self):
        """Tests the regular expression that adds spacing to
        words that contain punctuation."""

        test_case = "this-that her-2 tiff-1k description:gleason "
        expected_output = "this that her-2 tiff-1k description gleason "

        regex_sub_pair = regex_lib.get_words_with_punct_spacing_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]
        output = re.sub(regex_pattern, sub_str, test_case)

        self.assertEqual(
            output, expected_output, "Incorrect words with punctuation spacing result."
        )

    # 11
    def test_math_spacing_regex(self):
        """Tests the regular expression adding spaces
        between math operators."""

        test_case = "This is >95% 3+3=8  6/7"
        expected_output = "This is  > 95 %  3 + 3 = 8  6 / 7"

        regex_sub_pair = regex_lib.get_math_spacing_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]
        output = re.sub(regex_pattern, sub_str, test_case)

        self.assertEqual(output, expected_output, "Incorrect math operators spacing result.")

    # 12
    def test_dimension_spacing_regex(self):
        """Tests the regular expression for adding space
        between a dimensions."""

        test_case = "measuring 1.3x0.7x0.1 cm"
        expected_output = "measuring 1.3 x 0.7 x 0.1 cm"

        regex_sub_pair = regex_lib.get_dimension_spacing_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]
        output = re.sub(regex_pattern, sub_str, test_case)

        self.assertEqual(output, expected_output, "Incorrect dimension spacing result.")

    # 13
    def test_measure_spacing_regex(self):
        """Tests the regular expression for adding space between
        digits and measurements or common abbraviations."""

        test_case = "10mm histologic type 2 x 3cm. this is 3.0-cm "
        expected_output = "10 mm  histologic type 2 x 3 cm . this is 3.0 cm  "

        regex_sub_pair = regex_lib.get_measure_spacing_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]
        output = re.sub(regex_pattern, sub_str, test_case)

        self.assertEqual(output, expected_output, "Incorrect measure spacing result.")

    # 14
    def test_cassette_spacing_regex(self):
        """Tests the regular expression for adding
        spaces between the cassettes marking."""

        fix_spacing_test_list = [{"test": " 3e-3f", "expected_output": " 3e - 3f "}]

        regex_sub_pair = regex_lib.get_cassettes_spacing_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]

        for test_case in fix_spacing_test_list:
            test_case["test"] = re.sub(regex_pattern, sub_str, test_case["test"])
            self.assertEqual(
                test_case["test"],
                test_case["expected_output"],
                "Incorrect cassette spacing result.",
            )

    # 15
    def test_dash_digits_spacing_regex(self):
        """Tests the regular expression for adding spaces
        between dashes and digits."""

        test_case = "right 1:30-2:30 1.5-2.0 cm 0.9 cm for the 7-6"
        expected_output = "right 1:30 - 2:30 1.5 - 2.0 cm 0.9 cm for the 7 - 6"

        regex_sub_pair = regex_lib.get_dash_digits_spacing_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]
        output = re.sub(regex_pattern, sub_str, test_case)

        self.assertEqual(output, expected_output, "Incorrect dash digit spacing result.")

    # 16
    def test_literals_float_spacing_regex(self):
        """Tests the regular expression for literals and floats spacing."""

        test_case = " r18.0admission diagnosis: bi n13.30admission "
        expected_output = " r18.0 admission diagnosis: bi n13.30 admission "

        regex_sub_pair = regex_lib.get_literals_floats_spacing_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]
        output = re.sub(regex_pattern, sub_str, test_case)

        self.assertEqual(output, expected_output, "Incorrect literals floats spacing result.")

    # 17
    def test_fix_pluralization_regex(self):
        """Tests the regular expression that fix pluralization."""

        test_case = " specimen s code s "
        expected_output = " specimens codes "

        regex_sub_pair = regex_lib.get_fix_pluralization_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]
        output = re.sub(regex_pattern, sub_str, test_case)

        self.assertEqual(output, expected_output, "Incorrect fix pluralization result.")

    # 18
    def test_digits_words_spacing_regex(self):
        """Tests the regular expression for the digits and words spacing."""

        test_case = " 9837648admission "
        expected_output = " 9837648 admission "

        regex_sub_pair = regex_lib.get_digits_words_spacing_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]
        output = re.sub(regex_pattern, sub_str, test_case)

        self.assertEqual(output, expected_output, "Incorrect digits words spacing result.")

    # 19
    def test_remove_phone_numbers(self):
        """Tests the regular expression for replacing phone numbers."""

        phone_number_test_list = [
            {
                "test": "Ph: (123) 456 7890. It is (123)4567890.",
                "expected_output": "Ph:  PHONENUMTOKEN . It is  PHONENUMTOKEN .",
            },
            {
                "test": "PH: 123 456-7890. Call 1234567890 ",
                "expected_output": "PH:  PHONENUMTOKEN . Call  PHONENUMTOKEN  ",
            },
        ]

        regex_sub_pair = regex_lib.get_phone_number_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]

        for test_case in phone_number_test_list:
            test_case["test"] = re.sub(regex_pattern, sub_str, test_case["test"])
            self.assertEqual(
                test_case["test"], test_case["expected_output"], "Inocrrect phone removal result."
            )

    # 20
    def test_remove_dates(self):
        """Tests the regular expression for replacing dates."""

        dates_test_list = [
            {
                "test": "co: 03/09/2001 1015 completed: 03/10/01 at 3:34.",
                "expected_output": "co:  DATETOKEN completed:  DATETOKEN .",
            },
            {
                "test": " signed 06/20/2022 17:02 performed 01may2012",
                "expected_output": " signed  DATETOKEN performed  DATETOKEN ",
            },
            {
                "test": "report collected 15-dec-18 3:30:00 pm",
                "expected_output": "report collected  DATETOKEN ",
            },
            {
                "test": "report collected #RECD: 02/13/20-1151",
                "expected_output": "report collected #RECD:  DATETOKEN ",
            },
        ]

        regex_sub_pair = regex_lib.get_dates_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]

        for test_case in dates_test_list:
            test_case["test"] = re.sub(regex_pattern, sub_str, test_case["test"])
            self.assertEqual(
                test_case["test"],
                test_case["expected_output"],
                "Incorrect dates replacement result.",
            )

    # 21
    def test_remove_times(self):
        """Tests the regular expression for replacing times."""

        time_test_list = [
            {
                "test": "at 11:12 pm or 11.12am ",
                "expected_output": "at  TIMETOKEN  or  TIMETOKEN  ",
            },
            {
                "test": "at 9:52:07am. Rec: 06am 17:34",
                "expected_output": "at  TIMETOKEN  Rec:  TIMETOKEN   TIMETOKEN ",
            },
        ]

        regex_sub_pair = regex_lib.get_time_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]

        for test_case in time_test_list:
            test_case["test"] = re.sub(regex_pattern, sub_str, test_case["test"])
            self.assertEqual(
                test_case["test"],
                test_case["expected_output"],
                "Incorrect time replacement result.",
            )

    # 22
    def test_remove_addresses(self):
        """Tests the regular expression for replacing addresses."""

        addresses_test_list = [
            {
                "test": " 1234 north 500 west provo ca 12345-6789 ",
                "expected_output": "  ADDRESSTOKEN  ",
            },
            {
                "test": "111 st. landry street lafayette va. 12345",
                "expected_output": " ADDRESSTOKEN ",
            },
            {
                "test": "services llc. 123 e. crabcd street acbde ca 12345 ",
                "expected_output": "services llc.  ADDRESSTOKEN  ",
            },
            {"test": "123 jackson street ancloa ca 12345", "expected_output": " ADDRESSTOKEN "},
            {"test": "12 colabcd abcd viejo nc / 12345 ", "expected_output": " ADDRESSTOKEN  "},
        ]

        regex_sub_pair = regex_lib.get_address_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]

        for test_case in addresses_test_list:
            test_case["test"] = re.sub(regex_pattern, sub_str, test_case["test"])
            self.assertEqual(
                test_case["test"],
                test_case["expected_output"],
                "Incorrect address replacement result.",
            )

    # 23
    def test_dimensions_regex(self):
        """Tests the regular expression for replacing dimensions."""

        test_case = " 3.5 x 2.5 x 9.0 cm and 33 x 6.5 cm"
        expected_output = "  DIMENSIONTOKEN  cm and  DIMENSIONTOKEN  cm"

        regex_sub_pair = regex_lib.get_dimensions_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]
        output = re.sub(regex_pattern, sub_str, test_case)

        self.assertEqual(output, expected_output, "Incorrect dimension replecement result.")

    # 24
    def test_specimen_regex(self):
        """Tests the regular expression for replacing specimen names."""

        specimen_test_list = [
            {
                "test": " for s-21-009345 sh-22-0011300 ",
                "expected_output": " for  SPECIMENTOKEN   SPECIMENTOKEN  ",
            },
            {
                "test": " bio hsp-21-728 imm s22-063124 ",
                "expected_output": " bio  SPECIMENTOKEN  imm  SPECIMENTOKEN  ",
            },
        ]

        regex_sub_pair = regex_lib.get_specimen_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]

        for test_case in specimen_test_list:
            test_case["test"] = re.sub(regex_pattern, sub_str, test_case["test"])
            self.assertEqual(
                test_case["test"],
                test_case["expected_output"],
                "Incorrect specimen replacement result.",
            )

    # 25
    def test_decimal_segmented_numbers_regex(self):
        """Tests the regular expression for the decimal segmented
        numbers replacement."""

        test_case = " 1.78.9.87 "
        expected_output = "  DECIMALSEGMENTEDNUMBERTOKEN  "

        regex_sub_pair = regex_lib.get_decimal_segmented_numbers_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]
        output = re.sub(regex_pattern, sub_str, test_case)

        self.assertEqual(output, expected_output, "Incorrrect decimal segmented numbers result.")

    # 26
    def test_large_digits_seq_regex(self):
        """Tests the regular expression for replacement of large digits."""

        test_case = " 456123456 "
        expected_output = " DIGITSEQUENCETOKEN "

        regex_sub_pair = regex_lib.get_large_digits_seq_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]
        output = re.sub(regex_pattern, sub_str, test_case)

        self.assertEqual(output, expected_output, "Inocrrect large digits replacement result.")

    # 27
    def test_large_floats_seq_regex(self):
        """Tests the regular expression for replacement of large flaots."""

        test_case = " 456 123456.783 "
        expected_output = " 456 LARGEFLOATTOKEN  "

        regex_sub_pair = regex_lib.get_large_float_seq_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]
        output = re.sub(regex_pattern, sub_str, test_case)

        self.assertEqual(output, expected_output, "Incorrect large float replecement result.")

    # 28
    def test_trunc_decimal_float_regex(self):
        """Tests the regular expression for truncating decimals."""

        test_case = " 1.78  9.87 - 8.99 "
        expected_output = " 1.7  9.8 - 8.9 "

        regex_sub_pair = regex_lib.get_trunc_decimals_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]
        output = re.sub(regex_pattern, sub_str, test_case)

        self.assertEqual(output, expected_output, "Incorrect truncate decimal result.")

    # 29
    def test_cassette_name_regex(self):
        """Tests the regular expression for replacement of cassettes names."""

        cassettes_test_list = [
            {"test": " block:  1-e ", "expected_output": " block:  CASSETTETOKEN "},
            {"test": " in 7a f8 ", "expected_output": " in CASSETTETOKEN  CASSETTETOKEN "},
            {"test": " c2-1  1-ef ", "expected_output": " CASSETTETOKEN  CASSETTETOKEN "},
        ]

        regex_sub_pair = regex_lib.get_cassette_name_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]

        for test_case in cassettes_test_list:
            test_case["test"] = re.sub(regex_pattern, sub_str, test_case["test"])
            self.assertEqual(
                test_case["test"],
                test_case["expected_output"],
                "Incorrect cassettes' names removal result.",
            )

    # 30
    def test_duration_regex(self):
        """Tests the regular expression for remova"""
        test_case = "duration 02d2043058. "
        expected_output = "duration DURATIONTOKEN "
        regex_sub_pair = regex_lib.get_duration_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]
        output = re.sub(regex_pattern, sub_str, test_case)
        self.assertEqual(output, expected_output, "Incorrect duration removal result.")

    # 31
    def test_letter_num_seq_regex(self):
        """Tests the regular expression for letter num"""
        test_case = "f1234567  h123456789 "
        expected_output = " LETTERDIGITSTOKEN  LETTERDIGITSTOKEN "
        regex_sub_pair = regex_lib.get_letter_num_seq_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]
        output = re.sub(regex_pattern, sub_str, test_case)
        self.assertEqual(output, expected_output, "Incorrect letter seg num removal result.")

    # LAST
    def test_spaces_regex(self):
        """Tests the regular expression for removal of additional spaces."""

        test_case = "located around lower arm specimen   date"
        expected_output = "located around lower arm specimen date"

        regex_sub_pair = regex_lib.get_spaces_regex()
        regex_pattern = regex_sub_pair["regex_str"]
        sub_str = regex_sub_pair["sub_str"]
        output = re.sub(regex_pattern, sub_str, test_case)

        self.assertEqual(output, expected_output, "Incorrect space removal result.")


if __name__ == "__main__":
    unittest.main()
