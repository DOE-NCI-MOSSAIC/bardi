"""Shared test vectors for the regex library characterization tests.

PARITY_VECTORS below are copied verbatim from ``tests/regex_tests.py``, which
is frozen for enclave comparison — do not edit that file. Each vector records
the expected output of Python's ``re.sub`` (``re_expected``, the value asserted
in the frozen file) alongside the output observed from the production polars
(Rust regex) engine (``polars_expected``). The ``SAME`` sentinel means both
engines produce the identical string for that input — which, as of capture
(2026-07-30, polars-u64-idx lockfile version), holds for every vector.

EDGE_VECTORS are new, polars-engine-only characterization vectors: empty
strings, no-op passthroughs, unicode, and must-NOT-match near-misses. Their
expected values are whatever the production engine actually produced when they
were captured — they document current behavior, they do not bless it as
correct.

This module deliberately does not match the ``*_tests.py`` pytest collection
pattern, so it is imported as data, never collected.
"""

from typing import Dict, List, Union


class _Same:
    """Sentinel: polars output is identical to ``re_expected``."""

    def __repr__(self) -> str:  # pragma: no cover - repr for debugging only
        return "SAME"


SAME = _Same()

# The 13 uppercase substitution tokens the default PathologyReportRegexSet can
# emit. Lowercasing happens before regex substitution in CPUNormalizer, so
# these are the only uppercase letter runs a default-chain output may contain.
KNOWN_TOKENS = frozenset(
    {
        "URLTOKEN",
        "PHONENUMTOKEN",
        "DATETOKEN",
        "TIMETOKEN",
        "ADDRESSTOKEN",
        "DIMENSIONTOKEN",
        "SPECIMENTOKEN",
        "DECIMALSEGMENTEDNUMBERTOKEN",
        "DIGITSEQUENCETOKEN",
        "LARGEFLOATTOKEN",
        "CASSETTETOKEN",
        "DURATIONTOKEN",
        "LETTERDIGITSTOKEN",
    }
)

Vector = Dict[str, Union[str, _Same]]

# Keyed by the regex_lib getter name. Inputs and re_expected values are
# copied verbatim from tests/regex_tests.py (frozen).
PARITY_VECTORS: Dict[str, List[Vector]] = {
    # 0
    "get_escape_code_regex": [
        {
            "input": "\\x0dTesting escape codes\\x0d\\x0a\\x0d \\r30  ",
            "re_expected": " Testing escape codes     30  ",
            "polars_expected": SAME,
        },
    ],
    # 1
    "get_whitespace_regex": [
        {
            "input": "INVASIVE:\nNegative    IN SITU:\nN/A  IN \tThe result \r",
            "re_expected": "INVASIVE: Negative IN SITU: N/A IN The result ",
            "polars_expected": SAME,
        },
    ],
    # 2
    "get_urls_regex": [
        {
            "input": " Source: https://www.merck.com/keytruda_pi.pdf ",
            "re_expected": " Source:  URLTOKEN  ",
            "polars_expected": SAME,
        },
        {
            "input": " Libtayo: www.regeneron.com/libtayo_fpi.pdf Patient",
            "re_expected": " Libtayo:  URLTOKEN  Patient",
            "polars_expected": SAME,
        },
    ],
    # 3
    "get_special_punct_regex": [
        {
            "input": " wt-1, ck-7 (focal) negative; [sth] ab|cd",
            "re_expected": " wt-1  ck-7  focal  negative   sth  ab cd",
            "polars_expected": SAME,
        },
        {
            "input": " h * 1701 oak park blvd * lake charle",
            "re_expected": " h   1701 oak park blvd   lake charle",
            "polars_expected": SAME,
        },
    ],
    # 4
    "get_multiple_punct_regex": [
        {
            "input": "-----this is report ___ signature",
            "re_expected": " this is report   signature",
            "polars_expected": SAME,
        },
    ],
    # 5
    "get_angle_brackets_regex": [
        {
            "input": "<This should be fixed> But not this >90",
            "re_expected": " This should be fixed  But not this >90",
            "polars_expected": SAME,
        },
    ],
    # 6
    "get_percent_sign_regex": [
        {
            "input": "strong intensity >95%",
            "re_expected": "strong intensity >95 percent ",
            "polars_expected": SAME,
        },
    ],
    # 7
    "get_leading_digit_punctuation_regex": [
        {
            "input": " 13-unremarkable 1-e 22-years ",
            "re_expected": "  13 unremarkable   1 e   22 years  ",
            "polars_expected": SAME,
        },
    ],
    # 8
    "get_leading_punctuation_regex": [
        {
            "input": " -3a -anterior -result- :cassette ",
            "re_expected": " 3a  anterior  result-  cassette  ",
            "polars_expected": SAME,
        },
    ],
    # 9
    "get_trailing_punctuation_regex": [
        {
            "input": " -3a -anterior -result- :cassette ",
            "re_expected": " -3a -anterior  -result :cassette ",
            "polars_expected": SAME,
        },
    ],
    # 10
    "get_words_with_punct_spacing_regex": [
        {
            "input": "this-that her-2 tiff-1k description:gleason ",
            "re_expected": "this that her-2 tiff-1k description gleason ",
            "polars_expected": SAME,
        },
    ],
    # 11
    "get_math_spacing_regex": [
        {
            "input": "This is >95% 3+3=8  6/7",
            "re_expected": "This is  > 95 %  3 + 3 = 8  6 / 7",
            "polars_expected": SAME,
        },
    ],
    # 12
    "get_dimension_spacing_regex": [
        {
            "input": "measuring 1.3x0.7x0.1 cm",
            "re_expected": "measuring 1.3 x 0.7 x 0.1 cm",
            "polars_expected": SAME,
        },
    ],
    # 13
    "get_measure_spacing_regex": [
        {
            "input": "10mm histologic type 2 x 3cm. this is 3.0-cm ",
            "re_expected": "10 mm  histologic type 2 x 3 cm . this is 3.0 cm  ",
            "polars_expected": SAME,
        },
    ],
    # 14
    "get_cassettes_spacing_regex": [
        {
            "input": " 3e-3f",
            "re_expected": " 3e - 3f ",
            "polars_expected": SAME,
        },
    ],
    # 15
    "get_dash_digits_spacing_regex": [
        {
            "input": "right 1:30-2:30 1.5-2.0 cm 0.9 cm for the 7-6",
            "re_expected": "right 1:30 - 2:30 1.5 - 2.0 cm 0.9 cm for the 7 - 6",
            "polars_expected": SAME,
        },
    ],
    # 16
    "get_literals_floats_spacing_regex": [
        {
            "input": " r18.0admission diagnosis: bi n13.30admission ",
            "re_expected": " r18.0 admission diagnosis: bi n13.30 admission ",
            "polars_expected": SAME,
        },
    ],
    # 17
    "get_fix_pluralization_regex": [
        {
            "input": " specimen s code s ",
            "re_expected": " specimens codes ",
            "polars_expected": SAME,
        },
    ],
    # 18
    "get_digits_words_spacing_regex": [
        {
            "input": " 9837648admission ",
            "re_expected": " 9837648 admission ",
            "polars_expected": SAME,
        },
    ],
    # 19
    "get_phone_number_regex": [
        {
            "input": "Ph: (123) 456 7890. It is (123)4567890.",
            "re_expected": "Ph:  PHONENUMTOKEN . It is  PHONENUMTOKEN .",
            "polars_expected": SAME,
        },
        {
            "input": "PH: 123 456-7890. Call 1234567890 ",
            "re_expected": "PH:  PHONENUMTOKEN . Call  PHONENUMTOKEN  ",
            "polars_expected": SAME,
        },
    ],
    # 20
    "get_dates_regex": [
        {
            "input": "co: 03/09/2001 1015 completed: 03/10/01 at 3:34.",
            "re_expected": "co:  DATETOKEN completed:  DATETOKEN .",
            "polars_expected": SAME,
        },
        {
            "input": " signed 06/20/2022 17:02 performed 01may2012",
            "re_expected": " signed  DATETOKEN performed  DATETOKEN ",
            "polars_expected": SAME,
        },
        {
            "input": "report collected 15-dec-18 3:30:00 pm",
            "re_expected": "report collected  DATETOKEN ",
            "polars_expected": SAME,
        },
        {
            "input": "report collected #RECD: 02/13/20-1151",
            "re_expected": "report collected #RECD:  DATETOKEN ",
            "polars_expected": SAME,
        },
    ],
    # 21
    "get_time_regex": [
        {
            "input": "at 11:12 pm or 11.12am ",
            "re_expected": "at  TIMETOKEN  or  TIMETOKEN  ",
            "polars_expected": SAME,
        },
        {
            "input": "at 9:52:07am. Rec: 06am 17:34",
            "re_expected": "at  TIMETOKEN  Rec:  TIMETOKEN   TIMETOKEN ",
            "polars_expected": SAME,
        },
    ],
    # 22
    "get_address_regex": [
        {
            "input": " 1234 north 500 west provo ca 12345-6789 ",
            "re_expected": "  ADDRESSTOKEN  ",
            "polars_expected": SAME,
        },
        {
            "input": "111 st. landry street lafayette va. 12345",
            "re_expected": " ADDRESSTOKEN ",
            "polars_expected": SAME,
        },
        {
            "input": "services llc. 123 e. crabcd street acbde ca 12345 ",
            "re_expected": "services llc.  ADDRESSTOKEN  ",
            "polars_expected": SAME,
        },
        {
            "input": "123 jackson street ancloa ca 12345",
            "re_expected": " ADDRESSTOKEN ",
            "polars_expected": SAME,
        },
        {
            "input": "12 colabcd abcd viejo nc / 12345 ",
            "re_expected": " ADDRESSTOKEN  ",
            "polars_expected": SAME,
        },
    ],
    # 23
    "get_dimensions_regex": [
        {
            "input": " 3.5 x 2.5 x 9.0 cm and 33 x 6.5 cm",
            "re_expected": "  DIMENSIONTOKEN  cm and  DIMENSIONTOKEN  cm",
            "polars_expected": SAME,
        },
    ],
    # 24
    "get_specimen_regex": [
        {
            "input": " for s-21-009345 sh-22-0011300 ",
            "re_expected": " for  SPECIMENTOKEN   SPECIMENTOKEN  ",
            "polars_expected": SAME,
        },
        {
            "input": " bio hsp-21-728 imm s22-063124 ",
            "re_expected": " bio  SPECIMENTOKEN  imm  SPECIMENTOKEN  ",
            "polars_expected": SAME,
        },
    ],
    # 25
    "get_decimal_segmented_numbers_regex": [
        {
            "input": " 1.78.9.87 ",
            "re_expected": "  DECIMALSEGMENTEDNUMBERTOKEN  ",
            "polars_expected": SAME,
        },
    ],
    # 26
    "get_large_digits_seq_regex": [
        {
            "input": " 456123456 ",
            "re_expected": " DIGITSEQUENCETOKEN ",
            "polars_expected": SAME,
        },
    ],
    # 27
    "get_large_float_seq_regex": [
        {
            "input": " 456 123456.783 ",
            "re_expected": " 456 LARGEFLOATTOKEN  ",
            "polars_expected": SAME,
        },
    ],
    # 28
    "get_trunc_decimals_regex": [
        {
            "input": " 1.78  9.87 - 8.99 ",
            "re_expected": " 1.7  9.8 - 8.9 ",
            "polars_expected": SAME,
        },
    ],
    # 29
    "get_cassette_name_regex": [
        {
            "input": " block:  1-e ",
            "re_expected": " block:  CASSETTETOKEN ",
            "polars_expected": SAME,
        },
        {
            "input": " in 7a f8 ",
            "re_expected": " in CASSETTETOKEN  CASSETTETOKEN ",
            "polars_expected": SAME,
        },
        {
            "input": " c2-1  1-ef ",
            "re_expected": " CASSETTETOKEN  CASSETTETOKEN ",
            "polars_expected": SAME,
        },
    ],
    # 30
    "get_duration_regex": [
        {
            "input": "duration 02d2043058. ",
            "re_expected": "duration DURATIONTOKEN ",
            "polars_expected": SAME,
        },
    ],
    # 31
    "get_letter_num_seq_regex": [
        {
            "input": "f1234567  h123456789 ",
            "re_expected": " LETTERDIGITSTOKEN  LETTERDIGITSTOKEN ",
            "polars_expected": SAME,
        },
    ],
    # LAST
    "get_spaces_regex": [
        {
            "input": "located around lower arm specimen   date",
            "re_expected": "located around lower arm specimen date",
            "polars_expected": SAME,
        },
    ],
}

# Polars-engine-only characterization vectors: empty strings, unicode,
# passthrough (must-NOT-match), and near-miss cases. ``polars_expected`` is the
# empirically captured output of the production engine — several entries
# document over- or under-matching that a future fix would change.
EDGE_VECTORS: Dict[str, List[Dict[str, str]]] = {
    "get_escape_code_regex": [
        # \t / \r branch (`\\[stepr]`) eats the first letter of words after a
        # literal backslash: "\test" -> " est"
        {"input": "\\test \\react \\x0", "polars_expected": " est  eact \\x0"},
        # a lone backslash is untouched (needs a following [stepr] or \xNN)
        {"input": "literal \\ backslash", "polars_expected": "literal \\ backslash"},
    ],
    "get_whitespace_regex": [
        {"input": "", "polars_expected": ""},
        # \s is unicode-aware in both engines: NBSP pair collapses
        {"input": "a\u00a0\u00a0b", "polars_expected": "a b"},
        {"input": "no extra ws here", "polars_expected": "no extra ws here"},
    ],
    "get_urls_regex": [
        # no scheme/www prefix at a word boundary -> no match
        {
            "input": "wwwabc.com is not a url? http not either",
            "polars_expected": "wwwabc.com is not a url? http not either",
        },
    ],
    "get_dimension_spacing_regex": [
        # accented text passes through untouched around the match
        {"input": "caf\u00e9 1.3x0.7 cm", "polars_expected": "caf\u00e9 1.3 x 0.7 cm"},
        {"input": "0x0", "polars_expected": "0 x 0"},
    ],
    "get_cassettes_spacing_regex": [
        # neither alternation branch matches digit-letter '-' bare-digit
        {"input": " c2-1", "polars_expected": " c2-1"},
        # second branch ([a-z]\d - [a-z]\d) matches "c2-c3", but the sub_str
        # references groups 1-3 which belong to the *first* branch; both
        # engines expand unset groups to empty strings, erasing the text
        {"input": " 3e-3f and c2-c3 ", "polars_expected": " 3e - 3f  and     "},
    ],
    "get_phone_number_regex": [
        # pattern is unanchored: matches the first 10 digits inside longer runs
        {"input": "specimen 123 456 7890123", "polars_expected": "specimen  PHONENUMTOKEN 123"},
        {"input": "code 12345678901", "polars_expected": "code  PHONENUMTOKEN 1"},
    ],
    "get_dates_regex": [
        # single-separator decimals / fractions / times are NOT dates
        {"input": "see 2.3 for details", "polars_expected": "see 2.3 for details"},
        {"input": "sections 3/4", "polars_expected": "sections 3/4"},
        {"input": "at 3:34.", "polars_expected": "at 3:34."},
    ],
    "get_time_regex": [
        # third branch [0-2][0-9]:[0-5][1-9] rejects minutes ending in 0
        {"input": "ratio 10:59 vs 10:50", "polars_expected": "ratio  TIMETOKEN  vs 10:50"},
        # second branch \d{2}[ap]m over-matches invalid hours
        {"input": "99am", "polars_expected": " TIMETOKEN "},
    ],
    "get_address_regex": [
        # over-match: any "num words.. 2-letters 5-digits" shape is an address
        {
            "input": "12 to 15 percent of ca 12345",
            "polars_expected": " ADDRESSTOKEN ",
        },
        # no leading street number -> no match
        {"input": "ca 12345", "polars_expected": "ca 12345"},
    ],
    "get_dimensions_regex": [
        # digits required on both sides of x: prose 'x' does not match ...
        {"input": "2 x biopsy", "polars_expected": "2 x biopsy"},
        # ... but any 'N x N' counts, even when it is not a measurement
        {
            "input": "slides 2 x 4 were reviewed",
            "polars_expected": "slides  DIMENSIONTOKEN  were reviewed",
        },
        {"input": "x 2 x 3", "polars_expected": "x  DIMENSIONTOKEN "},
    ],
    "get_specimen_regex": [
        # middle segment must be exactly 2 digits
        {"input": " s-2021-009345 ", "polars_expected": " s-2021-009345 "},
        # last segment must be 3+ digits
        {"input": " s-21-09 ", "polars_expected": " s-21-09 "},
        # prefix longer than 3 letters: match starts inside the prefix
        {"input": " abcd-21-009345 ", "polars_expected": " a SPECIMENTOKEN  "},
    ],
    "get_cassette_name_regex": [
        {"input": " 13-e ", "polars_expected": " CASSETTETOKEN "},
        {"input": " a12-13 ", "polars_expected": " CASSETTETOKEN "},
        # second branch \b[a-z][\-]*\d{1}\s leaves the leading space in place
        {"input": " q5 ", "polars_expected": "  CASSETTETOKEN "},
    ],
}
