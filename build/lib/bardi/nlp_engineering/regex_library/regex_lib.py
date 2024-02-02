"""Library of pre-defined regular expression substitution pairs."""

from bardi.nlp_engineering.regex_library.regex_set import RegexSubPair


# 0
def get_escape_code_regex() -> RegexSubPair:
    """Matches escape codes such as \\x0d, \\x0a, etc.

    Returns
    -------

    RegexSubPair
         {regex pattern, replacement string}

    Example
    -------

    Input string: ::
    
        Codes\\x0d\\x0a\\x0d \\r30

    Output string: ::
    
        Codes      30
    """

    regex_sub_pair = {"regex_str": r"(\\x[0-9A-Fa-f]{2,})|\\[stepr]", "sub_str": " "}
    return regex_sub_pair


# 1
def get_whitespace_regex() -> RegexSubPair:
    """Matches any new line, carriage return tab and multiple
    spaces and replaces it with a single space.

    Returns
    -------

        RegexSubPair - (regex pattern, replacement string)

    Example
    -------

    Input string: ::
    
        INVASIVE:\\nNegative    IN SITU:\\nN/A  IN \\tThe result\ \r 

    Output string: ::
    
        INVASIVE: Negative IN SITU: N/A IN The result
    """

    regex_sub_pair = {"regex_str": r"[\r\n\t]|\s{2,}", "sub_str": " "}
    return regex_sub_pair


# 2
def get_urls_regex() -> RegexSubPair:
    """Matches a url and replaces it
    with a URLTOKEN.

    Returns
    -------

        RegexSubPair - (regex pattern, replacement string)

    Example
    -------

    Input string: ::
    
        Source: https://www.merck.com/keytruda_pi.pdf 

    Output string: ::
    
        Source: URLTOKEN 
    """
    regex_sub_pair = {
        "regex_str": r"\b(http[s]*:\/\/)[^\s]+|\b(www\.)[^\s]+",
        "sub_str": " URLTOKEN ",
    }
    return regex_sub_pair


# 3
def get_special_punct_regex() -> RegexSubPair:
    """Matches a set of chosen punctuation symbols \\_,();[]#{}*"'~?!|^
    and replaces them with a single space.

    Returns
    -------

        RegexSubPair - (regex pattern, replacement string)
    
    Example
    -------

    Input string: ::
    
        wt-1, ck-7 (focal) negative; [sth] ab|cd"

    Output string: ::
    
        wt-1  ck-7  focal  negative   sth  ab cd
    """
    regex_sub_pair = {
        "regex_str": r'[\\\_,\(\);\[\]#{}\*"\'\~\?!\|\^`]',
        "sub_str": " ",
    }
    return regex_sub_pair


# 4
def get_multiple_punct_regex() -> RegexSubPair:
    """Matches multiple occurences of symbols like -, .
    and _ replaces them with a single space.

    Returns
    -------

        RegexSubPair - (regex pattern, replacement string)
    
    Example
    -------
   
    Input string: ::
    
        -----this is report ___ signature

    Output string: ::
    
         this is report   signature
    """
    regex_sub_pair = {"regex_str": r"[\-\.:\/\_]{2,}", "sub_str": " "}
    return regex_sub_pair


# 5
def get_angle_brackets_regex() -> RegexSubPair:
    """Matches a content between matching angle brackets,
    keeps the content only, removes the brackets.

    Returns
    -------

        RegexSubPair - (regex pattern, replacement string)
    
    Example
    -------

    Input string: ::
    
        <This should be fixed> But not this >90

    Output string: ::
    
        This should be fixed But not this >90
    """
    regex_sub_pair = {"regex_str": r"<(.*?)>", "sub_str": r" \1 "}
    return regex_sub_pair


# 6
def get_percent_sign_regex() -> RegexSubPair:
    """Matches the % sign and replaces
    it with a word 'percent'.

    Returns
    -------

        RegexSubPair - (regex pattern, replacement string)
    
    Example
    -------
    
    Input string: ::
    
        strong intensity >95%

    Output string: ::
    
        strong intensity >95 percent 
    """
    regex_sub_pair = {"regex_str": r"%", "sub_str": " percent "}
    return regex_sub_pair


# 7
def get_leading_digit_punctuation_regex() -> RegexSubPair:
    """Matches numeric digits at the start of a word,
    followed by punctuation and additional characters.
    Proceeds to eliminate the punctuation and inserts
    a space between the digits an the word.

    Returns
    -------

        RegexSubPair - (regex pattern, replacement string)

    Example
    --------

    Input string: ::

        13-unremarkable 1-e 22-years 
        
    Output string: ::
    
        13 unremarkable   1 e   22 years  
    """
    regex_sub_pair = {
        "regex_str": r"(\b\d{1,})([\-\.:])([a-z]+)",
        "sub_str": r" \1 \3 ",
    }
    return regex_sub_pair


# 8
def get_leading_punctuation_regex() -> RegexSubPair:
    """Matches leading punctuation and removes it.

    Returns
    -------

        RegexSubPair - (regex pattern, replacement string)
    
    Example
    -------

    Input string: ::
    
        -3a -anterior -result- :cassette

    Output string: ::
    
        3a  anterior  result-  cassette     
    """
    regex_sub_pair = {"regex_str": r"(\s[\.:\-\\])([^\s]+)", "sub_str": r" \2 "}
    return regex_sub_pair


# 9
def get_trailing_punctuation_regex() -> RegexSubPair:
    """Matches trailing punctuation and removes it.

    Returns
    -------
        RegexSubPair - (regex pattern, replacement string)

    Example
    -------

    Input string: ::
    
         -3a -anterior -result- :cassette

    Output string: ::
    
         -3a -anterior  -result :cassette 
    """
    regex_sub_pair = {"regex_str": r"([^\s]+)([\.:\-\\]\s)", "sub_str": r" \1 "}
    return regex_sub_pair


# 10
def get_words_with_punct_spacing_regex() -> RegexSubPair:
    """Matches words with hyphen, colon or period and splits them.
    Requires the words to be at least two characters in length
    to avoid splitting words like ph.d.

    Returns
    -------

        RegexSubPair - (regex pattern, replacement string)
    
    Example
    -------

    Input string: ::
    
        this-that her-2 tiff-1k description:gleason

    Output string: ::
    
        this that her-2 tiff-1k description gleason 
    """
    regex_sub_pair = {
        "regex_str": r"([a-z0-9]{2,})([\-:\.])([a-z]{2,})",
        "sub_str": r"\1 \3",
    }
    return regex_sub_pair


# 11
def get_math_spacing_regex() -> RegexSubPair:
    """Matches "math operators symbols" like ><=%:
    and adds spaces aroud them.

    Returns
    -------

        RegexSubPair - (regex pattern, replacement string)
    
    Example
    -------

    Input string: ::
    
        This is >95% 3+3=8  6/7

    Output string: ::
    
        This is  > 95 %  3 + 3 = 8  6 / 7
    """
    regex_sub_pair = {"regex_str": r"([><=+%\/&:])", "sub_str": r" \1 "}
    return regex_sub_pair


# 12
def get_dimension_spacing_regex() -> RegexSubPair:
    """Matches digits and x and adds spaces between them.

    Returns
    -------

        RegexSubPair - (regex pattern, replacement string)
    
    Example
    -------

    Input string: ::
    
        measuring 1.3x0.7x0.1 cm

    Output string: ::
    
        measuring 1.3 x 0.7 x 0.1 cm
    """
    regex_sub_pair = {"regex_str": r"(\d+[.\d]*)([x])", "sub_str": r"\1 \2 "}
    return regex_sub_pair


# 13
def get_measure_spacing_regex() -> RegexSubPair:
    """Matches measurements in mm, cm and ml provides
    proper spacing between the digits and measure.
    Also provides specing between 11th -> 11 th.

    Returns
    -------

        RegexSubPair - (regex pattern, replacement string)
    
    Example
    -------

    Input string: ::
    
        10mm histologic type 2 x 3cm. this is 3.0-cm

    Output string: ::
    
        10 mm  histologic type 2 x 3 cm . this is 3.0 cm  
    """
    regex_sub_pair = {"regex_str": r"(\d+)[-]*([cpamt][mlhc])", "sub_str": r"\1 \2 "}
    return regex_sub_pair


# 14
def get_cassettes_spacing_regex() -> RegexSubPair:
    """Matches patterns like 5e-6f
    and adds spaces around them.

    Returns
    -------
        RegexSubPair - (regex pattern, replacement string)

    Examples
    --------

    Input string: ::
    
        3e-3f

    Output string: ::
    
         3e - 3f 
    """
    regex_sub_pair = {
        "regex_str": r"(\d{1,2}[a-z])(-)(\d{1,2}[a-z])|([a-z]\d{1,2})(-)([a-z]\d{1,2})",
        "sub_str": r"\1 \2 \3 ",
    }
    return regex_sub_pair


# 15
def get_dash_digits_spacing_regex() -> RegexSubPair:
    """Matches dashes around digits and adds
    spaces around the dashes.

    Returns
    -------
        RegexSubPair - (regex pattern, replacement string)

    Example
    -------

    Input string: ::
    
        right 1:30-2:30 1.5-2.0 cm 0.9 cm for the 7-6

    Output string: ::
    
        right 1:30 - 2:30 1.5 - 2.0 cm 0.9 cm for the 7 - 6
    """
    regex_sub_pair = {
        "regex_str": r"( [\d+]*[\.:]*\d+\s*)(-)(\s*[\d+]*[\.:]*\d+)",
        "sub_str": r"\1 \2 \3",
    }
    return regex_sub_pair


# 16
def get_literals_floats_spacing_regex() -> RegexSubPair:
    """Matches character followed by a float and a word.
    This is a common formating problem.
    e.g. r18.0admission ->  r18.0 admission

    Returns
    -------
        RegexSubPair - (regex pattern, replacement string)
    
    Example
    -------

    Input string: ::
    
        r18.0admission diagnosis: bi n13.30admission

    Output string: ::
    
         r18.0 admission diagnosis: bi n13.30 admission 
    """
    regex_sub_pair = {
        "regex_str": r"([a-z]{1,2})(\d+\.\d+)([a-z]+)",
        "sub_str": r"\1\2 \3",
    }
    return regex_sub_pair


# 17
def get_fix_pluralization_regex() -> RegexSubPair:
    """Matches s character after a word
    and attaches it back to the word. This restores
    plural nouns demages by removed punctuation.

    Returns
    -------
        RegexSubPair - (regex pattern, replacement string)

    Example
    -------

    Input string: ::
    
        specimen s code s

    Output string: ::
    
         specimens codes 
    """
    regex_sub_pair = {"regex_str": r"(\b[a-z]+)(\s+)([s]\s)", "sub_str": r"\1\3"}
    return regex_sub_pair


# 18
def get_digits_words_spacing_regex() -> RegexSubPair:
    """Matches digits that are attached to the
    beginning of a word.

    Returns
    -------
        
        RegexSubPair - (regex pattern, replacement string)

    Example
    -------

    Input string: ::
    
        9837648admission

    Output string: ::
    
        9837648 admission 
    """
    regex_sub_pair = {"regex_str": r"(\s\d{1,})([a-z]{2,}\s)", "sub_str": r"\1 \2"}
    return regex_sub_pair


# 19
def get_phone_number_regex() -> RegexSubPair:
    """Matches any phone number that consists
    of 10 digits with delimeters.

    Returns
    -------

        RegexSubPair - (regex pattern, replacement string)
    
    Example
    -------

    Input string: ::
    
        Ph: (123) 456 7890. It is (123)4567890.

    Output string: ::
    
        Ph:  PHONENUMTOKEN . It is  PHONENUMTOKEN .
    """
    regex_sub_pair = {
        "regex_str": r"\(*\d{3}\)*[-, ]*\d{3}[-, ]*\d{4}",
        "sub_str": " PHONENUMTOKEN ",
    }
    return regex_sub_pair


# 20
def get_dates_regex() -> RegexSubPair:
    """Matches dates of specified formats.

    Returns
    -------

        RegexSubPair - (regex pattern, replacement string)
    
    Example
    -------

    Input string: ::
    
       co: 03/09/2021 1015 completed: 03/10/21 at 3:34.

    Output string: ::
    
        co:  DATETOKEN completed:  DATETOKEN .
    """
    regex_list = [
        r"\d{1,2}\s*[\/,-\.]\s*\d{1,2}\s*[\/,-\.]\s*\d{2,4}\s*[at\s\-]*[\d{1,2}\s*[:\s*\d{1,2}]+]*(?:\s*[pa][m])*",
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*\d{1,2}\s*\d{2,4}",
        r"\b\d{1,2}\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*\d{2,4}",
        r"\d{1,2}-(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)-\d{2}\s*\d{1,2}[:\d{1,2}]+(?:\s*[pa][m])",
    ]
    consolidated_regex = f"{'|'.join(regex_list)}"
    regex_sub_pair = {"regex_str": consolidated_regex, "sub_str": " DATETOKEN "}
    return regex_sub_pair


# 21
def get_time_regex() -> RegexSubPair:
    """Matches time of format 11:20 am
    or 1.30pm or 9:52:07AM.

    Returns
    -------

        RegexSubPair - (regex pattern, replacement string)
    
    Example
    -------
    
    Input string: ::
    
        at 11:12 pm or 11.12am

    Output string: ::
    
        at  TIMETOKEN  or  TIMETOKEN  
    """
    regex_list = [
        r"(\d{1,2}\s*([:.]\s*\d{2}){1,2}\s*[ap]\.*[m]\.*)",
        r"\d{2}\s*[ap]\.*[m]\.*",
        r"[0-2][0-9]:[0-5][1-9]",
    ]
    consolidated_regex = f"{'|'.join(regex_list)}"
    regex_sub_pair = {"regex_str": consolidated_regex, "sub_str": " TIMETOKEN "}
    return regex_sub_pair


# 22
def get_address_regex() -> RegexSubPair:
    """Matches any address of format
    num (street name) in 1 to 6 words 2-letter state
    and short or long zip code.

    Returns
    -------

        RegexSubPair - (regex pattern, replacement string)
    
    Example
    -------

    Input string: ::
    
        1034 north 500 west provo ut 84604-3337

    Output string: ::
    
          ADDRESSTOKEN  
    """
    regex_sub_pair = {
        "regex_str": r"\d+\s([0-9a-z.]+[\s,]+){1,6}[a-z]{2}[./\s+]*\d{5}(-\d{4})*",
        "sub_str": " ADDRESSTOKEN ",
    }
    return regex_sub_pair


# 23
def get_dimensions_regex() -> RegexSubPair:
    """Matches 2D or 3D dimension measurements
    and adds spaces around them.

    Returns
    -------
        RegexSubPair - (regex pattern, replacement string)

    Example
    -------

    Input string: ::
    
        3.5 x 2.5 x 9.0 cm and 33 x 6.5 cm

    Output string: ::
    
          DIMENSIONTOKEN  cm and  DIMENSIONTOKEN  cm
    """
    regex_list = [
        r"\d+\.*\d*\s*x\s*\d+\.*\d*\s*x\s*\d+\.*\d*",
        r"\d+\.*\d*\s*x\s*\d+\.*\d*",
    ]
    consolidated_regex = f"{'|'.join(regex_list)}"
    regex_sub_pair = {"regex_str": consolidated_regex, "sub_str": " DIMENSIONTOKEN "}
    return regex_sub_pair


# 24
def get_specimen_regex() -> RegexSubPair:
    """Matches marking of a pathology speciman.

    Returns
    -------

        RegexSubPair - (regex pattern, replacement string)
    
    Example
    -------

    Input string: ::
    
        for s-21-009345 sh-22-0011300

    Output string: ::
    
         for  SPECIMENTOKEN   SPECIMENTOKEN  
    """
    regex_sub_pair = {
        "regex_str": r"[a-z]{1,3}[-]*\d{2}[-]\d{3,}[-]*",
        "sub_str": " SPECIMENTOKEN ",
    }
    return regex_sub_pair


# 25
def get_decimal_segmented_numbers_regex() -> RegexSubPair:
    """Matches combinations of digits and periods or dashes.

    Returns
    -------

        RegexSubPair - (regex pattern, replacement string)
    
    Example
    -------
    
    Input string: ::
    
         1.78.9.87

    Output string: ::
    
          DECIMALSEGMENTEDNUMBERTOKEN  
    """
    regex_sub_pair = {
        "regex_str": r"\d+[\.\-]\d+([\.\-]\d+)+",
        "sub_str": " DECIMALSEGMENTEDNUMBERTOKEN ",
    }
    return regex_sub_pair


# 26
def get_large_digits_seq_regex() -> RegexSubPair:
    """Matches large sequences of digits and replaces it.

    Returns
    -------

        RegexSubPair - (regex pattern, replacement string)
    
    Example
    -------
    
    Input string: ::
    
         456123456 
    
    Output string: ::
    
         DIGITSEQUENCETOKEN 
    """
    regex_sub_pair = {"regex_str": r"\s\d{3,}\s", "sub_str": " DIGITSEQUENCETOKEN "}
    return regex_sub_pair


# 27
def get_large_float_seq_regex() -> RegexSubPair:
    """Matches large floats and replace them.

    Returns
    -------

        RegexSubPair - (regex pattern, replacement string)
    
    Example
    -------
    
    Input string: ::
    
        456 123456.783

    Output string: ::
    
         456 LARGEFLOATTOKEN  
    """
    regex_sub_pair = {"regex_str": r"\s\d{2,}\.\d{1,}", "sub_str": " LARGEFLOATTOKEN "}
    return regex_sub_pair


# 28
def get_trunc_decimals_regex() -> RegexSubPair:
    """Matches floats and keeps only first decimal.

    Returns
    -------

        RegexSubPair - (regex pattern, replacement string)
    
    Example
    -------
    
    Input string: ::

        1.78  9.87 - 8.99

    Output string: ::
    
        1.7  9.8 - 8.9 
    """
    regex_sub_pair = {"regex_str": r"\s(\d+)(\.)(\d)(\d+)*\s", "sub_str": r" \1\2\3 "}
    return regex_sub_pair


# 29
def get_cassette_name_regex() -> RegexSubPair:
    """Matches cassettes markings of
    the specified format:

    Returns
    --------

        RegexSubPair - (regex pattern, replacement string)
    
    Example
    -------

    Input string: ::
    
        block:  1-e 

    Output string: ::
    
        block:  CASSETTETOKEN 
    """
    regex_list = [
        r"\s\d{1,2}[\-]*[a-z]{1,2}\s",
        r"\b[a-z][\-]*\d{1}\s",
        r"\s[a-z]\d{1,2}-\d{1,2}\s",
    ]
    consolidated_regex = f"{'|'.join(regex_list)}"
    regex_sub_pair = {"regex_str": consolidated_regex, "sub_str": " CASSETTETOKEN "}
    return regex_sub_pair


# 30
def get_duration_regex() -> RegexSubPair:
    """Matches duration specimen was treated:

    Returns
    -------

        RegexSubPair - (regex pattern, replacement string)
    
    Example
    -------
    
    Input string: ::
    
        32d0909091

    Output string: ::
    
         DURATIONTOKEN 
    """
    regex_sub_pair = {
        "regex_str": r" \d{1,2}d\d{6,9}[.\s]*",
        "sub_str": " DURATIONTOKEN ",
    }
    return regex_sub_pair


# 31
def get_letter_num_seq_regex() -> RegexSubPair:
    """Matches a character followed directly
    by 6 to 10 digits:

    Returns
    -------

        RegexSubPair - (regex pattern, replacement string)
    
    Example
    -------
    
    Input string: ::
    
        c001234567

    Output string: ::
    
         LETTERDIGITSTOKEN 
    """
    regex_sub_pair = {
        "regex_str": r"\b[a-z]\d{6,10}[.\s]*",
        "sub_str": " LETTERDIGITSTOKEN ",
    }
    return regex_sub_pair


#   LAST REGEX that removed additional spaces
def get_spaces_regex() -> RegexSubPair:
    """Matches additional spaces (artifact of applying
    other regex), matches not needed periods that can be removed.

    
    Returns
    -------

        RegexSubPair - (regex pattern, replacement string)
    
    Example
    -------
    
    Input string: ::
    
        located around lower arm specimen   date

    Output string: ::
    
        located around lower arm specimen date
    """
    regex_sub_pair = {"regex_str": r"\s{2,}|\\n", "sub_str": " "}
    return regex_sub_pair
