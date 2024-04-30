from typing import List

import polars as pl


def retain_inputs(
    df: pl.DataFrame, retain_input_fields: bool, fields: List[str], step_name: str
) -> pl.DataFrame:
    """Reusable function to apply each regex substitution
    normalization to each string field

    Parameters
    ----------

    df : pl.DataFrame
        a DataFrame containing the columns specified in the `fields` attribute
    retain_input_fields : bool
        should the original input fields be retained in a separate, renamed column
    fields : List[str]
        The name of the column(s) to be retained
    step_name : str
        The name of the step calling this function. Used to name the retained field.

    Returns
    -------
    pl.DataFrame
        a DataFrame with the specified fields
    """
    if retain_input_fields:
        df = df.with_columns(
            [pl.col(field).alias(f"{step_name}_input__{field}") for field in fields]
        )

    return df
