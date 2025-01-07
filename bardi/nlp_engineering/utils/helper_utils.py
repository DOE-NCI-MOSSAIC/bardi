import json
from typing import List

import polars as pl


def existing_split_mapping(
    data_path: str, format: str, mapping_write_path: str, unique_record_cols: List[str]
):
    """Extract the split from an existing data_fold0.csv to
    duplicate the same split for pre-processing comparison

    Combines and hashes the columns in unique_record_cols,
    referred to as composite_record_id, and creates a json
    mapping {composite_record_id: split}

    Parameters
    ----------

    data_path : str
        full filepath to the existing data file
    format : str
        format of the existing file (currently only 'csv')
    mapping_write_path : str
        where the created mapping should be saved to
    unique_record_cols : List[str]
        the set of columns that create a distinct record
        if only one column, still provide in a list

    Returns
    -------
    dict
        dictionary assigning each unique row to a given split
    """
    if format == "csv":
        df = pl.scan_csv(source=data_path)
    elif format == "parquet":
        df = pl.scan_parquet(source=data_path)
    else:
        raise NotImplementedError("Support for this format has not been implemented yet.")

    data_fold0_df = (
        df.select(*unique_record_cols, "split")
        .with_columns(
            [
                pl.concat_str([*unique_record_cols])
                .hash()
                .cast(pl.Utf8())
                .alias("composite_record_id")
            ]
        )
        .select("composite_record_id", "split")
    ).collect()

    mappings = data_fold0_df.to_struct(name="mappings").to_list()

    mapping = {d["composite_record_id"]: d["split"] for d in mappings}

    with open(mapping_write_path, "w") as f:
        json.dump(mapping, f, indent=4)
    f.close()
    return mapping
