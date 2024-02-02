from math import ceil
from typing import List

import pyarrow as pa


def chunk_pyarrow_table(
    data: pa.Table, row_count: int, min_batches: int
) -> List[pa.Table]:
    # PyArrow batch size is the maximum number of rows in a batch, so this is
    # calculated based on total number of rows and specified number of batches
    # in the batch_count keyword argument
    max_batch_size = ceil(row_count / min_batches)

    # The dataset.to_batches method returns a list of record_batches
    # (zero copy operation, just exposes data through different API)
    batches = data.to_batches(max_chunksize=max_batch_size)

    # The following generator takes each record batch and turns it into a PyArrow Table
    # creating a list of tables
    tables = [pa.Table.from_batches([batch]) for batch in batches]

    return tables
