import unittest
from json import dumps
from pathlib import Path

from duckdb import connect
from pandas import DataFrame
from pyarrow import Table, table, array

from gaudi.data import data_handlers


class TestDataHandlers(unittest.TestCase):
    """Tests the functionality of the functions in gaudi.data.data_handlers
    that create GAuDI Dataset objects from various sources"""

    def test_dataset_from_file(self):
        """Set of tests to ensure that the data_handlers.from_file
        function is correctly loading data and creating
        GAuDI Dataset objects"""

        # ======== Set-up ========
        repo_path = Path().resolve()

        d = {'col1': [1, 2, 3, 4], 'col2': ['str1', 'str2', 'str3', 'str4']}
        test_df = DataFrame(data=d)

        # ======== Parquet Filetype ========
        # Parquet Set-up
        parquet_path = f'{repo_path}/tests/test_data/test_data.parquet'
        test_df.to_parquet(parquet_path, engine='pyarrow', index=False)

        # Non-Chunked Dataset Test
        p_dataset_obj = data_handlers.from_file(source=parquet_path,
                                                format='parquet')

        # GAuDI Dataset object was created and returned?
        self.assertTrue(isinstance(p_dataset_obj, data_handlers.Dataset),
                        ('Object created and/or returned by the function'
                         ' was not a GAuDI Dataset object'))
        # Data is an Arrow Table?
        self.assertTrue(isinstance(p_dataset_obj.data, Table),
                        ('The data referenced in the object is not'
                         ' an Arrow Table'))
        # Recorded data length matches the source?
        self.assertEqual(p_dataset_obj.origin_row_count,
                         test_df.shape[0],
                         ('Recorded origin row count in the Dataset object'
                          ' does not match the test data'))
        # Data length matches the source?
        self.assertEqual(p_dataset_obj.data.num_rows, test_df.shape[0],
                         ('Data length in Arrow Table does not match'
                          ' the test data'))
        # Columns are correct?
        self.assertEqual(p_dataset_obj.data.column_names,
                         list(test_df.columns),
                         ('Columns of Arrow Table do not match'
                          ' the columns in the test data'))
        # Data types are correct
        # Format of source recorded correctly?
        self.assertEqual(p_dataset_obj.origin_format, "parquet",
                         ('Origin format incorrectly recorded'
                          ' in the GAuDI Dataset object'))
        # Data source path recorded correctly?
        self.assertEqual(p_dataset_obj.origin_file_path, parquet_path,
                         ('Origin file path recorded does'
                          ' not match the test data path'))

        # Chunked Dataset Test
        p_chunked_dataset_obj = data_handlers.from_file(source=parquet_path,
                                                        format='parquet',
                                                        min_batches=2)

        # GAuDI Dataset object was created and returned?
        self.assertTrue(isinstance(p_chunked_dataset_obj,
                                   data_handlers.Dataset),
                        ('Object created and/or returned by the function'
                         ' (with batches) was not a GAuDI Dataset object'))
        # Data is a List of Arrow Tables?
        self.assertTrue(isinstance(p_chunked_dataset_obj.data, list),
                        ('The data attribute in the object'
                         ' is not referencing a list'))
        for test_chunk in p_chunked_dataset_obj.data:
            self.assertTrue(isinstance(test_chunk, Table),
                            ('The contents of the data list'
                             ' are not Arrow Tables'))
        # Total length of data matches the source data?
        total_data_length = sum([table.num_rows
                                 for table in p_chunked_dataset_obj.data])
        self.assertEqual(total_data_length,
                         p_chunked_dataset_obj.origin_row_count,
                         ('Total length of the data in the list'
                          ' does not match recorded length'))
        self.assertEqual(total_data_length, test_df.shape[0],
                         ('Total length of the data in the list'
                          ' does not match the query results'))
        # Columns are correct?
        for test_chunk in p_chunked_dataset_obj.data:
            self.assertEqual(test_chunk.column_names, list(test_df.columns),
                             ('Columns of Arrow Table do not match'
                              ' the columns in the query results'))
        # Format of source recorded correctly?
        self.assertEqual(p_chunked_dataset_obj.origin_format, "parquet",
                         ('Origin format incorrectly recorded in'
                          ' the GAuDI Dataset object'))
        # Data source path recorded correctly?
        self.assertEqual(p_dataset_obj.origin_file_path, parquet_path,
                         ('Origin file path recorded'
                          ' does not match the test data path'))

        # ======== CSV Filetype ========
        # CSV Set-up
        csv_path = f'{repo_path}/tests/test_data/test_data.csv'
        test_df.to_csv(csv_path, index=False)

        # Non-Chunked Dataset Test
        c_dataset_obj = data_handlers.from_file(source=csv_path, format='csv')

        # GAuDI Dataset object was created and returned?
        self.assertTrue(isinstance(c_dataset_obj, data_handlers.Dataset),
                        ('Object created and/or returned by the function'
                         ' was not a GAuDI Dataset object'))
        # Data is an Arrow Table?
        self.assertTrue(isinstance(c_dataset_obj.data, Table),
                        ('The data referenced in the object'
                         ' is not an Arrow Table'))
        # Recorded data length matches the source?
        self.assertEqual(c_dataset_obj.origin_row_count, test_df.shape[0],
                         ('Recorded origin row count in the Dataset object'
                          ' does not match the test data'))
        # Data length matches the source?
        self.assertEqual(c_dataset_obj.data.num_rows, test_df.shape[0],
                         ('Data length in Arrow Table'
                          ' does not match the test data'))
        # Columns are correct?
        self.assertEqual(c_dataset_obj.data.column_names,
                         list(test_df.columns),
                         ('Columns of Arrow Table do not match'
                          ' the columns in the test data'))
        # Data types are correct
        # Format of source recorded correctly?
        self.assertEqual(c_dataset_obj.origin_format, "csv",
                         ('Origin format incorrectly recorded'
                          ' in the GAuDI Dataset object'))
        # Data source path recorded correctly?
        self.assertEqual(c_dataset_obj.origin_file_path, csv_path,
                         ('Origin file path recorded does not match'
                          ' the test data path'))

        # Chunked Dataset Test
        c_chunked_dataset_obj = data_handlers.from_file(source=csv_path,
                                                        format='csv',
                                                        min_batches=2)

        # GAuDI Dataset object was created and returned?
        self.assertTrue(isinstance(c_chunked_dataset_obj,
                                   data_handlers.Dataset),
                        ('Object created and/or returned by'
                         ' the function (with batches) was'
                         ' not a GAuDI Dataset object'))
        # Data is a List of Arrow Tables?
        self.assertTrue(isinstance(c_chunked_dataset_obj.data, list),
                        ('The data attribute in the object'
                         ' is not referencing a list'))
        for test_chunk in c_chunked_dataset_obj.data:
            self.assertTrue(isinstance(test_chunk, Table),
                            ('The contents of the data list'
                             ' are not Arrow Tables'))
        # Total length of data matches the source data?
        total_data_length = sum([table.num_rows
                                 for table in c_chunked_dataset_obj.data])
        self.assertEqual(total_data_length,
                         c_chunked_dataset_obj.origin_row_count,
                         ('Total length of the data in the list'
                          ' does not match recorded length'))
        self.assertEqual(total_data_length, test_df.shape[0],
                         ('Total length of the data in the list'
                          ' does not match the query results'))
        # Columns are correct?
        for test_chunk in c_chunked_dataset_obj.data:
            self.assertEqual(test_chunk.column_names, list(test_df.columns),
                             ('Columns of Arrow Table do not match the columns'
                              ' in the query results'))
        # Format of source recorded correctly?
        self.assertEqual(c_chunked_dataset_obj.origin_format, "csv",
                         ('Origin format incorrectly recorded'
                          ' in the GAuDI Dataset object'))
        # Data source path recorded correctly?
        self.assertEqual(c_dataset_obj.origin_file_path, csv_path,
                         ('Origin file path recorded does not match'
                          ' the test data path'))

    def test_dataset_from_duckdb(self):
        """Set of tests to ensure that the data_handlers.from_duckdb
        function is correctly querying and creating
        GAuDI Dataset objects"""

        # ======== Set-up ========
        # Connect to the test database file
        test_db_path = f'{Path().resolve()}/tests/test_data/test_db.duckdb'
        test_conn = connect(test_db_path)

        # Create test table
        setup_query = """
                      CREATE OR REPLACE TABLE test(
                          col1 INTEGER,
                          col2 VARCHAR
                      );

                      INSERT INTO test VALUES (1, 'str1');
                      INSERT INTO test VALUES (2, 'str2');
                      INSERT INTO test VALUES (3, 'str3');
                      INSERT INTO test VALUES (4, 'str4');
                      """
        test_conn.execute(setup_query)

        # ======== Testing ========
        test_query = """
                     SELECT
                         col1,
                         col2
                     FROM test;
                     """

        # Returning test query results as a Pandas DataFrame
        # for comparison to what the data_handler function is doing
        test_df = test_conn.execute(test_query).fetch_df()

        # A new connection to the db is created in the data_handler
        # function, so closing the one used for setup
        test_conn.close()

        # ======== Non-Chunked Dataset Tests ========
        dataset_obj = data_handlers.from_duckdb(path=test_db_path,
                                                query=test_query)

        # GAuDI Dataset object was created and returned?
        self.assertTrue(isinstance(dataset_obj, data_handlers.Dataset),
                        ('Object created and/or returned by the function'
                         ' was not a GAuDI Dataset object'))
        # Data is an Arrow Table?
        self.assertTrue(isinstance(dataset_obj.data, Table),
                        ('The data referenced in the object'
                         ' is not an Arrow Table'))
        # Recorded data length matches the source?
        self.assertEqual(dataset_obj.origin_row_count, test_df.shape[0],
                         ('Recorded origin row count in the Dataset object'
                          ' does not match the query results'))
        # Data length matches the source?
        self.assertEqual(dataset_obj.data.num_rows, test_df.shape[0],
                         ('Data length in Arrow Table does not match'
                          ' the query results'))
        # Columns are correct?
        self.assertEqual(dataset_obj.data.column_names, list(test_df.columns),
                         ('Columns of Arrow Table do not match the columns'
                          ' in the query results'))
        # Data types are correct
        # Format of source recorded correctly?
        self.assertEqual(dataset_obj.origin_format, "duckdb",
                         ('Origin format incorrectly recorded in'
                          ' the GAuDI Dataset object'))
        # Recorded query matches the test query?
        self.assertEqual(dataset_obj.origin_query, test_query,
                         ('Origin query recorded incorrectly in the Dataset'
                          ' object. It does not match the test query.'))

        # ======== Chunked Dataset Tests ========
        chunked_dataset_obj = data_handlers.from_duckdb(path=test_db_path,
                                                        query=test_query,
                                                        min_batches=2)

        # GAuDI Dataset object was created and returned?
        self.assertTrue(isinstance(chunked_dataset_obj, data_handlers.Dataset),
                        ('Object created and/or returned by the function'
                         ' (with batches) was not a GAuDI Dataset object'))
        # Data is a List of Arrow Tables?
        self.assertTrue(isinstance(chunked_dataset_obj.data, list),
                        ('The data attribute in the object'
                         ' is not referencing a list'))
        for test_chunk in chunked_dataset_obj.data:
            self.assertTrue(isinstance(test_chunk, Table),
                            ('The contents of the data list'
                             ' are not Arrow Tables'))
        # Total length of data matches the source data?
        total_data_length = sum([table.num_rows
                                 for table in chunked_dataset_obj.data])
        self.assertEqual(total_data_length,
                         chunked_dataset_obj.origin_row_count,
                         ('Total length of the data in the list'
                          ' does not match recorded length'))
        self.assertEqual(total_data_length, test_df.shape[0],
                         ('Total length of the data in the list'
                          ' does not match the query results'))
        # Columns are correct?
        for test_chunk in chunked_dataset_obj.data:
            self.assertEqual(test_chunk.column_names,
                             list(test_df.columns),
                             ('Columns of Arrow Table do not match'
                              ' the columns in the query results'))
        # Format of source recorded correctly?
        self.assertEqual(dataset_obj.origin_format, "duckdb",
                         ('Origin format incorrectly recorded'
                          ' in the GAuDI Dataset object'))
        # Recorded query matches the test query?
        self.assertEqual(dataset_obj.origin_query, test_query,
                         ('Origin query recorded incorrectly in the Dataset'
                          ' object. It does not match the test query.'))

    def test_dataset_from_pandas(self):
        """Set of tests to ensure that the data_handlers.from_pandas
        function is correctly creating GAuDI Dataset objects"""

        # Create a test Pandas DataFrame
        d = {'col1': [1, 2, 3, 4], 'col2': ['str1', 'str2', 'str3', 'str4']}
        df = DataFrame(data=d)

        # ======== Non-Chunked Dataset Tests ========
        dataset_obj = data_handlers.from_pandas(df)

        # GAuDI Dataset object was created and returned
        self.assertTrue(isinstance(dataset_obj, data_handlers.Dataset),
                        ('Object created and/or returned'
                         ' by the function was incorrect'))
        # Data is an Arrow Table
        self.assertTrue(isinstance(dataset_obj.data, Table),
                        ('The data referenced in the object'
                         ' is not an Arrow Table'))
        # Recorded data length matches the source
        self.assertEqual(dataset_obj.origin_row_count, df.shape[0],
                         ('Recorded origin row count'
                          ' does not match the Pandas DataFrame'))
        # Data length matches the source
        self.assertEqual(dataset_obj.data.num_rows, df.shape[0],
                         ('Data length in Table does not match'
                         ' Pandas DataFrame'))
        # Columns are correct
        self.assertEqual(dataset_obj.data.column_names, list(df.columns),
                         ('Columns of Table do not match'
                          ' the columns of the DataFrame'))
        # Data types are correct
        # Format of source recorded correctly
        self.assertEqual(dataset_obj.origin_format, "pandas",
                         ('Origin format incorrectly recorded'
                          ' in the GAuDI Dataset object'))

        # ======== Chunked Dataset Tests ========
        chunked_dataset_obj = data_handlers.from_pandas(df, min_batches=2)

        # GAuDI Dataset object was created and returned
        self.assertTrue(isinstance(chunked_dataset_obj, data_handlers.Dataset),
                        ('Object created and/or returned by the function'
                         ' (with batches) was incorrect'))
        # Data is a List of Arrow Tables
        self.assertTrue(isinstance(chunked_dataset_obj.data, list),
                        ('The data attribute in the object'
                         ' is not referencing a list'))
        for test_chunk in chunked_dataset_obj.data:
            self.assertTrue(isinstance(test_chunk, Table),
                            ('The contents of the data list'
                             ' are not Arrow Tables'))
        # Total length of data matches the source data
        total_data_length = sum([table.num_rows
                                 for table in chunked_dataset_obj.data])
        self.assertEqual(total_data_length,
                         chunked_dataset_obj.origin_row_count,
                         ('Total length of the data in the list'
                          ' does not match recorded length'))
        self.assertEqual(total_data_length, df.shape[0],
                         ('Total length of the data in the list'
                          ' does not match DataFrame length'))
        # Columns are correct
        for test_chunk in chunked_dataset_obj.data:
            self.assertEqual(test_chunk.column_names, list(df.columns),
                             ('Columns of Table do not match'
                              ' the columns of the DataFrame'))
        # Format of source recorded correctly
        self.assertEqual(dataset_obj.origin_format, "pandas",
                         ('Origin format incorrectly recorded'
                          ' in the GAuDI Dataset object'))

    def test_dataset_from_pyarrow(self):
        """Set of tests to ensure that the data_handlers.from_pyarrow
        function is correctly creating a GAuDI Dataset object"""

        # Create a test PyArrow Table
        col1 = array([1, 2, 3, 4])
        col2 = array(['str1', 'str2', 'str3', 'str4'])
        names = ['col1', 'col2']
        test_table = table([col1, col2], names=names)

        # ======== Non-Chunked Dataset Tests ========
        dataset_obj = data_handlers.from_pyarrow(test_table)

        # GAuDI Dataset object was created and returned
        self.assertTrue(isinstance(dataset_obj, data_handlers.Dataset),
                        ('Object created and/or returned'
                         ' by the function was incorrect'))
        # Data is an Arrow Table
        self.assertTrue(isinstance(dataset_obj.data, Table),
                        ('The data referenced in the object'
                         ' is not an Arrow Table'))
        # Recorded data length matches the source
        self.assertEqual(dataset_obj.origin_row_count,
                         test_table.num_rows,
                         ('Recorded origin row count does not match'
                          ' the Pandas DataFrame'))
        # Data length matches the source
        self.assertEqual(dataset_obj.data.num_rows, test_table.num_rows,
                         ('Data length in Table does not match'
                          ' Pandas DataFrame'))
        # Columns are correct
        self.assertEqual(dataset_obj.data.column_names,
                         test_table.column_names,
                         ('Columns of Table do not match the columns'
                          ' of the DataFrame'))
        # Data types are correct
        # Format of source recorded correctly
        self.assertEqual(dataset_obj.origin_format, "pyarrow",
                         ('Origin format incorrectly recorded'
                          ' in the GAuDI Dataset object'))

        # ======== Chunked Dataset Tests ========
        chunked_dataset_obj = data_handlers.from_pyarrow(test_table,
                                                         min_batches=2)

        # GAuDI Dataset object was created and returned
        self.assertTrue(isinstance(chunked_dataset_obj,
                                   data_handlers.Dataset),
                        ('Object created and/or returned by the function'
                         ' (with batches) was incorrect'))
        # Data is a List of Arrow Tables
        self.assertTrue(isinstance(chunked_dataset_obj.data, list),
                        ('The data attribute in the object'
                         ' is not referencing a list'))
        for test_chunk in chunked_dataset_obj.data:
            self.assertTrue(isinstance(test_chunk, Table),
                            ('The contents of the data list'
                             ' are not Arrow Tables'))
        # Total length of data matches the source data
        total_data_length = sum([chunk.num_rows
                                 for chunk in chunked_dataset_obj.data])
        self.assertEqual(total_data_length,
                         chunked_dataset_obj.origin_row_count,
                         ('Total length of the data in the list does not'
                          ' match recorded length'))
        self.assertEqual(total_data_length, test_table.num_rows,
                         ('Total length of the data in the list does not'
                          ' match DataFrame length'))
        # Columns are correct
        for test_chunk in chunked_dataset_obj.data:
            self.assertEqual(test_chunk.column_names,
                             test_table.column_names,
                             ('Columns of Table do not match'
                              ' the columns of the DataFrame'))
        # Format of source recorded correctly
        self.assertEqual(dataset_obj.origin_format, "pyarrow",
                         ('Origin format incorrectly recorded'
                          ' in the GAuDI Dataset object'))

    def test_dataset_from_json(self):
        """Set of tests to ensure that the data_handlers.from_json
        function is correctly creating a GAuDI Dataset object"""

        # Create a test JSON object
        d = {'col1': 1, 'col2': 'str1'}
        json_obj = dumps(d)

        dataset_obj = data_handlers.from_json(json_obj)

        # GAuDI Dataset object was created and returned
        self.assertTrue(isinstance(dataset_obj, data_handlers.Dataset),
                        ('Object created and/or returned'
                         ' by the function was incorrect'))
        # Data is an Arrow Table
        self.assertTrue(isinstance(dataset_obj.data, Table),
                        ('The data referenced in the object'
                         ' is not an Arrow Table'))
        # Recorded data length matches the source
        self.assertEqual(dataset_obj.origin_row_count, 1,
                         ('Recorded origin row count does'
                          ' not match the Pandas DataFrame'))
        # Data length matches the source
        self.assertEqual(dataset_obj.data.num_rows, 1,
                         ('Data length in Table does not match'
                          ' Pandas DataFrame'))
        # Columns are correct
        self.assertEqual(dataset_obj.data.column_names, list(d.keys()),
                         ('Columns of Table do not match the columns'
                          ' of the DataFrame'))
        # Data types are correct
        # Format of source recorded correctly
        self.assertEqual(dataset_obj.origin_format, "json",
                         ('Origin format incorrectly recorded'
                          ' in the GAuDI Dataset object'))


if __name__ == "__main__":
    unittest.main()
