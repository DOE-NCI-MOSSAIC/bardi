from typing import List, Union

from pyarrow import Table, large_list, large_string, list_, string


def validate_pyarrow_table(data: Table) -> None:
    """Confirm the data table passed is a PyArrow Table

    Keyword Arguments:
        data: PyArrow Table
            The table containing the data (passed by reference). The schema
            of this table will be checked to confirm string types. The table
            will also be confirmed to be a PyArrow Table.

    Raises:
        TypeError if the data table is not a PyArrow table
    """
    if isinstance(data, Table):
        pass
    else:
        raise TypeError("Data must be provided in a pyArrow.Table")


def validate_str_cols(fields: Union[List[str], str], data: Table) -> None:
    """Confirm the columns specified in the supplied data table are string columns

    Keyword Arguments:
        fields: Union[List[str], str]
            The field or fields that will be used in the method to
            be validated as string types

    Raises:
        TypeError if the columns supplied in the 'fields' keyword argument
        are not of PyArrow types string or large_string.
    """
    # Gather the string fields in the table
    all_pyarrow_schema_fields = [data.schema.field(name)
                                 for name in data.column_names]
    str_fields = [field.name for field in all_pyarrow_schema_fields
                  if field.type in [large_string(), string()]]

    if isinstance(fields, str):
        # If a single field was passed in to "fields" of the object,
        # check to ensure it is a string
        if fields not in str_fields:
            raise TypeError('The field indicated is not a string'
                            ' field and cannot be utilized with'
                            ' this method.')
        fields = [fields]
    elif isinstance(fields, list):
        # If a list of fields was passed in to "fields" of the object,
        # ensure they are all strings
        for field in fields:
            if field not in str_fields:
                raise TypeError(f'The field, "{field}", indicated is'
                                ' not a string field and cannot be'
                                ' utilized with this method')


def validate_list_str_cols(fields: Union[List[str], str], data: Table) -> None:
    """Confirm the columns specified in the supplied data table are list[string] columns

    Keyword Arguments:
        fields: Union[List[str], str]
            The field or fields that will be used in the method to
            be validated as string types

    Raises:
        TypeError if the columns supplied in the 'fields' keyword argument
        are not of PyArrow types list[string] or list[large_string].
    """
    # Gather the list fields in the table
    all_pyarrow_schema_fields = [data.schema.field(name)
                                 for name in data.column_names]
    list_str_fields = [field.name for field in all_pyarrow_schema_fields
                       if field.type in [large_list(large_string()),
                                         large_list(string()),
                                         list_(string()),
                                         list_(large_string())]]

    if isinstance(fields, str):
        # If a single field was passed in to "fields" of the object,
        # check to ensure it is a string
        if fields not in list_str_fields:
            raise TypeError('The field indicated is not a string'
                            ' field and cannot be utilized with'
                            ' this method.')
        fields = [fields]
    elif isinstance(fields, list):
        # If a list of fields was passed in to "fields" of the object,
        # ensure they are all strings
        for field in fields:
            if field not in list_str_fields:
                raise TypeError(f'The field, "{field}", indicated is'
                                ' not a string field and cannot be'
                                ' utilized with this method')
