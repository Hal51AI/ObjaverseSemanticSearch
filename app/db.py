import sqlite3
import tempfile
from typing import List

import pandas as pd


def create_db(
    df: pd.DataFrame,
    table_name: str = "objaverse",
    col_name: str = "top_aggregate_caption",
) -> str:
    """
    Create a sqlite database in memory with an index on the text column
    specified by `col_name`

    Parameters
    ==========
    df: pd.DataFrame
        The dataframe which we want to convert to a sqlite table
    table_name: str
        The name of the table to be created in the db
    col_name: str
        The text column to create an index over

    Returns
    =======
    str
        The path to the database file created
    """
    temp_db = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
    with sqlite3.connect(temp_db.name) as con:
        cur = con.cursor()
        df.to_sql(table_name, con, if_exists="replace", index=False)

        cur.execute(f"CREATE INDEX idx_caption ON {table_name} ({col_name});")
        con.commit()

    return temp_db.name


def query_db_match(
    db_path: str,
    match_list: List[str],
    table_name: str = "objaverse",
    col_name: str = "top_aggregate_caption",
) -> pd.DataFrame:
    """
    Query a sqlite database to find exact matches on a column from
    the `match_list`

    Parameters
    ==========
    db_path: str
        A path to the database
    match_list: List[str]
        A list of strings to find exact matches
    table_name: str
        The name of the table to query
    col_name: str
        The column name in the table to match from

    Returns
    =======
    pd.DataFrame
        A dataframe created out of the matches from the database
    """
    with sqlite3.connect(db_path) as con:
        query_str = f"""
            SELECT *
            FROM {table_name}
            WHERE {col_name}
            IN ({",".join(["?" for _ in match_list])})
        """
        return pd.read_sql_query(query_str, con, params=match_list)
