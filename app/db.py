from itertools import repeat
from typing import List

import aiocsv
import aiofiles
import aiosqlite
import ciso8601
import objaverse
import pandas as pd
from starlette.concurrency import run_in_threadpool


async def create_db(captions_file: str, database_path: str) -> str:
    # Read from captions file asynchronously
    async with aiofiles.open(captions_file) as fp:
        objaverse_items = [i async for i in aiocsv.AsyncDictReader(fp, delimiter=";")]

    async with aiosqlite.connect(database_path) as conn:
        # Create objaverse table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS objaverse (
                object_uid            TEXT PRIMARY KEY,
                top_aggregate_caption TEXT,
                probability           REAL
            );
        """)
        # Create index on captions for fast retrieval
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_caption ON objaverse (
                top_aggregate_caption
            );
        """)
        # Create metadata table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                uid             TEXT PRIMARY KEY,
                name            TEXT,
                staffpickedAt   TEXT,
                viewCount       INTEGER,
                likeCount       INTEGER,
                animationCount  INTEGER,
                description     TEXT,
                faceCount       INTEGER,
                vertexCount     INTEGER,
                license         TEXT,
                publishedAt     DATETIME,
                createdAt       DATETIME,
                isAgeRestricted INTEGER,
                userId          TEXT,
                userName        TEXT,
                FOREIGN KEY (uid)
                    REFERENCES objaverse(object_uid)
            );
        """)
        # Create a view to merge these two tables
        await conn.execute("""
            CREATE VIEW IF NOT EXISTS combined AS
            SELECT
                obj.object_uid,
                obj.top_aggregate_caption,
                obj.probability,
                meta.name,
                meta.staffpickedAt,
                meta.viewCount,
                meta.likeCount,
                meta.animationCount,
                meta.description,
                meta.faceCount,
                meta.vertexCount,
                meta.license,
                meta.publishedAt,
                meta.createdAt,
                meta.isAgeRestricted,
                meta.userId,
                meta.userName
            FROM
                objaverse obj
            JOIN
                metadata meta
            ON
                obj.object_uid = meta.uid;
        """)

        # Grab all the uids and table_ from `objaverse` table
        objaverse_uids = await conn.execute_fetchall("""
            SELECT object_uid FROM objaverse;
        """)
        objaverse_uids = [i[0] for i in objaverse_uids]
        objaverse_table_info = await conn.execute_fetchall("""
            PRAGMA table_info(objaverse);
        """)

        # Grab all uids and table_info from `metadata` table
        metadata_uids = await conn.execute_fetchall("""
            SELECT uid FROM metadata;
        """)
        metadata_uids = [i[0] for i in metadata_uids]
        metadata_table_info = await conn.execute_fetchall("""
            PRAGMA table_info(metadata);
        """)

        # Commit all changes
        await conn.commit()

    # A set of uids to compare with database and check for existance
    captions_file_uids = set([i["object_uid"] for i in objaverse_items])

    # Check if captions file and objaverse table have the same values or else populate
    if not captions_file_uids.issubset(set(objaverse_uids)):
        type_conversion_map = {"TEXT": str, "REAL": float, "INTEGER": int}

        # A key/value mapping of header information of the captions file to python types
        objaverse_extraction_map = {
            i[1]: type_conversion_map[i[2]] for i in objaverse_table_info
        }
        # Convert all items from csv file to their respective dtypes for insertion
        objaverse_insert_items = [
            [objaverse_extraction_map[key](val) for key, val in i.items()]
            for i in objaverse_items
        ]
        async with aiosqlite.connect(database_path) as conn:
            await conn.executemany(
                f"""
                INSERT OR IGNORE INTO
                    objaverse
                VALUES
                    ({','.join(repeat("?", len(objaverse_table_info)))})
            """,
                objaverse_insert_items,
            )

            await conn.commit()

    # Check if captions file and metadata table have the same values or else populate
    if not captions_file_uids.issubset(set(metadata_uids)):
        annotations = _subset_annotations(
            await run_in_threadpool(objaverse.load_annotations, captions_file_uids)
        )

        async with aiosqlite.connect(database_path) as conn:
            cols = [i[1] for i in metadata_table_info]

            annotation_items = [[i[j] for j in cols] for i in annotations]
            await conn.executemany(
                f"""
                INSERT OR IGNORE INTO
                    metadata
                VALUES
                    ({",".join(repeat("?", len(metadata_table_info)))})
            """,
                annotation_items,
            )

            await conn.commit()

    return database_path


async def query_db_match(
    database_path: str,
    match_list: List[str],
    table_name: str = "objaverse",
    col_name: str = "top_aggregate_caption",
) -> pd.DataFrame:
    """
    Query a sqlite database to find exact matches on a column from
    the `match_list`

    Parameters
    ==========
    database_path: str
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
    async with aiosqlite.connect(database_path) as con:
        query_str = f"""
            SELECT *
            FROM {table_name}
            WHERE {col_name}
            IN ({",".join(["?" for _ in match_list])});
        """
        async with con.execute(query_str, match_list) as cur:
            rows = await cur.fetchall()
            columns = [i[0] for i in cur.description]

    return pd.DataFrame(list(rows), columns=columns)


def _subset_annotations(annotations: dict) -> List[dict]:
    """
    Grabs a subset of metadata from objaverse.load_annotations

    Parameters
    ==========
    annotations: dict
        The annotations returned from objaverse.load_annotations

    Returns
    =======
    List[dict]
        A list containing the subset of items
    """
    extract_items = [
        "uid",
        "name",
        "staffpickedAt",
        "viewCount",
        "likeCount",
        "animationCount",
        "description",
        "faceCount",
        "vertexCount",
        "license",
        "publishedAt",
        "createdAt",
        "isAgeRestricted",
    ]
    metadata = []
    for annot in annotations.values():
        items = {item: annot[item] for item in extract_items}

        items["publishedAt"] = ciso8601.parse_datetime(items["publishedAt"])
        items["createdAt"] = ciso8601.parse_datetime(items["createdAt"])
        items["userId"] = annot["user"]["uid"]
        items["userName"] = annot["user"]["username"]
        metadata.append(items)

    return metadata
