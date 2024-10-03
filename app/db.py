import gzip
import json
import os
import urllib
from itertools import repeat
from typing import Any, Dict, List, Optional

import aiocsv
import aiofiles
import aiosqlite
import ciso8601
import objaverse
import pandas as pd
from starlette.concurrency import run_in_threadpool
from tqdm import tqdm


async def create_db(captions_file: str, database_path: str) -> str:
    """
    Create database from the `caption_file` and annotations from
    objaverse. Creates indexes in locations where searches are going to be
    as well as creates a view combining the `captions_file` and annotations.

    Parameters
    ==========
    captions_file: str
        File containing caption data and probability of classification
    database_path: str
        Location where database file is located. If database already exists
        checks and validates the contents.

    Returns
    =======
    str
        Path to the newly created database
    """
    # Read from captions file asynchronously
    async with aiofiles.open(captions_file) as fp:
        objaverse_items = [i async for i in aiocsv.AsyncDictReader(fp, delimiter=";")]

    async with aiosqlite.connect(database_path) as conn:
        await conn.execute("""
            PRAGMA synchronous = NORMAL;
        """)
        # Create objaverse table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS objaverse (
                object_uid            TEXT,
                top_aggregate_caption TEXT,
                probability           REAL
            );
        """)
        # Create index on for fast retrieval
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_caption ON objaverse (
                object_uid, top_aggregate_caption
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
                userProfile     TEXT AS ('https://sketchfab.com/' || userName),
                FOREIGN KEY (uid)
                    REFERENCES objaverse(object_uid)
            );
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_description ON metadata (
                description
            );
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_id ON metadata (
                userID
            );
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_name ON metadata (
                userName
            );
        """)
        # Create paths table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS paths (
                uid  TEXT PRIMARY KEY,
                path TEXT
            );
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_paths ON paths (
                uid
            );
        """)
        # Create a view to merge these two tables
        await conn.execute("""
            CREATE VIEW IF NOT EXISTS combined AS
            SELECT
                obj.rowid,
                obj.object_uid,
                obj.top_aggregate_caption,
                obj.probability,
                'https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/' || p.path || '/' || p.uid || '.glb' AS download_url,
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
                meta.userName,
                meta.userProfile
            FROM
                objaverse obj
            JOIN
                metadata meta
                ON obj.object_uid = meta.uid
            JOIN
                paths p
                ON obj.object_uid = p.uid;
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

        # Grab all uids and table_info from `paths` table
        paths_uids = await conn.execute_fetchall("""
            SELECT uid FROM paths;
        """)
        paths_uids = [i[0] for i in paths_uids]
        paths_table_info = await conn.execute_fetchall("""
            PRAGMA table_info(paths);
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
                    ({','.join(repeat("?", len(list(objaverse_table_info))))})
            """,
                objaverse_insert_items,
            )

            await conn.commit()

    # Check if captions file and metadata table have the same values or else populate
    if not captions_file_uids.issubset(set(metadata_uids)):
        annotations = await run_in_threadpool(
            _load_annotations, list(captions_file_uids)
        )

        async with aiosqlite.connect(database_path) as conn:
            cols = [i[1] for i in metadata_table_info]

            annotation_items = [[i[j] for j in cols] for i in annotations]
            await conn.executemany(
                f"""
                INSERT OR IGNORE INTO
                    metadata
                VALUES
                    ({",".join(repeat("?", len(list(metadata_table_info))))})
            """,
                annotation_items,
            )

            await conn.commit()

    if not captions_file_uids.issubset(set(paths_uids)):
        paths_insert_items = _load_objaverse_paths(list(captions_file_uids))
        async with aiosqlite.connect(database_path) as conn:
            await conn.executemany(
                f"""
                INSERT OR IGNORE INTO
                    paths
                VALUES
                    ({','.join(repeat("?", len(list(paths_table_info))))})
            """,
                paths_insert_items.items(),
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


def _load_objaverse_paths(uids: List[str]) -> Dict[str, str]:
    """
    This function loads a subset of object paths from the objaverse
    based on a list of unique identifiers (uids)

    Parameters
    ==========
    uids: list[str]:
        A list of uids. Each uid corresponds to a specific object
        within the objaverse dataset.

    Returns
    =======
    dict[str, str]
        A dictionary where each key is a UID from the input uids list,
        and each value is the corresponding path component
    """
    paths = objaverse._load_object_paths()
    paths = {k: v.split("/")[1] for k, v in paths.items()}
    return {key: paths[key] for key in uids if key in paths}


def _load_annotations(uids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Load the full metadata of all objects in the dataset.

    Code was modified from
        https://pypi.org/project/objaverse

    Args:
        uids: A list of uids with which to load metadata. If None, it loads
        the metadata for all uids.

    Returns:
        A dictionary mapping the uid to the metadata.
    """
    BASE_PATH = os.path.join(os.path.expanduser("~"), ".objaverse")
    _VERSIONED_PATH = os.path.join(BASE_PATH, "hf-objaverse-v1")

    metadata_path = os.path.join(_VERSIONED_PATH, "metadata")
    object_paths = objaverse._load_object_paths()
    dir_ids = tqdm(
        set(object_paths[uid].split("/")[1] for uid in uids)
        if uids is not None
        else [f"{i // 1000:03d}-{i % 1000:03d}" for i in range(160)]
    )

    out = []
    for i_id in dir_ids:
        json_file = f"{i_id}.json.gz"
        local_path = os.path.join(metadata_path, json_file)
        if not os.path.exists(local_path):
            hf_url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/metadata/{i_id}.json.gz"
            # wget the file and put it in local_path
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            urllib.request.urlretrieve(hf_url, local_path)
        with gzip.open(local_path, "rb") as f:
            data = json.load(f)
        if uids is not None:
            data = {uid: data[uid] for uid in uids if uid in data}
        out.extend(_subset_annotations(data))
        if uids is not None and len(out) == len(uids):
            break
    return out
