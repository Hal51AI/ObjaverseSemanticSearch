import os
import random
from typing import Dict, List

import objaverse
from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from scipy.special import softmax
from starlette.concurrency import run_in_threadpool

from ..dependencies import similarity_search_query
from ..models import ObjaverseItemResult
from ..utils import reformat_results

router = APIRouter()


@router.get(
    "",
    response_model=List[ObjaverseItemResult],
    responses={
        200: {
            "description": "A similarity query",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "object_uid": "2f0598cba938424688cdd048a90b8339",
                            "top_aggregate_caption": "a boat",
                            "probability": 0.439652563952344,
                            "similarity": 0.7142868041992188,
                            "metadata": {
                                "name": "boat",
                                "staffpickedAt": None,
                                "viewCount": 1775,
                                "likeCount": 28,
                                "animationCount": 0,
                                "description": "",
                                "faceCount": 18458,
                                "vertexCount": 10306,
                                "license": "by",
                                "publishedAt": "2021-04-29 23:22:23.021191",
                                "createdAt": "2021-04-29 22:45:13.799783",
                                "isAgeRestricted": False,
                                "userId": "49dd921047ae4da28aeaaa213e9a5d8a",
                                "userName": "jondameron5",
                            },
                        },
                        {
                            "object_uid": "56148c53e9664ee683b598fadb457992",
                            "top_aggregate_caption": "a black sailboat",
                            "probability": 0.3841391686492714,
                            "similarity": 0.6926552653312683,
                            "metadata": {
                                "name": "Schooner",
                                "staffpickedAt": None,
                                "viewCount": 844,
                                "likeCount": 26,
                                "animationCount": 0,
                                "description": "",
                                "faceCount": 7995,
                                "vertexCount": 5317,
                                "license": "by",
                                "publishedAt": "2019-06-26 06:51:11.054891",
                                "createdAt": "2019-06-26 06:46:39.725488",
                                "isAgeRestricted": False,
                                "userId": "0c477a42672142baadcefe9cc41845d2",
                                "userName": "Tobias.De.Maine",
                            },
                        },
                    ]
                }
            },
        }
    },
)
async def similarity(search: List[Dict] = Depends(similarity_search_query)):
    """
    Perform similarity search over a query and grab relevant metadata

    Schema
    ======
    | Schema                  | Type   | Description                                          |
    |-------------------------|--------|------------------------------------------------------|
    | `object_uid`            | UUID   | Unique identifier for objaverse assets               |
    | `top_aggregate_caption` | String | Caption generated from classifier                    |
    | `probability`           | Float  | The confidence/probability score from the classifier |
    | `similarity`            | Float  | The similarity score between the query               |
    | `metadata`              | Dict   | Asset information metadata                           |

    Metadata
    ========
    | Schema            | Type             | Description                                                                         |
    |-------------------|------------------|-------------------------------------------------------------------------------------|
    | `name`            | String           | The original name of the asset made by the author                                   |
    | `staffpickedAt`   | Datetime or null | High quality assets which have been selected by staff                               |
    | `viewCount`       | Integer          | The amount of times asset was viewed by another user at the time of data collection |
    | `likeCount`       | Integer          | The amount of times asset was liked by another user at the time of data collection  |
    | `animationCount`  | Integer          | The number of animations for the asset                                              |
    | `description`     | String           | A longer description made by the author                                             |
    | `faceCount`       | Integer          | The number of faces of the asset                                                    |
    | `vertexCount`     | Integer          | The number of verticies in the asset                                                |
    | `license`         | String           | The type of license used by the asset                                               |
    | `publishedAt`     | Datetime         | The time when the asset was published to the world                                  |
    | `createdAt`       | Datetime         | The time for which the asset was first created                                      |
    | `isAgeRestricted` | Boolean          | Whether the asset has been flagged to contain age restricted content                |
    | `userId`          | UUID             | The sketchfab user id used to query for more user information                       |
    | `userName`        | String           | The skechfab user name                                                              |
    """
    return list(map(reformat_results, search))


@router.get(
    "/glb",
    response_class=FileResponse,
    responses={200: {"content": {"model/gltf-binary": {"example": "binary blob..."}}}},
)
async def glb(search: List[Dict] = Depends(similarity_search_query)) -> FileResponse:
    """
    Perform similarity search over a query and grab a random glb file
    """
    # Grab a random item from the objects weighted by the similarity score
    weights = softmax([i["similarity"] for i in search])
    selection = random.choices([i["object_uid"] for i in search], weights=weights)

    # Download from objaverse
    glb_map = await run_in_threadpool(objaverse.load_objects, selection)

    # read file from filesystem
    filepath = list(glb_map.values())[0]

    return FileResponse(
        path=filepath,
        media_type="model/gltf-binary",
        filename=os.path.basename(filepath),
    )
