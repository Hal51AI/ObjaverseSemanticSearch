from typing import Annotated, Dict, List

from aiocache import Cache, cached
from aiocache.serializers import JsonSerializer
from fastapi import Query, Request


@cached(cache=Cache.MEMORY, serializer=JsonSerializer())
async def similarity_search_query(
    request: Request,
    query: Annotated[
        str, Query(description="Perform similarity search on the query string")
    ],
    top_k: Annotated[
        int, Query(description="Grab top k results based on similarity")
    ] = 10,
) -> List[Dict]:
    return await request.app.state.model.search(query, top_k=top_k)


async def all_licenses() -> Dict[str, Dict[str, str]]:
    return {
        "by": {
            "label": "CC Attribution",
            "fullName": "Creative Commons Attribution",
            "requirements": "Author must be credited. Commercial use is allowed.",
            "url": "http://creativecommons.org/licenses/by/4.0/",
            "slug": "by",
        },
        "by-sa": {
            "label": "CC Attribution-ShareAlike",
            "fullName": "Creative Commons Attribution-ShareAlike",
            "requirements": "Author must be credited. Modified versions must have the same license. Commercial use is allowed.",
            "url": "http://creativecommons.org/licenses/by-sa/4.0/",
            "slug": "by-sa",
        },
        "by-nd": {
            "label": "CC Attribution-NoDerivs",
            "fullName": "Creative Commons Attribution-NoDerivs",
            "requirements": "Author must be credited. Modified versions can not be distributed. Commercial use is allowed.",
            "url": "http://creativecommons.org/licenses/by-nd/4.0/",
            "slug": "by-nd",
        },
        "by-nc": {
            "label": "CC Attribution-NonCommercial",
            "fullName": "Creative Commons Attribution-NonCommercial",
            "requirements": "Author must be credited. No commercial use.",
            "url": "http://creativecommons.org/licenses/by-nc/4.0/",
            "slug": "by-nc",
        },
        "by-nc-sa": {
            "label": "CC Attribution-NonCommercial-ShareAlike",
            "fullName": "CC Attribution-NonCommercial-ShareAlike",
            "requirements": "Author must be credited. No commercial use. Modified versions must have the same license.",
            "url": "http://creativecommons.org/licenses/by-nc-sa/4.0/",
            "slug": "by-nc-sa",
        },
        "by-nc-nd": {
            "label": "CC Attribution-NonCommercial-NoDerivs",
            "fullName": "CC Attribution-NonCommercial-NoDerivs",
            "requirements": "Author must be credited. No commercial use. Modified versions can not be distributed.",
            "url": "http://creativecommons.org/licenses/by-nc-nd/4.0/",
            "slug": "by-nc-nd",
        },
        "cc0": {
            "label": "CC0 Public Domain",
            "fullName": "CC0 Public Domain",
            "requirements": "Credit is not mandatory. Commercial use is allowed.",
            "url": "http://creativecommons.org/publicdomain/zero/1.0/",
            "slug": "cc0",
        },
        "free-st": {
            "label": "Free Standard",
            "fullName": "Free Standard",
            "requirements": "Under basic restrictions, use worldwide, on all types of media, commercially or not, and in all types of derivative works",
            "url": "https://sketchfab.com/licenses",
            "slug": "free-st",
        },
        "st": {
            "label": "Standard",
            "fullName": "Standard",
            "requirements": "Under basic restrictions, use worldwide, on all types of media, commercially or not, and in all types of derivative works",
            "url": "https://sketchfab.com/licenses",
            "slug": "st",
        },
    }
