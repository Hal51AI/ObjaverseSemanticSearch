from typing import Annotated, List

from fastapi import APIRouter, HTTPException, Path, Request

from ..db import query_db_match
from ..models import ObjaverseMetadataResult

router = APIRouter()


@router.get(
    "/id/{uid}",
    response_model=List[ObjaverseMetadataResult],
    responses={
        404: {
            "description": "The uid provided doesnt exist",
            "content": {
                "application/json": {
                    "example": {"detail": "Could not find user with uid: nonce"}
                }
            },
        }
    },
)
async def by_user_id(
    request: Request, uid: Annotated[str, Path(description="The User ID")]
):
    result_df = await query_db_match(
        request.app.state.model.database_path,
        match_list=[uid],
        table_name="combined",
        col_name="userId",
    )
    result_dict = result_df.drop("rowid", axis=1).to_dict(orient="records")

    if not result_dict:
        raise HTTPException(404, f"Could not find user with uid: {uid}")

    return [ObjaverseMetadataResult(**i) for i in result_dict]  # type: ignore


@router.get(
    "/name/{name}",
    response_model=List[ObjaverseMetadataResult],
    responses={
        404: {
            "description": "The name provided doesnt exist",
            "content": {
                "application/json": {
                    "example": {"detail": "Could not find user with username: nonce"}
                }
            },
        }
    },
)
async def by_user_name(
    request: Request, name: Annotated[str, Path(description="The User Name")]
):
    result_df = await query_db_match(
        request.app.state.model.database_path,
        match_list=[name],
        table_name="combined",
        col_name="userName",
    )
    result_dict = result_df.drop("rowid", axis=1).to_dict(orient="records")

    if not result_dict:
        raise HTTPException(404, f"Could not find user with username: {name}")

    return [ObjaverseMetadataResult(**i) for i in result_dict]  # type: ignore
