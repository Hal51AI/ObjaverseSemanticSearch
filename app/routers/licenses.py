from typing import Annotated, Dict

from fastapi import APIRouter, Depends, HTTPException, Path
from fastapi.responses import JSONResponse

from ..dependencies import all_licenses
from ..models import LicenseInfo

router = APIRouter()


@router.get(
    "",
    response_class=JSONResponse,
    response_model=Dict[str, LicenseInfo],
)
async def licenses(
    license: Dict[str, LicenseInfo] = Depends(all_licenses),
) -> Dict[str, LicenseInfo]:
    """
    Get all types of licenses that are available
    """
    return license


@router.get(
    "/{slug}",
    response_class=JSONResponse,
    response_model=LicenseInfo,
)
async def license_slugs(
    slug: Annotated[str, Path(description="The slug to search for")],
    license: Dict[str, Dict[str, str]] = Depends(all_licenses),
) -> LicenseInfo:
    """
    Get information on a particular license based on the slug
    """
    if slug not in license:
        raise HTTPException(
            status_code=404,
            detail=f"The slug {slug} does not exist in: {list(license)}",
        )
    return LicenseInfo(**license[slug])
