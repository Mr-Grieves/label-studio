"""S3 utils used by Label Studio."""

import json
import logging
from collections.abc import Iterable
from functools import cache, lru_cache
from io import BytesIO
import os
from pathlib import Path
from typing import Any, BinaryIO, cast

import boto3
import numpy as np
from botocore.exceptions import BotoCoreError, ClientError
from PIL import Image

logger = logging.getLogger(__name__)


def get_env_var(var_name, default=None):
    """Get environment variable, or log and raise error if not defined."""
    value = os.getenv(var_name, default)
    if value is None:
        logger.error(f"Environment variable not defined: {var_name}")
        raise KeyError(f"Environment variable not defined: {var_name}")
    return value


class S3Client:
    """Wrapper class for botocore S3 client."""

    def __init__(self) -> None:
        """Initialize S3 client."""
        self.aws_storage_bucket_name = get_env_var("AWS_STORAGE_BUCKET_NAME")

        self.aws_access_key_id = get_env_var("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = get_env_var("AWS_SECRET_ACCESS_KEY")
        self.endpoint_url = get_env_var("AWS_ENDPOINT_URL")
        self.region_name = get_env_var("AWS_REGION_NAME")

        self.client = boto3.client(
            "s3",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            endpoint_url=self.endpoint_url,
            region_name=self.region_name,
        )

    @property
    def object_url_prefix(self) -> str:
        """Get the S3 object URL prefix."""
        return f"{self.endpoint_url}/{self.aws_storage_bucket_name}".rstrip("/") + "/"

    def get_json_blob(self, key: str | None) -> dict[str, Any]:
        """Get json blob from S3 storage.

        Returns an empty dict if the key is falsy or the object does not exist, so callers can
        continue processing instead of failing the entire operation. The missing-object case is
        handled here (not in the cached helper) so that negative results are never memoized -- a
        blob fetched from the cloud and uploaded later must be visible to a subsequent read.
        """
        if not key:
            return {}
        try:
            return _get_json_blob_from_s3(self, key)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code", "") in {"NoSuchKey", "NotFound", "404"}:
                logger.debug("No JSON found in S3 at key '%s'", key)
                return {}
            raise

    def get_image(self, key: str) -> np.ndarray:
        """Get image as numpy array from S3 bucket."""
        return cast("np.ndarray", _get_image_from_s3(self, key))

    def get_image_bytes(self, key: str) -> bytes:
        """Get image as bytes from S3 bucket."""
        return cast("bytes", _get_image_from_s3(self, key, as_bytes=True))

    def object_exists(self, object_key: str) -> bool:
        """Check if an object already exists in S3.

        I generally dislike flow control in exceptions, but this seems to
        be the best way of doing it...
        https://stackoverflow.com/questions/33842944/check-if-a-key-exists-in-a-bucket-in-s3-using-boto3
        """
        try:
            self.client.head_object(Bucket=self.aws_storage_bucket_name, Key=object_key)
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

        return True

    def upload(self, data: str | bytes | dict | Path | BinaryIO, object_key: str) -> None:
        """Upload a file to S3 storage.

        Args:
            data: Data to upload. Can be a string, dict (will be converted to json), bytes,
                Path to a file, or a file-like object.
            object_key: Key of the object in S3

        """
        if isinstance(data, Path):
            self.client.upload_file(Filename=str(data), Bucket=self.aws_storage_bucket_name, Key=object_key)
            return

        # Convert str or dict to bytes
        if isinstance(data, dict):
            data = json.dumps(data, ensure_ascii=False).encode("utf-8")
        elif isinstance(data, str):
            data = data.encode("utf-8")

        if isinstance(data, bytes):
            self.client.put_object(Body=data, Bucket=self.aws_storage_bucket_name, Key=object_key)
            return

        # By elimination (Path/dict/str/bytes handled above), only a file-like object is left
        if not hasattr(data, "read"):
            msg = f"Unsupported data type: {type(data)}"
            raise TypeError(msg)

        self.client.upload_fileobj(Fileobj=data, Bucket=self.aws_storage_bucket_name, Key=object_key)

    def download(self, object_key: str, local_path: Path) -> None:
        """Download a file from S3 storage to a local path.

        Args:
            object_key: Key of the object in S3
            local_path: Local path to save the file

        """
        self.client.download_file(Bucket=self.aws_storage_bucket_name, Key=object_key, Filename=str(local_path))

    def prefix_exists(self, prefix: str) -> bool:
        """Check whether any object exists under the given S3 prefix, with a single bounded call.

        `get_keys` pages through every key under the prefix, which is a lot of round trips for a
        yes/no; `MaxKeys=1` stops at the first.

        Args:
            prefix: Prefix to check, trailing slash included to avoid matching a sibling prefix.
        """
        response = self.client.list_objects_v2(Bucket=self.aws_storage_bucket_name, Prefix=prefix, MaxKeys=1)
        return response.get("KeyCount", 0) > 0

    def get_keys(
        self,
        prefix: str,
        suffixes: Iterable[str] | None = None,
        *,
        log_stats: bool = False,
    ) -> list[str]:
        """Get list of keys in S3 bucket with the given prefix and optional suffixes.

        Args:
            prefix: Prefix to filter keys
            suffixes: Optional list of suffixes to filter keys
            log_stats: Whether to log the total size of the keys found
        Returns:
            List of keys in the bucket matching the prefix and suffixes

        """
        # Make lowercase for case-insensitive matching; tuple is faster for endswith()
        _suffixes = tuple(s.lower() for s in suffixes) if suffixes else None
        keys: list[str] = []
        size = 0

        paginator = self.client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self.aws_storage_bucket_name, Prefix=prefix)

        for page in pages:
            contents = page.get("Contents")
            if not contents:
                return []
            for obj in contents:
                key = obj["Key"]
                if _suffixes is None or key.lower().endswith(_suffixes):
                    keys.append(key)
                    size += obj["Size"]

        if log_stats:
            logger.info(
                "There are %s files totalling %s MB in S3 bucket with prefix '%s'%s",
                f"{len(keys):,}",
                f"{size / 1024**2:.2f}",
                prefix,
                f" (for files suffixed {suffixes})" if suffixes else "",
            )

        return keys


get_s3_client = cache(S3Client)
"""Return the shared S3Client, constructed on first call.

Obtain it at the top of the work that needs it, not at module scope: imports stay free of AWS
env vars, and missing config fails before any processing runs.
"""


def require_s3() -> None:
    """Confirm the shared S3Client can reach the bucket, raising a readable error if not.

    Call at the top of a script's main() so bad config fails before any work, not part-way through.
    Kept out of get_s3_client() since it costs a network round-trip.
    """
    s3 = get_s3_client()
    try:
        # head_bucket needs s3:ListBucket, which these credentials already require for project listings
        s3.client.head_bucket(Bucket=s3.aws_storage_bucket_name)
    except (BotoCoreError, ClientError) as e:
        msg = (
            f"Cannot reach S3 bucket '{s3.aws_storage_bucket_name}' at {s3.endpoint_url}. "
            "Check the AWS_* credentials in the environment."
        )
        raise RuntimeError(msg) from e


# Caching methods is bad, so one solution is defining them at module level.
# See info and links at https://docs.astral.sh/ruff/rules/cached-instance-method/


@lru_cache(maxsize=1024)
def _get_json_blob_from_s3(s3: S3Client, key: str) -> dict[str, Any]:
    """Get json blob string from S3 storage.

    Raises ClientError if the object does not exist; the missing-object case is handled by the
    S3Client.get_json_blob wrapper. Keeping it out of this cached helper ensures negative results
    are not memoized, so a blob uploaded after a miss is visible to the next read.
    """
    response = s3.client.get_object(Bucket=s3.aws_storage_bucket_name, Key=key)
    return json.load(response["Body"])


def _get_image_from_s3(s3: S3Client, key: str, *, as_bytes: bool = False) -> np.ndarray | bytes:
    """Return image as numpy array from S3 bucket and key.

    Uncached, unlike _get_json_blob_from_s3: each key is read once, so a cache never hits and would
    only pin decoded arrays in the long-lived portal process.
    """
    response = s3.client.get_object(Bucket=s3.aws_storage_bucket_name, Key=key)
    image_bytes = response["Body"].read()
    if as_bytes:
        return image_bytes

    image_array = np.array(Image.open(BytesIO(image_bytes)))

    # Add channel dimension if missing
    grayscale_ndim = 2
    if len(image_array.shape) == grayscale_ndim:
        image_array = np.expand_dims(image_array, axis=-1)

    return image_array
