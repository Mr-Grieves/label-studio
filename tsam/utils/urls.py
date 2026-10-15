"""Turning Label Studio task media URLs back into the S3 keys they point at."""

import base64
import logging
from enum import StrEnum
from pathlib import PurePosixPath
from urllib.parse import parse_qs, urlparse

from label_studio_sdk import LseTask

logger = logging.getLogger(__name__)


class MediaType(StrEnum):
    """Enum for the types of media that can included in Label Studio task data."""

    IMAGE = "image"
    VIDEO = "video"
    VIDEO_PRESCAN = "video_prescan"
    MULTIFRAME = "multiframe"
    PREDICTION_IMAGE = "prediction_image"
    PREDICTION_VIDEO = "prediction_video"

    @property
    def task_key(self) -> str:
        """The key used in `task.data` for this media type."""
        return "images" if self is MediaType.MULTIFRAME else self.value

    @property
    def is_cine(self) -> bool:
        """True if this media type is a cine (i.e. has multiple frames), False otherwise."""
        return self in {MediaType.VIDEO, MediaType.VIDEO_PRESCAN, MediaType.MULTIFRAME, MediaType.PREDICTION_VIDEO}

    @classmethod
    def cine_types(cls) -> list["MediaType"]:
        """Return a list of MediaTypes that are cine (i.e. have multiple frames)."""
        return [mt for mt in cls if mt.is_cine]

    @classmethod
    def from_task_key(cls, task_key: str) -> "MediaType | None":
        """Return the MediaType stored under a `task.data` key, or None if it holds no media.

        >>> MediaType.from_task_key("images")
        <MediaType.MULTIFRAME: 'multiframe'>
        """
        return next((mt for mt in cls if mt.task_key == task_key), None)


def extract_s3_key_from_url(url: str) -> str:
    """Extract s3 key from a Label Studio task media URL.

    >>> extract_s3_key_from_url("s3://bucket/s3_prefix/path/to/image.jpg")
    's3_prefix/path/to/image.jpg'
    >>> extract_s3_key_from_url("https://s3.region.amazonaws.com/bucket/s3_prefix/path/to/image.jpg")
    's3_prefix/path/to/image.jpg'
    >>> extract_s3_key_from_url("https://bucket.s3.region.amazonaws.com/s3_prefix/path/to/image.jpg")
    's3_prefix/path/to/image.jpg'
    >>> extract_s3_key_from_url("/tasks/123/resolve/?fileuri=czM6Ly9idWNrZXQvczNfcHJlZml4L3BhdGgvdG8vaW1hZ2UuanBn")
    's3_prefix/path/to/image.jpg'
    >>> extract_s3_key_from_url("s3_prefix/path/to/image.jpg")
    Traceback (most recent call last):
    ValueError: Unexpected URL format: s3_prefix/path/to/image.jpg
    """
    parsed = urlparse(url)

    if url.startswith("s3://"):
        # Already an s3 URL, extract the key directly
        s3_key = parsed.path.lstrip("/")

    elif url.startswith(("http://", "https://")):
        # Path-style ('s3.<region>.amazonaws.com/<bucket>/<key>') spends a path segment on the bucket;
        # virtual-hosted style ('<bucket>.s3.<region>.amazonaws.com/<key>') names it in the host instead,
        # so dropping two segments there would eat the first segment of the key
        drop = 2 if parsed.netloc.startswith(("s3.", "s3-")) else 1  # the leading "/", plus any bucket segment
        p = PurePosixPath(parsed.path)
        s3_key = PurePosixPath(*p.parts[drop:]).as_posix()

    elif url.startswith("/tasks/") and parsed.path.endswith("/resolve/") and "fileuri" in parse_qs(parsed.query):
        # Parse the URL to get query parameters
        query_params = parse_qs(parsed.query)

        # Get the fileuri parameter, decode it, parse it, and remove the 's3://bucket/' part
        encoded_uri = query_params["fileuri"][0]
        decoded = base64.b64decode(encoded_uri).decode("utf-8")  # => s3://label.clari.us/s3_prefix/.../filename.ext
        parsed = urlparse(decoded)
        s3_key = parsed.path.lstrip("/")
    else:
        msg = f"Unexpected URL format: {url}"
        raise ValueError(msg)

    return s3_key


def is_s3_media_url(value: object) -> bool:
    """Whether `value` is a media URL `extract_s3_key_from_url` can turn into an S3 key.

    Deliberately stricter than that function, which will strip two path segments off *any* http
    URL: this decides whether an unrecognised `task.data` field holds media at all, so a metadata
    field pointing at, say, a cloud exam page must not be mistaken for a file.

    >>> is_s3_media_url("s3://bucket/prefix/image.jpg")
    True
    >>> is_s3_media_url("https://cloud.clarius.com/api/clariusai/exam-data/?exam_id=1")
    False
    >>> is_s3_media_url(12345)
    False
    """
    if not isinstance(value, str):
        return False
    if value.startswith("s3://"):
        return True

    parsed = urlparse(value)
    if value.startswith(("http://", "https://")):
        return parsed.netloc.startswith("s3.") or "amazonaws" in parsed.netloc
    return value.startswith("/tasks/") and parsed.path.endswith("/resolve/") and "fileuri" in parse_qs(parsed.query)


def extract_s3_keys_from_task(task: LseTask) -> dict[str, str | list[str]]:
    """Extract the S3 keys a task's media URLs point at, keyed by the `task.data` key holding them.

    Every field is considered, not only the known `MediaType` ones: a project's labelling config can
    name a media key whatever it likes -- a second video to compare against, say -- and a file is no
    less real for sitting under a key this repo has no name for. Known media keys stay strict, so an
    unreadable URL raises rather than being skipped; an unrecognised key is taken only when its value
    actually looks like a media URL.

    Allows multiple media types - it's up to the call site to decide how to handle them.
    """
    data = task.data
    if not data:
        msg = f"No data found for task {task.id}"
        raise ValueError(msg)

    s3_keys: dict[str, str | list[str]] = {}
    for key, value in data.items():
        if not value:
            continue
        urls = value if isinstance(value, list) else [value]
        if MediaType.from_task_key(key) is None and not all(is_s3_media_url(url) for url in urls):
            continue
        keys = [extract_s3_key_from_url(url) for url in urls]
        s3_keys[key] = keys if isinstance(value, list) else keys[0]

    if not s3_keys:
        msg = f"No media s3 keys found for task {task.id}"
        raise ValueError(msg)

    return s3_keys
