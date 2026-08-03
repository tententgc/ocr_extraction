"""Fetch a document from a public URL.

The platform is meant to accept arbitrary documents found on the internet, which
makes it a server-side request forgery target: a crafted URL could otherwise make
the server read its own metadata endpoint or an internal service. Every hop of
every redirect is therefore resolved and checked against private address ranges
before a connection is made.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
from dataclasses import dataclass
from urllib.parse import unquote, urlparse

import httpx

logger = logging.getLogger(__name__)

ALLOWED_SCHEMES = frozenset({"http", "https"})
MAX_REDIRECTS = 5
CONNECT_TIMEOUT_SECONDS = 15.0
READ_TIMEOUT_SECONDS = 60.0
USER_AGENT = "DocuFlow/1.0 (document ingestion)"

CONTENT_TYPE_SUFFIX = {
    "application/pdf": ".pdf",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/tiff": ".tif",
    "image/bmp": ".bmp",
    "image/gif": ".gif",
}
_FILENAME_RE = re.compile(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', re.IGNORECASE)
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


class FetchError(RuntimeError):
    """The URL could not be fetched, or is not safe to fetch."""


@dataclass(frozen=True)
class FetchedDocument:
    content: bytes
    filename: str
    content_type: str | None
    final_url: str


def _assert_public_host(url: str) -> None:
    """Refuse loopback, private, link-local and other non-public destinations."""
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise FetchError(f"Only http and https URLs are supported (got '{parsed.scheme or 'none'}').")
    host = parsed.hostname
    if not host:
        raise FetchError("URL has no host.")

    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise FetchError(f"Could not resolve host '{host}'.") from exc

    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global or address.is_multicast:
            raise FetchError(
                f"Refusing to fetch '{host}' - it resolves to the non-public address {address}."
            )


def _filename_from(response: httpx.Response, url: str) -> str:
    """Best available filename: Content-Disposition, URL path, then a default."""
    disposition = response.headers.get("content-disposition", "")
    match = _FILENAME_RE.search(disposition)
    if match:
        candidate = unquote(match.group(1)).strip()
        if candidate:
            return _sanitize(candidate)

    path = urlparse(url).path
    if path and "/" in path:
        candidate = unquote(path.rsplit("/", 1)[-1])
        if candidate and "." in candidate:
            return _sanitize(candidate)

    content_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
    suffix = CONTENT_TYPE_SUFFIX.get(content_type, "")
    return f"downloaded-document{suffix}"


def _sanitize(name: str) -> str:
    """Strip path separators and unsafe characters from a remote filename."""
    cleaned = _SAFE_NAME_RE.sub("_", name.replace("\\", "/").rsplit("/", 1)[-1]).strip("._")
    return cleaned[:120] or "downloaded-document"


def fetch_url(url: str, max_bytes: int) -> FetchedDocument:
    """Download a document, enforcing the size limit while streaming."""
    current = url.strip()
    if not current:
        raise FetchError("No URL was supplied.")
    if "://" not in current:
        current = f"https://{current}"

    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    timeout = httpx.Timeout(READ_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS)

    with httpx.Client(follow_redirects=False, timeout=timeout, headers=headers) as client:
        for _ in range(MAX_REDIRECTS + 1):
            _assert_public_host(current)
            try:
                with client.stream("GET", current) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise FetchError("Server sent a redirect without a destination.")
                        current = str(response.url.join(location))
                        continue

                    if response.status_code >= 400:
                        raise FetchError(f"Server returned HTTP {response.status_code} for {current}.")

                    declared = response.headers.get("content-length")
                    if declared and int(declared) > max_bytes:
                        raise FetchError(
                            f"Document is {int(declared) / 1e6:.1f} MB, above the "
                            f"{max_bytes / 1e6:.0f} MB limit."
                        )

                    chunks: list[bytes] = []
                    total = 0
                    for chunk in response.iter_bytes():
                        total += len(chunk)
                        if total > max_bytes:
                            raise FetchError(
                                f"Document exceeds the {max_bytes / 1e6:.0f} MB limit."
                            )
                        chunks.append(chunk)

                    content = b"".join(chunks)
                    if not content:
                        raise FetchError("The URL returned an empty document.")

                    return FetchedDocument(
                        content=content,
                        filename=_filename_from(response, current),
                        content_type=(response.headers.get("content-type") or "").split(";")[0].strip() or None,
                        final_url=current,
                    )
            except httpx.HTTPError as exc:
                raise FetchError(f"Could not fetch {current}: {exc}") from exc

    raise FetchError(f"Too many redirects (more than {MAX_REDIRECTS}).")
