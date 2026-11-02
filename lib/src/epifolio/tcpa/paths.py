"""Resolve the TCPA data source: a local TCPA_DATA_DIR, else the hosted release."""

import os
import urllib.error
import urllib.request
from pathlib import Path

# Dataset-specific artifact paths belong to the catalog and are checked by loaders.
REQUIRED_DATA = ("manifest.parquet",)
# Hosted release: the same layout as a local data directory, read over HTTP.
REMOTE = "https://projects.abdenlab.org/itcr/epifolio/notebooks/tcpa"
TIMEOUT = 30


class _RefuseRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


_NO_REDIRECT = urllib.request.build_opener(_RefuseRedirect)


class RemotePath(os.PathLike):
    """The subset of Path that the loaders use, over an HTTP data root.

    HTTP errors propagate: a missing remote object is a failure, never a
    reason to try another source.
    """

    def __init__(self, url):
        self.url = str(url).rstrip("/")

    def __str__(self):
        return self.url

    def __repr__(self):
        return f"RemotePath({self.url!r})"

    def __fspath__(self):
        return self.url

    def __truediv__(self, name):
        return RemotePath(f"{self.url}/{name}")

    @property
    def name(self):
        return self.url.rsplit("/", 1)[-1]

    def resolve(self):
        return self

    def is_relative_to(self, root):
        return self.url.startswith(str(root).rstrip("/") + "/")

    def exists(self):
        # Hosts may redirect a missing object to an HTML page; only a direct 200 counts.
        request = urllib.request.Request(self.url, method="HEAD")
        try:
            with _NO_REDIRECT.open(request, timeout=TIMEOUT) as response:
                return response.status == 200
        except urllib.error.HTTPError as error:
            if error.code == 404 or 300 <= error.code < 400:
                return False
            raise

    def read_bytes(self):
        with urllib.request.urlopen(self.url, timeout=TIMEOUT) as response:
            return response.read()

    def read_text(self):
        return self.read_bytes().decode()


def _validate(directory: Path, required: tuple[str, ...], source: str) -> Path:
    if not directory.is_dir():
        raise FileNotFoundError(f"{source}: data directory does not exist: {directory}")
    for name in required:
        if Path(name).name != name or name in {"", ".", ".."}:
            raise ValueError(f"Expected a direct data artifact name, got {name!r}")
        artifact = directory / name
        valid = artifact.is_file()
        if name.endswith(".zarr"):
            valid = artifact.is_dir() and any(
                (artifact / marker).is_file() for marker in (".zgroup", "zarr.json")
            )
        if not valid:
            raise FileNotFoundError(f"{source}: missing or invalid data artifact: {artifact}")
    return directory


def resolve_data_dir(*, required: tuple[str, ...] = REQUIRED_DATA) -> Path | RemotePath:
    """TCPA_DATA_DIR if set, else the hosted release.

    A local override never falls back: it must hold the required artifacts
    (presence only, not scientific correctness). Remote objects are checked
    when read, so resolving performs no network access.
    """
    if "TCPA_DATA_DIR" in os.environ:
        override = os.environ["TCPA_DATA_DIR"]
        if not override.strip():
            raise ValueError("TCPA_DATA_DIR must name a nonempty data directory")
        return _validate(Path(override).expanduser().resolve(), required, "TCPA_DATA_DIR")
    return RemotePath(REMOTE)
