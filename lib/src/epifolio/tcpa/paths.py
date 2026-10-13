"""Resolve the TCPA data source: explicit local data, else the hosted release."""
import os
from pathlib import Path
import tomllib
import urllib.request

# Dataset-specific artifact paths belong to the catalog and are checked by loaders.
REQUIRED_DATA = ('manifest.parquet',)
# Hosted release: the same layout as a local data directory, read over HTTP.
REMOTE = 'https://users.abdenlab.org/voy/itcr/epifolio/notebooks/tcpa'
# A checkout of the source project keeps its own local data and never reads REMOTE.
SOURCE_PROJECT = 'dev-tcpa'
TIMEOUT = 30


class RemotePath(os.PathLike):
    """The subset of Path that the loaders use, over an HTTP data root.

    HTTP errors propagate: a missing remote object is a failure, never a
    reason to try another source.
    """

    def __init__(self, url):
        self.url = str(url).rstrip('/')

    def __str__(self):
        return self.url

    def __repr__(self):
        return f'RemotePath({self.url!r})'

    def __fspath__(self):
        return self.url

    def __truediv__(self, name):
        return RemotePath(f'{self.url}/{name}')

    @property
    def name(self):
        return self.url.rsplit('/', 1)[-1]

    def resolve(self):
        return self

    def is_relative_to(self, root):
        return self.url.startswith(str(root).rstrip('/') + '/')

    def exists(self):
        request = urllib.request.Request(self.url, method='HEAD')
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status == 200

    def read_bytes(self):
        with urllib.request.urlopen(self.url, timeout=TIMEOUT) as response:
            return response.read()

    def read_text(self):
        return self.read_bytes().decode()


def _validate(directory: Path, required: tuple[str, ...], source: str) -> Path:
    if not directory.is_dir():
        raise FileNotFoundError(f'{source}: data directory does not exist: {directory}')
    for name in required:
        if Path(name).name != name or name in {'', '.', '..'}:
            raise ValueError(f'Expected a direct data artifact name, got {name!r}')
        artifact = directory / name
        valid = artifact.is_file()
        if name.endswith('.zarr'):
            valid = artifact.is_dir() and any(
                (artifact / marker).is_file() for marker in ('.zgroup', 'zarr.json')
            )
        if not valid:
            raise FileNotFoundError(f'{source}: missing or invalid data artifact: {artifact}')
    return directory


def _project(start: Path) -> tuple[Path, str | None] | None:
    """Return the nearest ancestor project directory and its declared name."""
    for directory in (start, *start.parents):
        config = directory / 'pyproject.toml'
        if config.is_file():
            with config.open('rb') as stream:
                return directory, tomllib.load(stream).get('project', {}).get('name')
    return None


def resolve_data_dir(*, anchor: str | Path | None = None,
                     required: tuple[str, ...] = REQUIRED_DATA) -> Path | RemotePath:
    """Resolve override > caller anchor > cwd's source project > hosted release.

    An anchor is an existing notebook file or its directory. An explicit local
    choice never falls back: TCPA_DATA_DIR, data adjacent to the anchor, and a
    source-project checkout must each hold valid local data or fail. The hosted
    release is used only where no local data is claimed, and unrelated cwd or
    parent data directories are never searched. Artifact validation checks
    file/store presence, not scientific correctness; remote objects are checked
    when read, so resolving performs no network access.
    """
    if 'TCPA_DATA_DIR' in os.environ:
        override = os.environ['TCPA_DATA_DIR']
        if not override.strip():
            raise ValueError('TCPA_DATA_DIR must name a nonempty data directory')
        return _validate(Path(override).expanduser().resolve(), required, 'TCPA_DATA_DIR')
    if anchor is not None:
        location = Path(anchor).expanduser().resolve()
        if not location.exists():
            raise FileNotFoundError(f'Caller anchor does not exist: {location}')
        directory = location if location.is_dir() else location.parent
        project = _project(directory)
        if (directory / 'data').exists() or (project and project[1] == SOURCE_PROJECT):
            return _validate(directory / 'data', required, 'Caller anchor')
        return RemotePath(REMOTE)
    project = _project(Path.cwd().resolve())
    if project and project[1] == SOURCE_PROJECT:
        return _validate(project[0] / 'data', required, 'Working-directory project')
    return RemotePath(REMOTE)
