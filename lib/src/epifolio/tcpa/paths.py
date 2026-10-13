"""Resolve explicit local data without downloading or guessing dataset identity."""
import os
from pathlib import Path
import tomllib

# Dataset-specific artifact paths belong to the catalog and are checked by loaders.
REQUIRED_DATA = ('manifest.parquet',)


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


def resolve_data_dir(*, anchor: str | Path | None = None,
                     required: tuple[str, ...] = REQUIRED_DATA) -> Path:
    """Resolve override > explicit caller anchor > cwd's nearest project root.

    An anchor is an existing notebook file or its directory. An explicit choice
    never falls back. Without an anchor, the nearest ancestor pyproject.toml must
    identify dev-tcpa; this prevents selecting an unrelated cwd/data directory.
    Artifact validation checks file/store presence, not scientific correctness.
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
        return _validate(directory / 'data', required, 'Caller anchor')
    cwd = Path.cwd().resolve()
    for directory in (cwd, *cwd.parents):
        config = directory / 'pyproject.toml'
        if config.is_file():
            with config.open('rb') as stream:
                project = tomllib.load(stream).get('project', {})
            if project.get('name') != 'dev-tcpa':
                raise FileNotFoundError(f'Working directory belongs to another project: {config}')
            return _validate(directory / 'data', required, 'Working-directory project')
    raise FileNotFoundError('No TCPA project found; supply a caller anchor or TCPA_DATA_DIR')
