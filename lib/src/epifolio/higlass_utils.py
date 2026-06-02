"""HiGlass view-config helpers for the index explorer notebook."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from . import ASSETS

# The bundled HiGlass index view config (resgen-hosted TCGA bulk multivec tracks).
INDEX_VIEWCONF_PATH = ASSETS / "conf" / "higlass_index_viewconf.json"

# Plugin track types in the index view config need their JS plugin bundle
# declared inline so higlass-python can load it client-side.
MULTIVEC_PLUGIN_URL = (
    "https://unpkg.com/higlass-multivec@0.3.3/dist/higlass-multivec.js"
)
MULTIVEC_PLUGIN_TYPES = frozenset(
    {"horizontal-multivec", "horizontal-stacked-bar"}
)


def load_index_viewconf(path: str | Path | None = None) -> dict:
    """Read the HiGlass index view config JSON.

    Defaults to the copy bundled in ``epifolio/assets/conf``.
    """
    source = INDEX_VIEWCONF_PATH if path is None else Path(path)
    return json.loads(source.read_text(encoding="utf-8"))


def inject_plugin_urls(
    viewconf: dict,
    *,
    plugin_url: str = MULTIVEC_PLUGIN_URL,
    plugin_types: frozenset[str] = MULTIVEC_PLUGIN_TYPES,
) -> dict:
    """Return a copy of ``viewconf`` with ``plugin_url`` set on plugin tracks.

    Walks every track (including nested ``contents``) across all views and
    positions, adding ``plugin_url`` to tracks whose ``type`` is a plugin type
    and that do not already declare one.
    """
    normalized = copy.deepcopy(viewconf)

    def _walk(tracks: list) -> None:
        for track in tracks:
            track_type = track.get("type")
            if track_type in plugin_types and "plugin_url" not in track:
                track["plugin_url"] = plugin_url
            contents = track.get("contents")
            if isinstance(contents, list):
                _walk(contents)

    for view in normalized.get("views", []):
        for tracks in view.get("tracks", {}).values():
            if isinstance(tracks, list):
                _walk(tracks)

    return normalized
