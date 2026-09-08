# CNV/cCRE Post-PR Report

**Date:** 2026-09-03
**Branch:** `feat/cnvnb`
**Scope:** `notebooks/cnv_ccre_explorer.py`, `lib/src/epifolio/cnv.py`, launch/runtime behaviour
**Companions:** [cnvintegration.md](cnvintegration.md) (the plan), [cnvhardening.md](cnvhardening.md) (the port audit)
**Status:** investigation only — no code changed by this report

---

## 0. TL;DR

Three classes of problem, all verified by execution. The third one turned out to have a solution already
sitting in the project's own data directory.

| # | Finding | Severity | Blocked on |
|---|---|---|---|
| **L1** | `uvx marimo run` + declined sandbox installs **none** of the inline deps → `ModuleNotFoundError: jscatter` | launch blocker | nothing (doc/flag fix) |
| **L2** | Sandbox resolves `epifolio` from **abdenlab's default branch**, which has no `cnv` module → `ModuleNotFoundError: epifolio.cnv` | launch blocker | push `feat/cnvnb` upstream |
| **L3** | The sandbox prompt has **opposite correct answers** depending on launcher (`uvx` → `y`, `uv run` → `n`) | usability trap | nothing (doc fix) |
| **L4** | uv **rejects** two `epifolio` entries in one script block, so a fallback pair is not expressible — but `--with*` **overrides** a script block, so no edit is needed at all (M1c) | resolved | nothing |
| **F1** | Panels 5/6 flicker on hover: `mo.ui.anywidget` turns jscatter's synced `hovering` trait into a reactive dependency, forcing a kernel round-trip + full re-render of **both** canvases per mouse move | UX blocker | nothing (~5 LOC) |
| **F2** | Panel-3 rebuild silently **discards the active lasso** (`mo.state` recreated inside a reactive cell) | data-loss bug | nothing (~3 LOC) |
| **F3** | Panel-1/2 genomic window does **not** reach panels 5/6; panel-5 and panel-6 lassos are mutually invisible | missing feature | nothing (uses discarded `inv`) |
| **F4** | Selected cCREs cannot be **named** — every panel array is pre-aggregated | missing feature | see **F5** |
| **F5** ⭐ | **The locus/accession table already exists and already ships**: `tcga.atac.pca.ccre.pq` has exactly **2,348,854 rows** — bit-for-bit the CNV payload's `n_ccre` — carrying `chrom, start, end, rDHS, cCRE, class` | unblocks F4 | partially nothing; full lasso→cCRE needs one producer column |
| **F6** *(2026-09-04)* | Panel 6's color dictionary, **defined explicitly** in §8.1 (6 CN bins → hex, sourced from `copy_number_color_map.json`), re-verified empirically against the installed jscatter — not just by reading source — across 12 biosamples with zero mismatches; one un-triggered fallback closed with a guard; panel 5's apparent "random" colors are a **continuous cmap autoscaling per render**, not a mis-assignment; neither panel has a title — jscatter has no title primitive | cosmetic + one latent bug | nothing (~2-13 LOC per item) |

**The single most valuable fact in this report:** `n_ccre = 2,348,854` in the CNV header is *the same number of
rows* as every `*.ccre.pq` file in `itcr/epifolio/embeddings/`. The join key we assumed was missing is a
project-wide convention that the CNV payload simply never referenced.

---

## 1. Launch failures

### L1 — declining the sandbox under `uvx` installs nothing

```
$ uvx marimo run notebooks/cnv_ccre_explorer.py
Run in a sandboxed venv containing this notebook's dependencies? [Y/n]: n
Installed 25 packages in 200ms
...
ModuleNotFoundError: No module named 'jscatter'
```

The notebook carries PEP 723 inline metadata (`notebooks/cnv_ccre_explorer.py:1-18`) declaring
`jupyter-scatter[all]>=1.0.1`, pandas, numpy, and `epifolio`. **On the `uvx` path the sandbox is the only
mechanism that installs any of it.** The "25 packages" line is `marimo` and its own transitive deps — nothing
from the script block. `import jscatter` is the first statement of the first cell (`:28`), so it raised
immediately and marimo's dataflow graph marked all 23 downstream cells `An ancestor raised an exception`.

The active `(epifolio-notebooks)` venv is *not* a fallback: `uvx` deliberately ignores `VIRTUAL_ENV` and runs
its tool in an isolated environment. Verified:

```
$ uvx --from marimo==0.21.1 python -c "import jscatter"
ModuleNotFoundError: No module named 'jscatter'
```

…while the project venv has it:

```
.venv/lib/python3.11/site-packages/jscatter   → jupyter_scatter 1.0.1 ✓
```

### L2 — the sandbox installs a real `epifolio` that predates the `cnv` module

Accepting the sandbox gets `jscatter` and fails one import later, at `:33`:

```
ModuleNotFoundError: No module named 'epifolio.cnv'
```

Cause is `notebooks/cnv_ccre_explorer.py:13`:

```python
"epifolio @ git+https://github.com/abdenlab/epifolio-notebooks.git#subdirectory=lib",
```

No `@ref`, so uv resolves **abdenlab's default branch**. `cnv.py` is not on it:

```
$ git ls-tree -r --name-only upstream/main -- lib/src/epifolio | grep -E 'cnv|nmf'
lib/src/epifolio/nmf.py          ← present
                                 ← no cnv.py
```

The module exists in exactly one commit, `aa8f226`, and that commit is **not on abdenlab at all**. The remotes
are split:

```
origin    https://github.com/vedatonuryilmaz/dev_epifolio-notebooks.git   ← has feat/cnvnb (aa8f226)
upstream  https://github.com/abdenlab/epifolio-notebooks.git              ← no feat/cnvnb branch
```

`git ls-remote upstream` lists no `feat/cnvnb`. So the notebook's dependency block currently points at a
published version of itself that cannot run it. This is the promise in [README.md](README.md) line 56 —
"uv reads this and installs all deps, including `epifolio` itself" — holding only for code that has landed
upstream.

Confirmed the fork pin resolves and ships the module:

```
$ uv pip install --no-deps \
  "epifolio @ git+https://github.com/vedatonuryilmaz/dev_epifolio-notebooks.git@aa8f226#subdirectory=lib"
→ site-packages/epifolio/cnv.py present ✓
```

Note the ref must be a bare SHA or a plain branch name; `@feat/cnvnb` and `@refs/heads/feat/cnvnb` both make
uv look for `refs/tags/…` and fail with `couldn't find remote ref`.

### L3 — the prompt is a trap: opposite correct answers per launcher

The prompt is triggered by the **inline script block in the file**, not by `uvx`. It therefore appears on the
`uv run` path too, and the safe answer inverts:

| Launcher | Answer `y` (sandbox) | Answer `n` (ambient) |
|---|---|---|
| `uvx marimo run …` | deps installed, but `epifolio` from abdenlab main → **L2 failure** | no deps at all → **L1 failure** |
| `uv run marimo run …` | same isolated sandbox → **L2 failure** | project `.venv`, editable `lib/` → **works** |

Both observed failures are this table's three red cells. The working invocation, verified end-to-end this time
by driving a real websocket session so the cells actually executed:

```
$ uv run --no-sync marimo run --no-sandbox --headless --no-token --port 2791 \
    notebooks/cnv_ccre_explorer.py

ops seen: ['cell-op', 'completed-run', 'kernel-ready', 'model-lifecycle', ...]
error payloads: 0
$ grep -c -iE "traceback|ModuleNotFound|error_id" run.log
0
```

Zero error_ids, against 23 in the failing runs. And the env `uv run` provides is correct:

```
$ uv run --no-sync python -c "import epifolio.cnv as c, jscatter, sys; print(sys.prefix, c.__file__)"
/…/dev_epifolio-notebooks/.venv   /…/lib/src/epifolio/cnv.py
```

### L4 — a fallback pair is not expressible, but an override is

Keeping the abdenlab URL and *adding* the fork pin beside it does not work. uv errors before resolution:

```
error: Requirements contain conflicting URLs for package `epifolio`:
- git+https://github.com/abdenlab/epifolio-notebooks.git#subdirectory=lib
- git+https://github.com/vedatonuryilmaz/dev_epifolio-notebooks.git@aa8f226#subdirectory=lib
```

One distribution name maps to exactly one URL; no resolver does "try A, fall back to B," and PEP 508 markers
select on interpreter and platform, never on "does this repo contain a `cnv` module."

**This turns out not to matter.** The substitution belongs on the command line, not in the file — see M1c.

### M1 — launch mitigations

| ID | Action | Effort | Notes |
|---|---|---|---|
| **M1a** | Add `--no-sandbox` to every local invocation in [README.md](README.md) (lines 98-101, 117-120) and [cnvintegration.md](cnvintegration.md) (lines 493-496) | 8 lines | Never rely on the prompt; the correct answer depends on the launcher |
| **M1b** | Push `feat/cnvnb` to `abdenlab` and merge | — | **The real fix.** The existing unpinned URL starts working with no file edit |
| **M1c** ⭐ | Use the **no-edit override routes** below for dev, prod-shaped testing, and sharing | 0 lines | **Nothing needs editing.** `--with*` overrides a script block; the committed file stays merge-ready |
| **M1f** | Fork pin on a throwaway `demo/cnvnb` branch, only for a bare shareable URL | 1 line, never merged | Keeps `feat/cnvnb` clean so merge day needs no revert |
| **M1d** | Document that `marimo export html` is broken for this notebook | 2 lines | `base64.b64encode` receives `str` from the jscatter buffers in `marimo/_messaging/notification.py:223`. Crash is **after** all cells run — export-only, does not affect `marimo run`. Upstream marimo bug; worth an issue |

### M1c — running it without editing the notebook ⭐

**No edit to the committed file is required for any of the three things you might want.** The mechanism is an
asymmetry in how uv treats overrides:

| Combination | Result |
|---|---|
| `--with-editable ./lib` + **script block** dependency | local lib **wins**, no error ✅ |
| `--with-editable ./lib` + `--with-requirements` entry | `conflicting URLs for package epifolio` ❌ |

`--with*` flags override a PEP 723 script block but *conflict* with `--with-requirements`. One `grep -v` removes
the conflict, which gives three no-edit routes:

**Route 1 — open it locally (dev).** Works today against the editable `lib/`:

```bash
uv run marimo run --no-sandbox notebooks/cnv_ccre_explorer.py
```

**Route 2 — test the production-shaped environment against your working tree.** This is the one that answers
"prove it works before merging": it builds the *sandbox* environment, not your dev venv, then substitutes your
local `lib/`:

```bash
uv export --script notebooks/cnv_ccre_explorer.py | grep -v '^epifolio @' > /tmp/req.txt
uv run --isolated --no-project --with-requirements /tmp/req.txt --with-editable ./lib \
  marimo run --no-sandbox notebooks/cnv_ccre_explorer.py
```

Verified: `Installed 77 packages` — the same count as the real sandbox — with `epifolio` resolving to
`lib/src/epifolio/cnv.py`, uncommitted edits included.

**Route 3 — someone else runs it with no checkout, committed file untouched.** Same shape, fork URL in place of
the editable path:

```bash
uv run --isolated --no-project --with-requirements /tmp/req.txt \
  --with "epifolio @ git+https://github.com/vedatonuryilmaz/dev_epifolio-notebooks.git@aa8f226#subdirectory=lib" \
  marimo run --no-sandbox notebooks/cnv_ccre_explorer.py
```

Worth wrapping Routes 2 and 3 in a `make preview` / `make preview-remote` target so nobody has to remember the
`grep`.

Two things that do **not** work, recorded so they are not retried:

- `UV_OVERRIDE` — silently ignored for script-block dependencies; the git URL still won.
- marimo's own `--sandbox` cannot be told to substitute a dependency. `construct_uv_command`
  (`marimo/_cli/sandbox.py:225-260`) builds `uv run --with-requirements <tmpfile>` from the script block and only
  layers `--with` from an `additional_deps` parameter that is reachable from the quarto plugin, not from any
  `marimo run` CLI flag. Route 2 replicates that command with the override added.

### M1f — the fork pin, demoted

A pin edited into `:13` is now **only** needed for a bare shareable URL — someone who will not paste four flags.
It should **not** live on `feat/cnvnb`, because a merge-blocking edit cannot be the mechanism for testing before
merge. Put it on a throwaway branch instead, regenerated whenever a demo is needed:

```bash
git checkout -B demo/cnvnb feat/cnvnb
sed -i '' 's|abdenlab/epifolio-notebooks.git#|vedatonuryilmaz/dev_epifolio-notebooks.git@aa8f226#|' \
  notebooks/cnv_ccre_explorer.py
git commit -am "demo pin — DO NOT MERGE" && git push -f origin demo/cnvnb
```

Share:

```bash
uvx marimo run https://raw.githubusercontent.com/vedatonuryilmaz/dev_epifolio-notebooks/demo/cnvnb/notebooks/cnv_ccre_explorer.py
```

**The PR branch never carries the pin, so merge day needs no revert.** Verified separately that a pinned copy runs
under a true `uvx marimo run --sandbox`: `Installed 77 packages`, live server, **zero** tracebacks or `error_id`s
in the log. Use the SHA, not `@feat/cnvnb` — a branch name sends uv looking under `refs/tags/…` and it fails with
`couldn't find remote ref`; a SHA is also immutable against a force-push to the fork.

**Operational note discovered while testing:** `lib/src/epifolio/assets/data/cnv_ccre/` does **not** exist in the
working tree, so `cnv.source()` (`lib/src/epifolio/cnv.py:165-172`) falls through past the unpopulated canonical
mirror to the interim host. Every read in this investigation resolved to:

```
https://users.abdenlab.org/voy/itcr/epifolio/notebooks/cnv_ccre/summary_genome_tracks.parquet
```

This is exactly the exposure [cnvintegration.md](cnvintegration.md) §4.7 / risk table line 713 predicted, and it
is live right now on the default path — not just the `uvx`/Molab path.

---

## 2. F1/F2 — the panel 5/6 flicker is a kernel round-trip per mouse move

Not a rendering artifact. The chain, end to end:

**Step 1 — jscatter syncs hover to the kernel.** `jscatter/widget.py:93`:

```python
hovering = Int(None, allow_none=True).tag(sync=True)
```

`sync=True` is bidirectional, so every point the cursor crosses pushes a trait update to Python.

**Step 2 — marimo accepts any trait as a new element value.** `marimo/_plugins/ui/_impl/from_anywidget.py`,
`anywidget._convert_value` — no filtering by trait name:

```python
self.widget.set_state(value)
return cast(AnyWidgetState, value)
```

**Step 3 — the reactive graph fans it out.** `panel5_widget` gets a new value per hover tick, so marimo reruns
every cell referencing it, minus the defining cell — the rule the notebook already documents at
`notebooks/cnv_ccre_explorer.py:196-198`. That is the readout cell at `:255`, which references `panel5_widget`
**only** to compose `mo.vstack([panel5_widget, readout5])`. It defines `panel5`; the layout cell at `:357`
references `panel5` *and* `panel6`; so the entire bottom row re-renders and **both** WebGL canvases remount.
That is precisely why moving the cursor from panel 5 to panel 6 flickers both.

**Step 4 — and the rerun is not cheap.** With no lasso active the readouts fall back to the full table
(`:260`, `selected_dens = dens_tbl`) and do a pure-Python pass over it. Measured on
`ACCx-TCGA-OR-A5J2-01A-31-A616-42-X013-S12`:

| Table | Rows | Build | Per-hover Python pass |
|---|---|---|---|
| `density_table` (panel 5) | 8,461 | 19.5 ms | **11.1 ms** |
| `cn_table` (panel 6) | 23,370 | 9.1 ms | **20.2 ms** |

≈31 ms of kernel work plus a websocket round-trip, per hover event, feeding a full DOM remount of two canvases.

### The repo already solved this

`notebooks/nmf_sample_explorer.py:366-369` deliberately does **not** wrap:

```python
# Display the raw jscatter widget (not wrapped in mo.ui.anywidget) so its
# lasso selection fires the trait observer registered below ...
scatter_widget = scatter.widget
```

It renders that raw widget in its layout (`:750`) and gets selection purely through `.observe` (`:384`). A raw
widget is not a marimo `UIElement`, so referencing it creates **no reactive edge** and hover triggers nothing.
The CNV notebook already registers the identical `.observe` → `mo.state` wiring at `:246-248` — the
`mo.ui.anywidget` wrapper at `:249` / `:309` is redundant on top of it and is the sole source of the flicker.

### F2 — the lasso is silently destroyed by a panel-3 click

`get_sel5, set_sel5 = mo.state(None)` lives *inside* the scatter cell (`:247`), which depends on `active_cn`.
Clicking panel 3 reruns that cell, so `mo.state` is recreated from scratch and the user's selection vanishes with
no feedback. Same for panel 6 at `:307`.

### M2 — flicker and selection-loss mitigations

| ID | Action | Files / LOC | Risk |
|---|---|---|---|
| **M2a** | Drop `mo.ui.anywidget` at `:249` and `:309`; render `scatter5.widget` / `scatter6.widget` raw, matching `nmf_sample_explorer` | notebook, ~5 LOC | **Low.** The `.observe` → `mo.state` selection path already present keeps working unchanged. Blast radius is the two readout cells' dependency lists |
| **M2b** | Hoist both `mo.state` pairs into their own cell with no dependency on `active_cn`, and clear the selection **visibly** on a CN-filter change | notebook, ~3 LOC | Low. Fixes F2. Positional indices do not survive a table rebuild, so preserving them would highlight the wrong points — key selections by bin coordinate under M3 instead (§7.4) |
| **M2c** | Replace the readouts' `.to_pylist()` + Python loops (`:263-268`, `:323-332`) with Arrow compute (`pc.sum`, `pc.greater`, `group_by().aggregate()`) | notebook, ~12 LOC | Low. Removes the 11/20 ms passes; matters even after M2a because it also runs on every legitimate lasso |
| **M2d** | File the `hovering`-trait issue upstream with marimo: an opt-in trait allowlist on `mo.ui.anywidget` (e.g. `mo.ui.anywidget(w, watch=["selection"])`) would fix this class of bug for every widget, not just jscatter | — | Not a blocker; M2a is sufficient locally |

---

## 3. F3 — selection propagation map

Tracing every reactive edge in the notebook as committed:

| Interaction | What it drives | Reaches panels 5/6? |
|---|---|---|
| **Panel 1** (CN per cCRE across genome) box/zoom | `sync1` → `clamp_window` → `set_window` **only** (`:168-173`) | **No** |
| **Panel 2** (ATAC track) zoom | same shared window | **No** |
| **Panel 3** (cCREs per copy number, bar) click/box | `bar_levels` → `active_cn` → panels **4, 5, 6** | **Yes** — as a categorical CN filter |
| **Panel 5** lasso | `sel5` → panel 5's own readout | no cross-talk |
| **Panel 6** lasso | `sel6` → panel 6's own readout | no cross-talk |

Panel 1's `on_change` **discards the selected points entirely** — it reads `panel1.ranges` for the axis window
and nothing else. Neither scatter cell (`:224`, `:281`) references `get_window()`, so the genomic window is
structurally invisible to the bottom row. **The only top-panel control that reaches panels 5/6 is panel 3's
copy-number filter**, and it acts as a filter on which CN levels enter the tables — not as a materialized point
selection. A lasso in 5 and a lasso in 6 are mutually invisible, and neither feeds back upward.

### M3 — link panels 5 ↔ 6 (notebook-local, no data change)

Both tables derive from the same `p6_row`, and `density_table`'s own docstring establishes the join is
**bit-exact**: *"both axes are drawn from the same GRID_CENTERS, so grouping on the raw float pair is
bit-exact."* The mapping already exists and is thrown away — `lib/src/epifolio/cnv.py:688`:

```python
uniq, inv = np.unique(pairs, axis=0, return_inverse=True)   # ← inv is discarded
```

`inv` is a total map from panel-6 row → panel-5 bin.

| ID | Action | Files / LOC | Risk |
|---|---|---|---|
| **M3a** | Return `inv` from `density_table` (or a sibling `density_table_with_index`) | `cnv.py`, ~3 LOC | Low. Additive; keep the existing return shape to avoid touching call sites, or return a small dataclass |
| **M3b** | Panel-6 lasso → panel-5 highlight: `np.unique(inv[sel6])` gives the panel-5 bins to highlight via `scatter5.widget.selection` | notebook, ~6 LOC | Low |
| **M3c** | Panel-5 lasso → panel-6 highlight: `np.flatnonzero(np.isin(inv, sel5))` gives the panel-6 rows | notebook, ~6 LOC | Low. Guard against feedback loops — set the sibling's `selection` only when it differs, exactly as `nmf_sample_explorer.py:389-392` does |
| **M3d** | Show a shared cCRE-weighted readout for the linked selection, replacing the two independent ones | notebook, ~10 LOC | Low |

This is the largest real gain available with **zero** producer involvement.

---

## 4. F4/F5 — naming the selected cCREs: the key is not missing ⭐

### 4.1 Why no panel can name a cCRE today

Full column inventory of the CNV payload:

```
summary_genome_tracks.parquet            summary_signal_distributions.parquet
  p1_x  p1_y  p1_n        (panel 1)        violin_cancer_x/y, violin_healthy_x/y,
  p2_x  p2_y  p2_n        (panel 2)        violin_n, violin_q_cancer/healthy   (panel 4)
  cn_counts               (panel 3)        scatter_cn_x  scatter_cn_y
  biosample cancer patient_id  (index)     scatter_cn_level  scatter_cn_v      (panels 5+6)
  age diagnosis encode_* n_ccre sex tissue (header)
  cn_order                (shared)         biosample cancer patient_id, cn_order
```

No accession, no `chrom`/`start`/`end`, nothing joinable — in **either** file. And every array is already
reduced, including the two that sound per-cCRE:

| Array | Rows (ACCx / BLCA / BRCA) | What a row is |
|---|---|---|
| `n_ccre` (header scalar) | **2,348,854** — identical for all 386 biosamples | the cCRE universe |
| `p1_x` | 4,681 / 4,749 / 4,751 | CN **segment**, despite `panel_1`'s "at each cCRE" docstring |
| `p2_x` | 257,993 / 245,973 / 282,984 | binned ATAC track position |
| `scatter_cn_*` | 23,370 / 28,542 / 30,346 | **2D-histogram bin** on `GRID_EDGES`; `scatter_cn_v` is a *count* |

Panels 5 and 6 read the *same* four `scatter_cn_*` arrays — panel 6 keeps the `(level, x, y)` bins, panel 5
aggregates them again over level. A lasso point in panel 6 is a bin holding hundreds of cCREs. The aggregation
is irreversible, and the readouts are honest about it: they report cCRE *counts* (`sum(w5)`, `total6`), never
identities.

### 4.2 The locus table already exists, already ships, and already matches

`notebooks/nmf_ccre_explorer.py:44` reads a file nobody wired into the CNV notebook:

```python
df = pl.scan_parquet("https://projects.abdenlab.org/itcr/epifolio/embeddings/tcga.atac.pca.ccre.pq")
```

Its schema — 56 columns, of which six are exactly what F4 needs:

```
chrom  start  end  rDHS  cCRE  class  PC1 … PC50

┌───────┬───────┬───────┬──────────────┬──────────────┬─────────┐
│ chrom ┆ start ┆ end   ┆ rDHS         ┆ cCRE         ┆ class   │
╞═══════╪═══════╪═══════╪══════════════╪══════════════╪═════════╡
│ chr1  ┆ 10033 ┆ 10250 ┆ EH38D4327497 ┆ EH38E2776516 ┆ pELS    │
│ chr1  ┆ 10385 ┆ 10713 ┆ EH38D4327498 ┆ EH38E2776517 ┆ pELS    │
│ chr1  ┆ 16097 ┆ 16381 ┆ EH38D6144701 ┆ EH38E3951272 ┆ CA-CTCF │
└───────┴───────┴───────┴──────────────┴──────────────┴─────────┘
```

**And the row count is `2,348,854` — bit-for-bit the CNV payload's `n_ccre`.**

This is a project-wide convention, not a coincidence. Every `*.ccre.pq` in `itcr/epifolio/embeddings/` carries
the identical six metadata columns and the identical row count:

```
adjusted.nmf.ccre.pq     adjusted.pca.ccre.pq     confounded.nmf.ccre.pq
confounded.pca.ccre.pq   tcga.atac.nmf.ccre.pq    tcga.atac.pca.ccre.pq
```

Verified:

```
tcga.atac.nmf.ccre.pq   cols= 30  meta=['chrom','start','end','rDHS','cCRE','class']  rows=2,348,854
adjusted.pca.ccre.pq    cols= 56  meta=['cCRE','chrom','class','end','rDHS','start']  rows=2,348,854
tcga.atac.pca.ccre.pq   cols= 56  meta=['chrom','start','end','rDHS','cCRE','class']  rows=2,348,854
```

Load cost for the five columns that matter:

| Metric | Value |
|---|---|
| Rows × cols | 2,348,854 × 5 |
| Remote `scan_parquet` → `collect` | **3.6 s** |
| In-memory | **81.7 MB** |
| `class` vocabulary | `PLS, pELS, dELS, CA, CA-CTCF, CA-TF, CA-H3K4me3, TF` |
| Distinct chroms | 24 — matches `len(cnv.CHROMS)` |
| Sorted by `start` within `chrom` | ✅ 0 out-of-order rows |

### 4.3 ⚠️ The ordering gotcha — do not join by row position

The cCRE parquet orders chromosomes **lexicographically**; `cnv.CHROMS` (from bioframe) is **karyotypic**:

```
cCRE parquet : chr1, chr10, chr11, chr12, chr13, chr14, …
cnv.CHROMS   : chr1, chr2,  chr3,  chr4,  chr5,  chr6,  … chr22, chrX, chrY
```

So **parquet row index ≠ the genome-linear order underlying `CHROM_OFFSET` / `GENOME_LEN = 3088269832`.** Any
link must key on `(chrom, start)`, or re-sort explicitly into `cnv.CHROMS` order once at load time. A positional
`zip` would silently mis-assign ~2.3 M cCREs and produce plausible-looking nonsense. This is the single highest-risk
detail in this whole document.

### 4.4 M4 — what the locus table unlocks *today*, with no producer change

Panel 1/2 x-coordinates are genome-linear offsets, and `cnv.py:321-325` already inverts them:

```python
boundaries = [CHROM_OFFSET[c] for c in CHROMS] + [GENOME_LEN]
idx = np.searchsorted(boundaries, x, side="right") - 1
```

Given the shared window `get_window() == (lo, hi)`, decompose it into per-chromosome local intervals and range-join
the locus table. This yields **"which cCREs are in the current genomic window"** — real, nameable, exportable —
without touching the producer.

| ID | Action | Files / LOC | Risk |
|---|---|---|---|
| **M4a** | Add `epifolio.cnv.load_loci(columns=…) -> pl.DataFrame` reading `tcga.atac.pca.ccre.pq` **as delivered**, selecting only the six metadata columns, with the `cnv.CHROMS` re-sort and the genome-linear `pos` derivation done *in the loader* | `cnv.py`, ~25 LOC | Low. **Constraint: read the file in its shipped layout. Do not propose re-sharding, re-sorting on disk, or a new derived file** — all variability belongs in the loader, exactly as `source()`/`panel_tags()` already handle payload variation |
| **M4b** | Add `CCRE_INDEX` alongside `REMOTE`/`FALLBACK` (`cnv.py:150-152`), plus the same local→canonical→fallback fall-through, so the locus table follows the established three-tier discovery | `cnv.py`, ~6 LOC | Low |
| **M4c** | Wrap the load in `mo.cache` in the notebook, as `read_row` already is (`:60-64`) — 3.6 s / 81.7 MB is a once-per-session cost, not per-interaction | notebook, ~4 LOC | Low. Watch the 50 MB `output_max_bytes` cap (`:16-17`): never render the whole frame, only a head/count |
| **M4d** | New panel or `mo.ui.table`: cCREs in the current window, filtered by `active_cn` when panel 3 is engaged, with a CSV download | notebook, ~30 LOC | Medium — this is net-new UI, and `--no-sandbox`/output-cap behaviour should be re-checked with it in place |
| **M4e** | Cross-check `class` against `panel_4`/`cn_counts` totals on one biosample before trusting any join | throwaway script | Low. Cheap guard against a silent mis-key |

**Scope limit, stated plainly:** M4 answers *"which cCREs are in this genomic window"*. It does **not** answer
*"which cCREs are in this z-score lasso"* — the window and the lasso are different selections over different
axes, and only the window has a coordinate to join on.

### 4.5 M5 — lasso → named cCREs (needs exactly one producer column)

To name the cCREs inside a panel-5/6 lasso, the producer must emit a per-cCRE → bin assignment. With the locus
table supplying `chrom/start/end/cCRE/class`, the ask is now **one small column**, not a locus re-export:

| Property | Spec |
|---|---|
| What | Per-biosample, per-cCRE bin assignment, **row-aligned to `*.ccre.pq` order** (state the ordering convention in file metadata) |
| Encoding | `bin_id` as `uint16` — `N_GRID = 192`, so `yi*192 + xi ≤ 36,863`, comfortably inside `uint16` — plus `cn_level` as `uint8` |
| Raw size | 3 bytes × 2,348,854 = **7.05 MB** per biosample |
| Compressed | Substantially less: cCREs are genome-sorted and CN is segmental, so `cn_level` is highly RLE/dictionary-compressible |
| Layout | One row group per biosample so a selection fetches ~7 MB, not the whole 386-biosample set. Comparable to panel 2's existing ~258 K-point track and well inside the 50 MB cap |
| Consumer side | Lasso bins → `np.isin(bin_id, sel_bins)` → row indices → join the locus table for `chrom/start/end/cCRE/class`. Makes **M3a's `inv` unnecessary for identity**, though M3 remains the right zero-cost interim |

| ID | Action | Owner | Blocked on |
|---|---|---|---|
| **M5a** | Open a producer request for the `bin_id`/`cn_level` column with the spec above, including the "row-aligned to `*.ccre.pq`, ordering declared in metadata" requirement | whoever generates `summary_signal_distributions.parquet` | — |
| **M5b** | Bump `SCHEMA_VERSION` (`cnv.py:32`) and extend `assert_edges_match` (`:65`) to validate the new column and its declared row count against `n_ccre` — fail loudly rather than mis-join | us, on delivery | M5a |
| **M5c** | *Resolved by §7.4 — no separate decision.* Keying selections by bin coordinate (done as part of M3) makes them survive a panel-3 click by construction, with or without M5 | us | — |
| **M5d** | *Alternative if the producer prefers:* per-cCRE `cancer_z`/`healthy_z` as `float16` (+`cn_level`) — 5 bytes/cCRE, ~11.7 MB/biosample, and lets the client re-bin at any `N_GRID`. Strictly more flexible, ~65% larger | producer's call | M5a |

If the user is offering to supply a parquet directly: the highest-value file is **not** another locus table — that
exists. It is this per-biosample `bin_id` column, row-aligned to `tcga.atac.pca.ccre.pq`.

---

## 5. Impact radius, consolidated

| Tier | Change | Files | LOC | Data change | Risk | Blocked on |
|---|---|---|---|---|---|---|
| **0** | M1a launch flags in docs | README, 2 docs | ~8 | none | low | nothing |
| **0** | **M1c no-edit override routes** (+ optional `make preview` targets) | Makefile / docs | ~6 | none | low | nothing |
| **0** | M2a unwrap anywidget → **kills the flicker** | notebook `:249,:309` | ~5 | none | low | nothing |
| **0** | M2b hoist `mo.state` → stops selection loss | notebook | ~3 | none | low | nothing |
| **0** | M2c Arrow-compute readouts | notebook | ~12 | none | low | nothing |
| **1** | M3 link panels 5 ↔ 6 via the discarded `inv` | `cnv.py:688`, notebook | ~25 | none | low | nothing |
| **1** | M4a-c `load_loci()` + three-tier discovery + cache | `cnv.py`, notebook | ~35 | **none — reads a file that already ships** | low | nothing |
| **2** | M4d window → named-cCRE table/panel + export | notebook | ~30 | none | medium | M4a |
| **3** | M5 lasso → named cCREs | producer + `cnv.py` + notebook | ~40 | **one new column** | medium | producer |
| **—** | M1b push `feat/cnvnb` upstream | — | — | none | low | maintainer |
| **0** | M6a close `cn_colors()`'s str/label mismatch | `cnv.py:136-142` | ~4 | none — same JSON | low | nothing |
| **0** | M6e guard: assert no CN label falls outside `color_map6` | `cnv.py:789-803` | ~2 | none | low | nothing |
| **0** | M6b pin panel-5's color domain (`color_norm`) | notebook, new cell + `:236-269` | ~8 | none | low | nothing |
| **1** | M6d panel-5/6 titles | notebook `:236-269`, `:272-301`, `:415-439` | ~10 | none | low | nothing |
| **2** *(optional)* | M6c discrete density-bin legend | `cnv.py` + new JSON + notebook | ~35 | **one new asset file** | medium | M6b landed first |

**Recommended order:** M2a → M2b → M1a/M1c → M4a-c → M3 → M4d, with M1b and M5a opened in parallel since both
have external lead time. M2a alone resolves the reported flicker; M4a-c is the cheapest path to naming cCREs and needs nothing
from anyone.

---

## 6. Evidence appendix — reproduce every number

```bash
# Launch (the only correct local invocation)
uv run marimo run --no-sandbox notebooks/cnv_ccre_explorer.py

# L1 — uvx tool env is isolated from the active venv
uvx --from marimo==0.21.1 python -c "import jscatter"

# L2 — the cnv module is not upstream
git ls-tree -r --name-only upstream/main -- lib/src/epifolio | grep -E 'cnv|nmf'
git ls-remote upstream 'refs/heads/*'

# L4 — uv refuses two entries for one package
printf '%s\n' '# /// script' '# dependencies = [' \
  '#     "epifolio @ git+https://github.com/abdenlab/epifolio-notebooks.git#subdirectory=lib",' \
  '#     "epifolio @ git+https://github.com/vedatonuryilmaz/dev_epifolio-notebooks.git@aa8f226#subdirectory=lib",' \
  '# ]' '# ///' 'import epifolio' > /tmp/dup.py
uv run --no-project /tmp/dup.py    # → error: conflicting URLs for package `epifolio`

# M1c Route 2 — prod-shaped env against the local working tree, no file edit
uv export --script notebooks/cnv_ccre_explorer.py | grep -v '^epifolio @' > /tmp/req.txt
uv run --isolated --no-project --with-requirements /tmp/req.txt --with-editable ./lib \
  python -c "import epifolio.cnv as c; print(c.__file__)"     # → lib/src/epifolio/cnv.py

# M1c Route 3 — same, but from the fork, for someone with no checkout
uv run --isolated --no-project --with-requirements /tmp/req.txt \
  --with "epifolio @ git+https://github.com/vedatonuryilmaz/dev_epifolio-notebooks.git@aa8f226#subdirectory=lib" \
  python -c "import epifolio.cnv as c; print(c.__file__)"

# The asymmetry that makes it work: --with* overrides a script block, but
# conflicts with --with-requirements. Hence the grep -v above.

# M1f — a pinned copy under a true sandbox (expect 0 tracebacks in the log)
sed 's|abdenlab/epifolio-notebooks.git#|vedatonuryilmaz/dev_epifolio-notebooks.git@aa8f226#|' \
  notebooks/cnv_ccre_explorer.py > /tmp/pinned.py
uvx marimo run --sandbox --headless --no-token --port 2793 /tmp/pinned.py

# NB. Verify runs by grepping the server log for `traceback|ModuleNotFound|error_id`.
# A websocket probe matching only output.channel == "marimo-error" misses cell
# failures that arrive on other channels — it reported 0 on a run whose log had 21.

# F1 — the offending synced trait
grep -n "hovering" .venv/lib/python3.11/site-packages/jscatter/widget.py

# F1 — panel sizes and per-hover cost
uv run --no-sync python - <<'PY'
import time, epifolio.cnv as cnv
D = cnv.open_dataset([cnv.source(n) for n in (cnv.TRACKS, cnv.DISTRIBUTIONS)])
C = cnv.load_cohort(D)
cancer = sorted(C)[0]; pat = sorted(C[cancer])[0]; bs = C[cancer][pat][0]
cn = [str(l) for l in cnv.load_panel(D, bs, "header", "shared")["cn_order"]]
r6 = cnv.load_panel(D, bs, "6")
d = cnv.density_table(r6, cn, cn); c = cnv.cn_table(r6, cn, cn)
print(len(d), "panel5 rows;", len(c), "panel6 rows")
t = time.perf_counter()
cz, hz, w = (d[k].to_pylist() for k in ("cancer_z", "healthy_z", "cCREs"))
sum(wi for ci, hi, wi in zip(cz, hz, w) if ci > hi)
print(f"readout5 python pass {(time.perf_counter()-t)*1000:.1f} ms")
PY

# F5 — the locus table matches n_ccre exactly
uv run --no-sync python - <<'PY'
import polars as pl
lf = pl.scan_parquet("https://projects.abdenlab.org/itcr/epifolio/embeddings/tcga.atac.pca.ccre.pq")
print([c for c in lf.collect_schema().names() if not c.startswith("PC")])
print("rows:", lf.select(pl.len()).collect().item())   # 2,348,854 == n_ccre
PY

# F5 — every *.ccre.pq shares the convention
curl -sL https://projects.abdenlab.org/itcr/epifolio/embeddings/ \
  | grep -o -E "[A-Za-z0-9._-]+\.ccre\.pq" | sort -u

# 4.3 — the ordering trap
uv run --no-sync python -c "import epifolio.cnv as c; print(c.CHROMS[:6])"   # karyotypic
# vs the parquet's lexicographic chr1, chr10, chr11, …
```

---

## 7. Decisions

1. **M1b timing — push and merge; do not carry a pin on `feat/cnvnb`.** The branch stays merge-ready and
   unmodified; testing and demos go through the M1c override routes, and a bare shareable URL goes on the
   throwaway `demo/cnvnb` branch (M1f). Merge day therefore needs no revert. Add a guard so a pin can never
   reach the PR branch by accident:

   ```bash
   ! grep -rq "vedatonuryilmaz" notebooks/ || { echo "fork pin present"; exit 1; }
   ```

   The reason to hurry is unchanged: while the code ships from a personal fork it duplicates the exact
   single-point-of-failure already carried by the data host.

2. **Data stays remote — this is settled, and the current design is correct.** All payloads live on abdenlab
   servers and are read from there; the `FALLBACK` tier is doing its job until the canonical mirror is
   populated, at which point it is a one-constant deletion (`cnv.py:152`). No local hosting, no committed
   payload — [.gitignore](.gitignore) already states the local directory is a dev convenience, never tracked.

   Two things still worth fixing, neither of which changes that design:

   - **`_reachable()` probes with `HEAD` only** (`cnv.py:155-162`). If the canonical host ever answers HEAD
     differently from GET, `source()` keeps silently choosing the fallback and the migration looks complete
     when it is not. Once `REMOTE` is populated the probe has no job left — `source()` becomes local-else-
     canonical, which also removes a 5-second stall per file when offline.
   - **The failure message is opaque.** With the host down, `open_dataset` dies inside
     `polars.read_parquet_metadata` with an object-store 502 after a 10-retry, ~180 s budget. A short
     "payload unreachable at `<url>` — the abdenlab host may be down" beats a polars stack trace.

   *Observed while writing this:* every abdenlab path — both mirrors and `embeddings/` — returned **502** for
   HEAD and ranged GET, having served fine an hour earlier. Transient, but it is what a user meets with no
   local copy, so the message above is worth the ten lines.

3. **`bin_id`, not `float16` z-scores (M5, not M5d).** Panels only ever render bins; `N_GRID = 192` is a
   constant baked into `GRID_EDGES` and validated by `assert_edges_match()`, so the client has no use for
   re-binning. `bin_id` is smaller (7 MB vs 11.7 MB per biosample) and compresses far better — cCREs are
   genome-sorted and copy number is segmental, so `cn_level` is long RLE runs, while float16 z-scores are
   high-entropy. The decisive argument is correctness, not size: `density_table`'s grouping is bit-exact
   *because* both axes come from the same `GRID_CENTERS`, and shipping float16 z-scores would create a second,
   lower-precision path to the same bin assignment that could disagree at bin edges — reintroducing the class
   of silent mis-join that §4.3 warns about. `bin_id` keeps the producer as the single source of truth.

4. **Key selections by bin coordinate, and the clear-vs-remap question disappears.** It is only awkward today
   because `sel5`/`sel6` are *positional indices* into tables that are rebuilt when `active_cn` changes — row
   4,102 addresses a different bin before and after, so preserving the indices highlights the wrong points
   silently, which is worse than losing the selection. A bin's identity is its `(cancer_z, healthy_z)` pair,
   and that is stable across a CN-filter change: filtering changes which cCREs land in a bin, never where the
   bin is.

   - **Now (~3 LOC):** clear on filter change, but *visibly* — a line in the readout saying so. Silent loss is
     the unacceptable part.
   - **With M3 (free):** M3 already handles `inv`, the bin↔row mapping. Key the `mo.state` on coordinate pairs
     instead of indices while you are in there. The selection then survives CN changes by construction, panel
     5↔6 cross-highlighting works, and **M5c stops being a separate decision** — it is the same mechanism.

   §7's original framing as M2b *vs* M5c was a false binary; both collapse into M3.

---

## 8. F6 — panel 5/6 color encoding and titles *(addendum, 2026-09-04)*

Investigated after a report that panel 5/6 colors "get assigned randomly." First pass (2026-09-04, morning)
was checked against the wrong tree — `/Users/voy/Downloads/jupyter-scatter-main`, an unrelated dev clone, not
what the project actually runs. Re-verified against the **installed** package,
`.venv/lib/python3.11/site-packages/jscatter` (`jupyter-scatter==1.0.1`, pinned in the notebook's script
block), which does differ from the dev clone (`diff` is non-empty) and carries one extra fallback path the dev
clone doesn't (§8.1). **No code changed by this addendum** — same investigation-only status as the rest of this
report.

### 8.1 Panel 6 — the color dictionary, defined

The dictionary is not implicit — it's this, sourced from `lib/src/epifolio/assets/conf/copy_number_color_map.json`
(already shipping, read via `cn_colors()`, `cnv.py:136-142`):

| Copy-number bin | Hex | Swatch meaning |
|---|---|---|
| `0` | `#2c7bb6` | homozygous loss |
| `1` | `#abd9e9` | single-copy loss |
| `2` | `#dfdfdf` | diploid / neutral |
| `3` | `#fdae61` | single-copy gain |
| `4+` | `#d7191c` | amplification |
| `Missing` | `#404040` | no call |

`notebooks/cnv_ccre_explorer.py:275-276` builds this into `color_map6` (re-keyed through `cn_label()` to
`"0.0"`/`"1.0"`/… to match the table's column values) and passes it to `jscatter.Scatter(color_map=color_map6)`
at `:287`. Confirmed in the **installed** jscatter source that a dict `color_map` is matched to the data **by
label, not position** — `get_map_order()` (`jscatter.py:147-153`) builds `order[code] = map_keys.index(cat)`
per category, so pandas' own (alphabetical) category-code order never leaks through.

**Re-verified empirically, not just by reading source**, against the real installed package and real payload
data — constructing the actual `cn_tbl6` / `color_map6` / `jscatter.Scatter(...)` objects `cnv.py` and the
notebook produce, across 12 (cancer, patient) pairs and simulated `active_cn` subsets (mimicking a panel-3
click), then checking `map_order[order[code]] == cat` for every category on every widget:

```
n_checked: 12
distinct cn_order tuples seen: {('0', '1', '2', '3', '4+', 'Missing')}
mismatches: 0
```

`cn_order` itself is invariant across every biosample checked — it is not a per-biosample set that needs
separate handling, it is this fixed 6-entry table above, every time. Every category, every biosample, every
partial CN-filter subset tested resolved to the correct color. **If the running notebook is still showing
different colors than the table above** (e.g. a glasbey/okabe rainbow rather than the blue→grey→red ramp), that
is a live discrepancy this check did not reproduce — the next step would be a screenshot of the mismatch plus
which biosample/CN-filter state it followed, so the actual triggering path can be chased rather than guessed at.

**One installed-version-specific fallback worth knowing about, even though not observed to trigger:** the
installed 1.0.1 (`jscatter.py:1698-1706`, absent from the dev clone) silently **discards the whole dict** and
reverts to the auto glasbey/okabe palette if the data ever carries more distinct categories than the dict has
entries for:

```python
if (
    self._color_categories is not None
    and self._color_map is not None
    and len(self._color_categories) > len(self._color_map)
):
    self._color_map = None
    self._color_map_order = None
```

Today `keep = np.isin(lvl, [cn_label(c) for c in active_cn])` (`cnv.py:797`) guarantees `copy_number` can only
ever contain labels drawn from `active_cn ⊆ cn_order`, so the category count is always ≤ 6 = `len(color_map6)`
and this path stays cold. It would go live only if a future payload ever carried a CN label outside the six
above — worth a one-line assertion at load time (`assert set(lvl) <= set(color_map6)`) so a mismatch fails
loudly instead of silently reverting to random-looking colors, which is the exact symptom reported.

| ID | Action | Where | Calculation | LOC | Risk |
|---|---|---|---|---|---|
| **M6e** | Add `assert set(lvl) <= set(color_map6)` (or equivalent) right after building `lvl` in `cn_table()`, so an out-of-dictionary CN label fails loudly instead of silently falling back to glasbey/okabe | `cnv.py:789-803` | None — a set comparison against the already-built dict | ~2 | Low. Pure guard; only fires on data this dict doesn't already cover |

### 8.2 The pre-existing str/label fragility in `cn_colors()`

The fragility: `cn_colors()` (`cnv.py:136-142`) looks the JSON up by `str(level)` — bare digits, `"0"`/`"1"`/…,
matching the JSON's own keys — while every consumer of its *output* re-keys through `cn_label()`, which
reformats to `"0.0"`/`"1.0"`. The two paths agree today only because `cn_label("0")` round-trips to `"0.0"`. If
`cn_order`'s raw values ever arrive already decimal-formatted (`str(0.0) == "0.0"`), the JSON lookup misses and
every point silently falls back to `MISSING_GREY` — a real bug, just not the one currently on screen.

**Constraint that makes this fix non-trivial:** `cn_colors()`'s return dict is also consumed by `panel_3()`
(`cnv.py:530,540`) as `palette[c] for c in cn_order` — a lookup by the **raw**, unnormalized key. Re-keying
`cn_colors()`'s output through `cn_label()` to "fix" the fragility would break panel 3. The correct fix resolves
the *value*, not the *key*:

```python
def cn_colors(levels: Iterable[object]) -> dict[str, str]:
    color_map = load_color_map(ASSETS / "conf" / "copy_number_color_map.json")
    return {
        str(level): color_map.get(str(level)) or color_map.get(cn_label(level), MISSING_GREY)
        for level in levels
    }
```

| ID | Action | Where | Calculation | LOC | Risk |
|---|---|---|---|---|---|
| **M6a** | Try both the raw and `cn_label()`-normalized form when resolving each level against the JSON; keep the returned dict keyed by `str(level)`, unchanged, so `panel_3` and `color_map6`'s re-keying both keep working | `cnv.py:136-142` | None — pure lookup, no new numbers | ~4 | Low. Additive `or`-fallback; existing callers see identical output for today's data |

**Color dict location: no new file.** It's `lib/src/epifolio/assets/conf/copy_number_color_map.json`, already
shipping, already the single source of truth for panels 3 and 6. This fix only makes the code that reads it more
robust.

### 8.3 Panel 5 — the "randomness" is a moving color scale, not a mis-assignment

`color_by="log_cCREs", color_map="inferno"` (`cnv.py`-computed value, notebook `:250-251`) is continuous.
jscatter auto-normalizes a continuous map to the *current* data's min/max (`create_default_norm()`,
jscatter.py:392) every render. `log_cCREs` is recomputed per `active_cn` selection in `_density_table_impl`
(`cnv.py:730-757`), so the filtered subset's min/max — and therefore what a given shade of yellow *means* —
shifts on every panel-3 click or biosample switch. Within one render every point is colored correctly; across
renders the scale's meaning drifts. That drift is what reads as "random."

**Where the fix is calculated, and from what:** the domain must come from the biosample's **full, unfiltered**
`log_cCREs` range — not the `active_cn`-filtered one currently feeding the widget — so the scale stays fixed
while the user toggles copy-number states. `density_table_with_index()` (`cnv.py:778-783`) already computes
this; calling it once with `active_cn=cn_order` (every level, not the filtered subset) gives the full-range
table to pull `min`/`max` of `log_cCREs` from.

```python
# new cell, depends only on p6_row/cn_order — NOT active_cn, so marimo
# does not recompute it on every panel-3 click
full_density = cnv.density_table_with_index(p6_row, cn_order, cn_order).table
log_lo = pc.min(full_density["log_cCREs"]).as_py()
log_hi = pc.max(full_density["log_cCREs"]).as_py()
```

```python
# panel-5 cell, :244-260 — add one kwarg
scatter5 = jscatter.Scatter(
    ...,
    color_by="log_cCREs",
    color_map="inferno",
    color_norm=(log_lo, log_hi),   # ← new
    ...,
)
```

| ID | Action | Where | Calculation | LOC | Risk |
|---|---|---|---|---|---|
| **M6b** | Add a cell computing `(log_lo, log_hi)` from the **unfiltered** density table, keyed on `p6_row`/`cn_order` only; pass `color_norm=(log_lo, log_hi)` into the panel-5 `Scatter(...)` call | new cell + `notebook:244-260` | `pc.min`/`pc.max` over `density_table_with_index(p6_row, cn_order, cn_order).table["log_cCREs"]`, once per biosample | ~8 | Low. Reuses an existing function with different args; no `cnv.py` change |
| **M6c** *(optional, not recommended as default)* | Discretize `log_cCREs` into fixed bins and give panel 5 a categorical legend like panel 6's | `cnv.py` (new binning step near `:730-757`) + **new** `lib/src/epifolio/assets/conf/density_bin_color_map.json` (mirrors `copy_number_color_map.json`'s `{label: hex}` shape) + notebook `:244-260` | Bin edges (e.g. quantiles of the same unfiltered `log_cCREs` column) computed once per biosample — no edge set is correct for every biosample without checking its distribution first | ~35 | Medium. This is the one item here that adds a new asset file and new binning logic; do only if a fixed discrete legend is actually wanted over a stable continuous one |

**Recommendation: M6b, not M6c.** It fixes the actual complaint (scale drifts across interactions) with no new
file, no new binning logic, and no risk of most points collapsing into one bucket under a badly-chosen edge set.

### 8.4 Titles — a layout addition, not a jscatter feature

Checked the jscatter source directly: `Scatter` has no `title` kwarg or attribute, only per-channel legend
headers (`color_title`, which is what prints "Color" atop the current legend). Panels 1/2's titles come from
`genome_layout()` (`cnv.py:383`) via Plotly's own `fig.update_layout(title=...)` (`cnv.py:473-477`,
`:512-516`) — a `go.Figure`-only mechanism panels 5/6 don't have, since they're raw `jscatter` widgets
(`:267`, `:299`). A title has to be composed at the marimo layout level instead.

`notebooks/cnv_ccre_explorer.py` has **zero** existing `mo.md()` calls — this is a new layout pattern for this
file, not an extension of one already in use. Land it minimally:

```python
# panel-5 cell (:244-267) and panel-6 cell (:272-301) — one f-string each,
# same "N · description — {biosample}" convention as panel_1/panel_2
title5 = f"5 · Aggregated cCRE density, cancer vs. healthy z-score — {biosample}"
title6 = f"6 · Per-copy-number z-score scatter — {biosample}"
# ... add title5 / title6 to each cell's return tuple

# layout cell (:415-439) — thread the two new names through the arg list,
# replace the bare widgets in the hstack:
mo.hstack(
    [mo.vstack([mo.md(f"**{title5}**"), panel5_widget]),
     mo.vstack([mo.md(f"**{title6}**"), panel6_widget])],
    widths="equal", gap=1.5,
)
```

| ID | Action | Where | Calculation | LOC | Risk |
|---|---|---|---|---|---|
| **M6d** | Build `title5`/`title6` strings in the panel-5/6 cells; wrap each widget in `mo.vstack([mo.md(...), widget])` at the layout cell | notebook `:244-267`, `:272-301`, `:415-439` | None — string formatting only, reuses `biosample` already in scope | ~10 | Low. Mechanical: marimo wires cells by variable name, so both the panel-5/6 cells' `return` tuples **and** the layout cell's `def _(...)` argument list need the two new names added, or the cell fails to resolve them |

### 8.5 Summary: what's calculated where, and where dicts live

| Item | Calculation | Computed from | New file? |
|---|---|---|---|
| Panel 6 color dictionary | none (lookup only) — the 6-row table in §8.1 | `lib/src/epifolio/assets/conf/copy_number_color_map.json` (existing) | No |
| Panel 6 guard (M6e) | none (set comparison) | `color_map6`'s keys vs. `lvl` in `cn_table()` | No |
| Panel 6 str/label fix (M6a) | none (dict lookup with fallback) | same JSON, resolved through both `str(level)` and `cn_label(level)` | No |
| Panel 5 fixed domain (M6b) | `min`/`max` of `log_cCREs` | `density_table_with_index(p6_row, cn_order, cn_order)`, unfiltered, once per biosample | No |
| Panel 5 discrete bins (M6c, optional) | bin edges over `log_cCREs` | same unfiltered table, once per biosample | Yes — `density_bin_color_map.json` |
| Panel 5/6 titles | none (string format) | `biosample`, already in scope in both cells | No |

Five of the six land with **zero new files** — the color source of truth stays the one JSON already in the
repo, and the two calculated quantities (panel-5 domain, and bin edges if M6c is ever pursued) both reduce to
"call the existing density function with the full `cn_order`, once, outside the `active_cn` dependency chain."
