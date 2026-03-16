# Changelog

## Unreleased

### Changed
- EXESS now uses `fetch_outputs()` for in-memory results and `save_outputs()` for workspace persistence, with shared output handling between the two paths
- Added `fetch_object()` as the in-memory object-store helper and removed `download_object()`
- `fetch_object()` and `save_object()` now share archive extraction logic
- Updated EXESS docs, examples, and tests to use the new fetch/save naming

## 6.10.2

### Fixed
- Don't print message about run being restored if the run is canceled or failed: this usually means that the module instance couldn't be started at all because the account tier doesn't support running that module instance. The other potential cause is if a module isn't available for a particular target and the user tries to use that combination.
- Print and store `trace` field from a run properly when it's either canceled or failed

## 6.10.1

### Fixed
- Make `exess.interaction_energy` a thin wrapper around `exess.energy`, homogenizing the interfaces
- `FragKeywords` stores and allows setting `reference_fragment` directly, better conforming to upstream config design
- CLAUDE.md updated for devs using the nix flake
- Clean up .gitignore glitch

## 6.10.0

### Removed
- remove ML mentions since it's disabled (for now) in upstream EXESS

### Changed
- **Breaking:** Default level of theory changed from RestrictedHF to RestrictedKSDFT with B3LYP/cc-pVDZ across all EXESS functions (`exess`, `energy`, `interaction_energy`, `qmmm`, `optimization`)
- Default basis set for `qmmm` changed from STO-3G to cc-pVDZ

### Added
- nnxtb docs, tutorial, and example script
- `ksdft_keywords` parameter to `interaction_energy`, `qmmm`, and `optimization` functions

### Fixed
- Update `ks` keyword to `ks_dft` in `qmmm` and `optimization` REX DSL templates
- Add force push to docs deploy workflow to avoid out of sync issues (don't care about maintaining history on bot managed branch)

## 6.9.0

### Removed
- `exess.chelpg()` function — use `exess.energy()` with `ExportKeywords(export_chelpg_charges=True)` instead

### Changed
- CHELPG example and tutorial updated to use `exess.energy()` with export keywords and JSON output
- Rename docs project to "QDX" and set html\_title to "QDX Documentation"

## 6.8.0

### Changed
- Bump `auto3d_rex` module revision (staging and prod)

### Added
- `Auto3DResult` and `Auto3DStats` dataclasses for structured conformer output
- `save_outputs()` in `auto3d` module with bespoke dataclasses for the result TRCs and stats; TRCs downloaded and constructed fully in-memory

### Fixed
- Disambiguate `prepare_protein` error message in `save_outputs()` fallback

## 6.7.3

### Changed
- Update docs style to match QDX website: dark mode, logo linking to qdx.co, nav links to EXESS and Rush
- Add QDX logo to docs sidebar
- Add "Home" link above global toc in docs sidebar
- Simplify docs templates: remove EXESS-specific header, breadcrumbs, sidebar, and toc overrides
- Remove custom EXESS nav builder from docs conf.py
- Update EXESS docs: free version access, free for academics, remove email contact in favor of request forms
- Expand EXESS limitations documentation (relativistic, heavy elements, excited-state methods)
- Remove x2c-SVPall from basis set table (relativistic not currently supported)

### Fixed
- Center 2D charge map molecule in CHELPG HTML visualization

## 6.7.2

### Changed
- Docs deploy now targets qdx-main-landing repo instead of exess-webapp

## 6.7.1

### Fixed
- fix github action for docs publishing

## 6.7.0

### Changed
- `.env` file detection now walks up parent directories from `cwd`, so examples subfolders find the repo-root `.env` automatically

### Added
- Add tags to rex runs with runtime and SDK metadata
- Workflow to auto-deploy docs to gh-pages and exess-webapp when merged to main
- CI changelog check in lint workflow — PRs must update `CHANGELOG.md`
- Script to run all examples for easier testing
- Charge-colored 2D aspirin structure to CHELPG output
- Docs: basis set warnings with example code links across tutorials

### Fixed
- CHELPG tutorial code and output visualisation cleanup
- Bar chart and 3D structure colors now match 2D visualization in CHELPG output
- Changelog CI check was only checking the latest commit

## 6.6.0

### Changed
- `from_json()` now always returns a list in the case of a single path input, which is more consistent
  especially given the typing challenges with determining single-vs-many output types for file inputs

### Added
- test for getting CHELPG charges from exess.energy (exess.chelpg to be removed soon)
- add all deps needed for examples to pyproject.toml dev deps

### Fixed
- work around json import error due to shadowing
- gradually type, and fix incomplete or erroneous typing, in numerous places

## 6.5.1

### Fixed
- `save_energy_outputs()` now handles list inputs from `collect_run()`
- `save_energy_outputs()` now includes missing return statement for HDF5 output handling
- `save_object()` conditional write logic that prevented file extraction for single-file tar archives
- `save_object()` now skips directories in tar archives and finds actual files to extract
- `save_energy_outputs()` now gracefully handles missing HDF5 files (returns tuple with None)
- Added error handling for unknown output formats in `save_energy_outputs()`
- All example scripts now use UTF-8 encoding for file operations (Windows compatibility)
- Fixed temp file permissions issue in `prepare_protein()` (Windows compatibility)
- Fixed `save_outputs()` to properly handle list/tuple results from `collect_run()`
- Fixed `prepare_complex()` to respect the `collect` parameter when calling `run_prepare_protein()`
- Updated CHELPG example script to use `save_energy_outputs()` for proper HDF5 extraction

## 6.5.0

### Changed
- Updated EXESS module paths to latest versions (staging: `exess_rex`, `exess_qmmm_rex`; prod: `exess_rex`)

### Added
- PyPI publish workflow using OIDC trusted publishing, triggered on `v*` tags
- CI test timeouts with queue-aware slow test skipping — slow tests auto-skip when Rush queues are busy
- `pytest-timeout` dependency with per-test timeouts (300s default, 600s for slow tests)
- `--run-slow-force` pytest option to bypass queue check
- `run_tests.sh` argument handling (`--quick`, `--slow`, `--all`)
