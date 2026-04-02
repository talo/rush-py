# Changelog

## Unreleased

### Added
- Add smol-similarity module

## 7.0.0

No new changes since 7.0.0rc3.

### Other
- Add GitHub Actions to build and deploy Sphinx docs previews for PRs via GitHub Pages

## 7.0.0rc3

### Changed
- Make EXESS `calculation`, `energy`, and `interaction_energy` functions output exports as JSON by default

### Other
- Add GitHub Action for making releases
- Clean up the rest of the GitHub Actions

## 7.0.0rc2

### Fixed
- Distinguish backend and module run failures
- Tweak pytest config based on scientific python recs and make some other small cleanups

## 7.0.0rc1

### Changed
- Split the non-module Rush implementation into session, runs, and objects modules, and clean up type annotations
- Improve docs' API reference structure, including exposing all the type declarations properly

## 7.0.0b3

### Changed
- Tests now write to a temporary workspace that doesn't clutter the git tree
- Can configure tests' workspace directory via `--rush-workspace-dir` flag to pytest
- Tests more thoroughly check for valid behavior and output
- Removed `run_tests.sh` favor of calling pytest manually (it's not too hard)

### Added
- Report walltime and SUs in `RushRunInfo`

## 7.0.0b2

### Changed
- Modules that take molecules as input are now overloaded to behave as expected for multiple types, including TRCs, tuples of the required TRC components, individual Topologies where possible, and object-store references in multiple forms
- Renamed mol types' `to_json` to `to_dict`, since it returns a dict

### Added
- Provided `RushRun` type that modules functions return to manage the run
- Modules now use `RushRun.collect()` for blocking until run completion and getting access to results as remote object-store references
- Modules now use `RushRun.fetch()` for in-memory results, and `RushRun.save()` for workspace persistence, with shared output handling between the two latter paths.
- Provide `module.ResultRef` types for references to remote module output
- Provide `module.Result` types for fetched module output
    - E.g., EXESS parsed output types are now `exess.Result`, `exess.Calculation`, and `exess.ManyBodyExpansion`, with the calculation available at `result.calc`
- Provide `module.ResultPaths` types for saved module output
    - Mirrors the structure of `ResultRef` and `Result` per module, but each field is a path to the local saved file for each field
- Provide similar `TRCRef` and `TRCPaths` classes with the same design pattern, and `TRCRef.upload()` for uploading TRCs to the Rush object store
- Provide `RushObject` class wrapping raw virtual object JSON data
- Updated docs, examples, and tests to use the new fetch/save naming

## 7.0.0b1

### Added
- Provide per- Rush module `fetch_outputs()` and `save_outputs()` functions
- Provide `fetch_object()` as the in-memory object-store helper and removed `download_object()`
- `fetch_object()` and `save_object()` now share archive extraction logic

## 6.10.2

### Fixed
- Rewrite wordy intro sentence in docs landing page for clarity
- Don't print message about run being restored if the run is canceled or failed: this usually means that the module instance couldn't be started at all because the account tier doesn't support running that module instance. The other potential cause is if a module isn't available for a particular target and the user tries to use that combination.
- Print and store `trace` field from a run properly when it's either canceled or failed
- Remove AI-generated mismatches page from docs

### Added
- Info on predicting runtime

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
