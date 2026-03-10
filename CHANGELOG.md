# Changelog

## 6.7.2
### Changed
- Docs deploy now targets qdx-main-landing repo instead of exess-webapp

## 6.7.1
### Fixed
- fix github action for docs publishing

## 6.7.0

### Changed
- `.env` file detection now walks up parent directories from `cwd`, so examples subfolders find the repo-root `.env` automatically

### Fixed
- CHELPG tutorial code and output visualisation cleanup
- Bar chart and 3D structure colors now match 2D visualization in CHELPG output
- Changelog CI check was only checking the latest commit

### Added
- Add tags to rex runs with runtime and SDK metadata
- Workflow to auto-deploy docs to gh-pages and exess-webapp when merged to main
- CI changelog check in lint workflow — PRs must update `CHANGELOG.md`
- Script to run all examples for easier testing
- Charge-colored 2D aspirin structure to CHELPG output
- Docs: basis set warnings with example code links across tutorials

## 6.6.0

### Changed
- `from_json()` now always returns a list in the case of a single path input, which is more consistent
  especially given the typing challenges with determining single-vs-many output types for file inputs

### Fixed
- work around json import error due to shadowing
- gradually type, and fix incomplete or erroneous typing, in numerous places

### Added
- test for getting CHELPG charges from exess.energy (exess.chelpg to be removed soon)
- add all deps needed for examples to pyproject.toml dev deps

## 6.5.1

### Changed

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

### Added

## 6.5.0

### Changed
- Updated EXESS module paths to latest versions (staging: `exess_rex`, `exess_qmmm_rex`; prod: `exess_rex`)

### Added
- PyPI publish workflow using OIDC trusted publishing, triggered on `v*` tags
- CI test timeouts with queue-aware slow test skipping — slow tests auto-skip when Rush queues are busy
- `pytest-timeout` dependency with per-test timeouts (300s default, 600s for slow tests)
- `--run-slow-force` pytest option to bypass queue check
- `run_tests.sh` argument handling (`--quick`, `--slow`, `--all`)
