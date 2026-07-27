# Changelog

All notable changes are documented here. Releases follow semantic versioning.

## [0.2.0] - 2026-07-26

### Added

- Token-boundary-safe completion-only masking with dependency-light tests.
- GPU-free `--dry-run` validation for configs, datasets, and phase splits.
- Reproducible GSM8K result records with seeds, data fingerprints, environment
  versions, hardware, and commit provenance.
- Python 3.10–3.13 CI, linting, package checks, release attestations, and signed
  distribution-channel provenance.
- Contribution, security, citation, and structured issue-reporting guidance.

### Changed

- The public GSM8K comparison was rerun under the completion-only objective.
- Custom stratified DataLoaders are now prepared by Accelerate and account for
  distributed world size when constructing effective batches.
- Configuration validation now rejects misspelled phase options before model loading.
- The package metadata uses the SPDX `MIT` expression and bounded ML-stack compatibility
  ranges.

## [0.1.1] - 2026-07-11

- Added assistant-completion-only training labels.
- Fixed nested-brace handling in boxed answers.
- Improved compatibility across Transformers and TRL versions.

## [0.1.0] - 2026-06-23

- Initial library release extracted from the silver-medal competition solution.

[0.2.0]: https://github.com/DaoyuanLi2816/tracedistill/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/DaoyuanLi2816/tracedistill/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/DaoyuanLi2816/tracedistill/releases/tag/v0.1.0
