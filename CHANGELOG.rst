Changelog
=========

This changelog follows the great advice from https://keepachangelog.com/.

Each section will have a title of the format ``X.Y.Z (YYYY-MM-DD)`` giving the version of the package and the date of release of that version. Unreleased changes i.e. those that have been merged into `main` (e.g. with a .dev suffix) but which are not yet in a new release (on pypi) are added to the changelog but with the title ``X.Y.Z (unreleased)``. Unreleased sections can be combined when they are released and the date of release added to the title.

Subsections for each version can be one of the following;

- ``Added`` for new features.
- ``Changed`` for changes in existing functionality.
- ``Deprecated`` for soon-to-be removed features.
- ``Removed`` for now removed features.
- ``Fixed`` for any bug fixes.
- ``Security`` in case of vulnerabilities.

Each individual change should have a link to the pull request after the description of the change.

1.0.1 (unreleased)
------------------

Changed
^^^^^^^
- Reactivated CI and added PR approval guardrails (https://github.com/azukds/model_interpreter/pull/8)
- Updated CI configuration to follow tubular's python-package.yml (https://github.com/azukds/model_interpreter/pull/8)
- Switched to prek for pre-commit checks (https://github.com/azukds/model_interpreter/pull/8)
- Fixed spelling mistakes and removed stale tubular references (https://github.com/azukds/model_interpreter/pull/8)
- Added example doctests to interpreter.py (https://github.com/azukds/model_interpreter/pull/10)

1.0.0 (2024-08-06)
-------------------

Added
^^^^^
- Moved config to pyproject.toml approach
- Updated envs and impacted logic (e.g to accommodate new shap behaviour)
- Setup devcontainer config for codespaces
- Updated build pipeline

1.0.0 (2024-08-06)
-------------------

Added
^^^^^
- Open source release of the package on Github
