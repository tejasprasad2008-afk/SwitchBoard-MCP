# Release Checklist

Follow these steps in order to release a new version of Switchboard to PyPI.

## 1. Bump Version

Edit `pyproject.toml`:

```toml
version = "0.1.0"   # ← change this
```

Also update the version in `__init__.py` if it exists.

## 2. Update CHANGELOG.md

Add a new section at the top:

```markdown
## [0.1.1] - 2026-04-14

### Added
- New feature X

### Fixed
- Bug fix Y
```

Update the `[Unreleased]` link dates if applicable.

## 3. Run Full Test Suite

```bash
pytest tests/ -v --tb=short
python cli_test.py dry_run
python cli_test.py task_routing
```

All tests must pass. The CLI demos must work with zero API keys.

## 4. Lint

```bash
ruff check .
```

No errors or warnings.

## 5. Build the Package

```bash
python -m build
```

This creates `dist/switchboard_mcp-X.Y.Z-py3-none-any.whl` and `dist/switchboard-mcp-X.Y.Z.tar.gz`.

## 6. Check the Build

```bash
twine check dist/*
```

Must report no issues.

## 7. Upload to PyPI

```bash
twine upload dist/*
```

You'll need PyPI credentials. Use an API token:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD="pypi-YOUR_TOKEN_HERE"
twine upload dist/*
```

## 8. Tag the Git Release

```bash
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

## 9. Create GitHub Release

Go to GitHub → Releases → Draft new release:

- **Tag:** `v0.1.0`
- **Title:** `v0.1.0`
- **Body:** Copy the changelog section for this version
- Check "Set as the latest release"

## 10. Verify

```bash
pip install switchboard-mcp
switchboard --help
```

Confirm the package installs and the CLI works from PyPI.
