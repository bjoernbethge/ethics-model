# PyPI Publishing Setup for ethics-model

This document explains how to configure PyPI publishing for the ethics-model package using GitHub Actions and trusted publishing.

## Prerequisites

1. A PyPI account at https://pypi.org
2. Admin access to the GitHub repository

## Setup Instructions

### 1. Configure PyPI Trusted Publishing

PyPI's trusted publishing eliminates the need for API tokens by using OpenID Connect (OIDC) to verify that the package is being published from the correct GitHub repository.

1. Go to https://pypi.org and log in
2. Navigate to your account settings
3. Go to "Publishing" section
4. Click "Add a new pending publisher"
5. Fill in the details:
   - **PyPI Project Name**: `ethics-model`
   - **Owner**: `bjoernbethge`
   - **Repository name**: `ethics-model`
   - **Workflow name**: `publish-pypi.yml`
   - **Environment name**: (leave empty)
6. Click "Add"

**Note**: For the first publish, you need to create the project as a "pending publisher" before the package exists on PyPI.

### 2. Create a Release

Once trusted publishing is configured, you can publish to PyPI by creating a GitHub release:

1. Go to https://github.com/bjoernbethge/ethics-model/releases/new
2. Click "Choose a tag" and create a new tag (e.g., `v0.1.0`)
3. Set the release title (e.g., "ethics-model v0.1.0")
4. Add release notes describing the changes
5. Click "Publish release"

The GitHub Action will automatically:
- Build the package
- Run quality checks
- Publish to PyPI using trusted publishing

### 3. Manual Publishing (Optional)

You can also trigger publishing manually without creating a release:

1. Go to https://github.com/bjoernbethge/ethics-model/actions/workflows/publish-pypi.yml
2. Click "Run workflow"
3. Select the branch to publish from
4. Click "Run workflow"

## Verifying the Package

After publishing, verify your package at:
- https://pypi.org/project/ethics-model/

Install it using:
```bash
pip install ethics-model
```

## Troubleshooting

### First Publish Fails

If the first publish fails with "project does not exist", make sure you:
1. Created the pending publisher on PyPI first
2. Used the exact workflow filename (`publish-pypi.yml`)
3. The repository owner and name match exactly

### Permission Denied

If you get permission errors:
1. Verify the trusted publisher is configured correctly on PyPI
2. Ensure the workflow has `id-token: write` permissions
3. Check that the repository owner matches the PyPI project owner

## Version Management

The package version is defined in `pyproject.toml`:

```toml
[project]
name = "ethics-model"
version = "0.1.0"
```

Before creating a new release:
1. Update the version in `pyproject.toml`
2. Commit the change
3. Create a new release with a matching tag (e.g., `v0.1.0`)

## Security

Trusted publishing is more secure than API tokens because:
- No long-lived credentials stored in GitHub secrets
- Automatic verification of publisher identity
- Per-repository and per-workflow restrictions
- Automatic token rotation

## Additional Resources

- [PyPI Trusted Publishing Guide](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions: Publishing Python Packages](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python#publishing-to-package-registries)
