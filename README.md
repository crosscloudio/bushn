# Bushn

[![build status](https://gitlab.crosscloud.me/CrossCloud/bushn/badges/master/build.svg)](https://gitlab.crosscloud.me/CrossCloud/bushn/commits/master)
[![coverage report](https://gitlab.crosscloud.me/CrossCloud/bushn/badges/master/coverage.svg)](https://gitlab.crosscloud.me/CrossCloud/bushn/commits/master)

    # Switch to your local virtualenv
    workon cc-bushn-py35
    # Setup everything
    python setup.py develop
    # ...
    # Make changes
    # ...
    # Run tests
    python setup.py test

## Versioning

Ensure that `bumpversion` (or install it using `pip install bumpversion`) is
installed (configuration see `setup.cfg`). Call `bumpversion` with either
`major`, `minor` or `patch`. This will increase all version numbers and automatically
create a commit and tag for the new current version.


    # Current version 1.0.0. Bump version 1.0.1
    bumpversion patch

