# kpm-tools

=======

# KPM Tools

[![PyPI](https://img.shields.io/pypi/v/kpm-tools.svg)][pypi_]
[![Status](https://img.shields.io/pypi/status/kpm-tools.svg)][status]
[![Python Version](https://img.shields.io/pypi/pyversions/kpm-tools)][python version]
[![License](https://img.shields.io/pypi/l/kpm-tools)][license]

[![Read the documentation at https://kpm-tools.readthedocs.io/](https://img.shields.io/readthedocs/kpm-tools/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/piskunow/kpm-tools/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/piskunow/kpm-tools/branch/main/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi_]: https://pypi.org/project/kpm-tools/
[status]: https://pypi.org/project/kpm-tools/
[python version]: https://pypi.org/project/kpm-tools
[read the docs]: https://kpm-tools.readthedocs.io/
[tests]: https://github.com/piskunow/kpm-tools/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/piskunow/kpm-tools
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

## Features

- KPM expansion of typical spectal functions
  - Density of states
  - Green's functions
  - Kubo conductivity
  - Chern marker
- Time evolution
- Tools for system with periodic boundaries
  - KPM vector factories that produces tiles
  - Velocity operators adapted to periodic boundaries
  - Distance operators adapted to periodic boundaries

## Requirements

- Python >=3.9

## Installation

You can install _KPM Tools_ via [pip] from [PyPI]:

```console
$ pip install kpm-tools
```

## Python API

Please see the [python api reference] for details.

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [BSD 2-Clause license][license],
_KPM Tools_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [@cjolowicz]'s [Hypermodern Python Cookiecutter] template.

[@cjolowicz]: https://github.com/cjolowicz
[pypi]: https://pypi.org/
[hypermodern python cookiecutter]: https://github.com/cjolowicz/cookiecutter-hypermodern-python
[file an issue]: https://github.com/piskunow/kpm-tools/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/piskunow/kpm-tools/blob/main/LICENSE
[contributor guide]: https://github.com/piskunow/kpm-tools/blob/main/CONTRIBUTING.md
[python api reference]: https://kpm-tools.readthedocs.io/en/latest/api.html
