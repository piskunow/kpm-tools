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

KPM Tools is an open-source Python package that extends the Kernel Polynomial Method (KPM) functionalities of [Kwant](https://kwant-project.org/), a popular software for quantum transport simulations in tight-binding models. Our package specifically enhances the KPM expansion capabilities within the realm of closed tight-binding systems.

## Features

- Advanced KPM expansion of typical spectral functions like Density of States, Green's Functions, Kubo Conductivity, and Chern Markers.
- Extremely efficient time evolution operator expansion
- Additional functionalities like KPM vector factories producing tiles, and velocity and distance operators adapted to periodic boundaries.

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

Contributions, especially in documentation and the 'concatenator' function/module, are very welcome. For more information, see our [Contributor Guide].

## License

Distributed under the terms of the [BSD 2-Clause license][license],
_KPM Tools_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

Many of the ideas implemented in KPM Tools originate from the early work on KPM expansion on the paper

_Computation of topological phase diagram of disordered PbSnTe using the kernel polynomial method_.

[Phys. Rev. Research 2, 013229 (2020)](https://doi.org/10.48550/arXiv.1905.02215)
[arXiv:1905:02215](https://arxiv.org/abs/1905.02215)

Consider citing that work if you use this package on a publication.

### Acknowledgments to Kwant

KPM Tools is built upon the robust and efficient foundation provided by Kwant. We extend our gratitude to the Kwant authors and contributors for their work in developing a versatile platform for quantum transport simulations. KPM Tools aims to complement Kwant's capabilities in KPM expansions, adhering to the high standards of quality and performance set by the Kwant project.

### Project Template

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
