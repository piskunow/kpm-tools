"""Sphinx configuration."""
project = "KPM Tools"
author = "Pablo Piskunow"
copyright = "2023, Pablo Piskunow"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
    "nbsphinx",
]
nbsphinx_allow_errors = True
autodoc_typehints = "description"
html_theme = "furo"
