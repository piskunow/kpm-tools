"""Sphinx configuration."""

import inspect
import os
import sys

from sphinx.util import inspect as sphinx_inspect


project = "KPM Tools"
author = "Pablo Piskunow"
copyright = "2023, Pablo Piskunow"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "sphinx_click",
    "myst_parser",
    "nbsphinx",
]
nbsphinx_allow_errors = True
autodoc_typehints = "description"
html_theme = "furo"
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
nbsphinx_prolog = r"""
.. raw:: html

    <div class="admonition tip">
        <p class="admonition-title">Tip</p>
        <p>Run this Jupyter Notebook locally:
        <a id="downloadNotebookLink" href="javascript:void(0);" class="download-notebook-button" target="_blank" download>
            Download Notebook
        </a>
        </p>
    </div>

    <script>
    document.addEventListener('DOMContentLoaded', function() {
        var currentPath = window.location.pathname;
        var notebookName = currentPath.split('/').pop().replace('.html', '.ipynb');
        var downloadLink = "{{ readthedocs_download_url | e }}" + notebookName;
        document.getElementById('downloadNotebookLink').href = downloadLink;
    });
    </script>


  """  # noqa: B950


def linkcode_resolve(domain, info):
    """Resolve links to source."""
    if domain != "py":
        return None
    if not info["module"]:
        return None

    # Replace with your project's GitHub repository URL
    github_repo = "https://github.com/piskunow/kpm-tools"

    # Get the module object
    module = sys.modules.get(info["module"])
    if module is None:
        return None

    # avoid math directive to be parsed by this extension.
    if "kpm_generator" in info["module"].split("."):
        return None

    # Get the source file path of the module
    filename = sphinx_inspect.getsourcefile(module)
    if filename is None:
        return None

    # Trim the filename to a path relative to the project root
    rel_fn = os.path.relpath(filename, start=os.path.dirname(__file__))

    # Get the line number of the object within the module
    obj = module
    for part in info["fullname"].split("."):
        obj = getattr(obj, part, None)

    if obj is None:
        return None

    try:
        lines, _ = inspect.getsourcelines(obj)
    except Exception:
        return None

    line = inspect.getsourcelines(obj)[1]
    return f"{github_repo}/blob/main/{rel_fn}#L{line}"
