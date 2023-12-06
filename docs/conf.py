"""Sphinx configuration."""

import inspect
import os
import sys


project = "KPM Tools"
author = "Pablo Piskunow"
copyright = "2023, Pablo Piskunow"
autodoc_mock_imports = ["kwant"]
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


def get_object_line_number(info):
    """Return object line number from module."""
    try:
        module = sys.modules.get(info["module"])
        if module is None:
            return None

        # walk through the nested module structure
        obj = module
        for part in info["fullname"].split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return None

        return inspect.getsourcelines(obj)[1]
    except (TypeError, OSError):
        return None


def linkcode_resolve(domain, info):
    """Point to the source code repository, file and line number."""
    # only add links to python modules
    if domain != "py":
        return None
    if not info["module"]:
        return None

    filename = "src/" + info["module"].replace(".", "/")

    # point to the right repository, branch, file and line
    github_repo = "https://github.com/piskunow/kpm-tools"

    line = get_object_line_number(info)
    if line is None:
        return None

    # Determine the branch based on RTD version
    rtd_version = os.getenv("READTHEDOCS_VERSION", "latest")
    github_branch = "develop" if rtd_version == "develop" else "main"

    return f"{github_repo}/blob/{github_branch}/{filename}.py#L{line}"
