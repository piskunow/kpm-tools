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


def linkcode_resolve(domain, info):
    """Resolve and link to source."""
    if domain != "py":
        return None
    if not info["module"]:
        return None

    # Determine the branch based on RTD version
    rtd_version = os.getenv("READTHEDOCS_VERSION", "latest")
    github_branch = "develop" if rtd_version == "latest" else "main"

    github_repo = "https://github.com/piskunow/kpm-tools"

    module = sys.modules.get(info["module"])
    if module is None:
        return None

    filename = inspect.getsourcefile(module)
    if filename is None:
        return None

    # Adjust the file path for the repository structure
    package_dir = "src/kpm_tools"
    rel_fn = os.path.relpath(filename, start=os.path.dirname(package_dir))

    obj = module
    for part in info["fullname"].split("."):
        obj = getattr(obj, part, None)

    if obj is None:
        return None

    try:
        line = inspect.getsourcelines(obj)[1]
        return f"{github_repo}/blob/{github_branch}/{rel_fn}#L{line}"
    except Exception:
        return None
