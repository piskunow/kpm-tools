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
