"""Nox sessions."""

import os
import shlex
import shutil
import sys
from pathlib import Path
from textwrap import dedent

import nox

try:
    from nox_poetry import Session
    from nox_poetry import session
except ImportError:
    message = f"""\
    Nox failed to import the 'nox-poetry' package.

    Please install it using the following command:

    {sys.executable} -m pip install nox-poetry"""
    raise SystemExit(dedent(message)) from None


package = "kpm_tools"
python_versions = ["3.12", "3.11"]
# Python version ReadTheDocs builds with, per .readthedocs.yml. Kept in sync so
# the docs-requirements session validates docs/requirements.txt on the same
# interpreter RTD will actually use.
rtd_python_version = "3.12"
nox.needs_version = ">= 2021.6.6"
nox.options.sessions = (
    "pre-commit",
    "pip-audit",
    "mypy",
    "tests",
    "typeguard",
    "xdoctest",
    "docs-requirements",
    "docs-build",
)


def activate_virtualenv_in_precommit_hooks(session: Session) -> None:
    """Activate virtualenv in hooks installed by pre-commit.

    This function patches git hooks installed by pre-commit to activate the
    session's virtual environment. This allows pre-commit to locate hooks in
    that environment when invoked from git.

    Args:
        session: The Session object.
    """
    assert session.bin is not None  # noqa: S101

    # Only patch hooks containing a reference to this session's bindir. Support
    # quoting rules for Python and bash, but strip the outermost quotes so we
    # can detect paths within the bindir, like <bindir>/python.
    bindirs = [
        bindir[1:-1] if bindir[0] in "'\"" else bindir
        for bindir in (repr(session.bin), shlex.quote(session.bin))
    ]

    virtualenv = session.env.get("VIRTUAL_ENV")
    if virtualenv is None:
        return

    headers = {
        # pre-commit < 2.16.0
        "python": f"""\
            import os
            os.environ["VIRTUAL_ENV"] = {virtualenv!r}
            os.environ["PATH"] = os.pathsep.join((
                {session.bin!r},
                os.environ.get("PATH", ""),
            ))
            """,
        # pre-commit >= 2.16.0
        "bash": f"""\
            VIRTUAL_ENV={shlex.quote(virtualenv)}
            PATH={shlex.quote(session.bin)}"{os.pathsep}$PATH"
            """,
        # pre-commit >= 2.17.0 on Windows forces sh shebang
        "/bin/sh": f"""\
            VIRTUAL_ENV={shlex.quote(virtualenv)}
            PATH={shlex.quote(session.bin)}"{os.pathsep}$PATH"
            """,
    }

    hookdir = Path(".git") / "hooks"
    if not hookdir.is_dir():
        return

    for hook in hookdir.iterdir():
        if hook.name.endswith(".sample") or not hook.is_file():
            continue

        if not hook.read_bytes().startswith(b"#!"):
            continue

        text = hook.read_text()

        if not any(
            Path("A") == Path("a") and bindir.lower() in text.lower() or bindir in text
            for bindir in bindirs
        ):
            continue

        lines = text.splitlines()

        for executable, header in headers.items():
            if executable in lines[0].lower():
                lines.insert(1, dedent(header))
                hook.write_text("\n".join(lines))
                break


def install_with_kwant(session: Session) -> None:
    """Install this package together with kwant.

    kwant ships as an sdist only, so this compiles it and needs a C/C++
    toolchain. It cannot go through the ``kwant`` extra: kwant's legacy
    setup.py imports numpy to locate its headers, and PEP 517 build isolation
    runs that build in a clean environment without numpy, so the build dies
    with "NumPy header directory cannot be determined". Installing this package
    first puts numpy in the environment, and ``--no-build-isolation`` lets
    kwant's build see it. setuptools/wheel are explicit because isolation is
    what would normally have supplied them, and 3.12+ venvs omit setuptools.

    The previous ``build_kwant`` session cloned kwant and ran
    ``python setup.py install``, which modern setuptools no longer supports.

    Args:
        session: The Session object.
    """
    session.install(".")
    session.install("setuptools", "wheel")
    session.install("--no-build-isolation", "kwant")


@session(name="pre-commit", python=python_versions[0])
def precommit(session: Session) -> None:
    """Lint using pre-commit."""
    args = session.posargs or [
        "run",
        "--all-files",
        "--hook-stage=manual",
        "--show-diff-on-failure",
    ]
    session.install(
        "black",
        "darglint",
        "flake8",
        "flake8-bandit",
        "flake8-bugbear",
        "flake8-docstrings",
        "flake8-rst-docstrings",
        "isort",
        "pep8-naming",
        "pre-commit",
        "pre-commit-hooks",
        "pyupgrade",
    )

    session.run("pre-commit", *args)
    if args and args[0] == "install":
        activate_virtualenv_in_precommit_hooks(session)


@session(name="pip-audit", python=python_versions[0])
def pip_audit(session: Session) -> None:
    """Scan dependencies for known vulnerabilities.

    Replaces the old ``safety`` session: safety 3.x requires an account and an
    API key, which is unworkable in CI for a volunteer-maintained project.

    Args:
        session: The Session object.
    """
    requirements = session.poetry.export_requirements()
    session.install("pip-audit")
    session.run("pip-audit", f"--requirement={requirements}", "--strict")


@session(name="docs-requirements", python=rtd_python_version)
def docs_requirements(session: Session) -> None:
    """Check docs/requirements.txt resolves on the interpreter RTD uses.

    Nothing else installs docs/requirements.txt, so a pin that needs a newer
    Python than .readthedocs.yml provides passes every other check and only
    breaks the ReadTheDocs build after merge. This resolves it without
    installing anything.

    Args:
        session: The Session object.
    """
    session.run(
        "python",
        "-m",
        "pip",
        "install",
        "--dry-run",
        "--ignore-installed",
        "--report",
        os.devnull,
        "-r",
        "docs/requirements.txt",
    )


@session(python=python_versions)
def mypy(session: Session) -> None:
    """Type-check using mypy."""
    args = session.posargs or ["src", "tests", "docs/conf.py"]
    install_with_kwant(session)
    session.install("mypy", "pytest")
    session.run("mypy", *args)
    if not session.posargs:
        session.run("mypy", f"--python-executable={sys.executable}", "noxfile.py")


@session(python=python_versions)
def tests(session: Session) -> None:
    """Run the test suite."""
    install_with_kwant(session)
    session.install("coverage[toml]", "pytest", "pygments")

    try:
        session.run("coverage", "run", "--parallel", "-m", "pytest", *session.posargs)
    finally:
        if session.interactive:
            session.notify("coverage", posargs=[])


@session(python=python_versions[0])
def coverage(session: Session) -> None:
    """Produce the coverage report."""
    args = session.posargs or ["report"]

    session.install("coverage[toml]")

    if not session.posargs and any(Path().glob(".coverage.*")):
        session.run("coverage", "combine")

    session.run("coverage", *args)


@session(python=python_versions[0])
def typeguard(session: Session) -> None:
    """Runtime type checking using Typeguard."""
    install_with_kwant(session)
    session.install("pytest", "typeguard", "pygments")

    session.run("pytest", f"--typeguard-packages={package}", *session.posargs)


@session(python=python_versions)
def xdoctest(session: Session) -> None:
    """Run examples with xdoctest."""
    if session.posargs:
        args = [package, *session.posargs]
    else:
        args = [f"--modname={package}", "--command=all"]
        if "FORCE_COLOR" in os.environ:
            args.append("--colored=1")

    install_with_kwant(session)

    session.install("xdoctest[colors]")
    session.run("python", "-m", "xdoctest", *args)


# The docs build deliberately does NOT install kwant, mirroring
# .readthedocs.yml: docs/conf.py sets autodoc_mock_imports = ["kwant"], and
# every tutorial notebook ships with stored outputs, so nbsphinx leaves them
# unexecuted. Compiling kwant here would only make the build slower and more
# fragile. nbsphinx does need the `pandoc` *binary* on PATH (the PyPI `pandoc`
# package is only a wrapper); on ReadTheDocs that comes from apt_packages.
DOCS_DEPS = (
    "sphinx",
    "sphinx-click",
    "nbsphinx",
    "ipykernel",
    "furo",
    "myst-parser",
)


@session(name="docs-build", python=python_versions[0])
def docs_build(session: Session) -> None:
    """Build the documentation."""
    args = session.posargs or ["docs", "docs/_build"]
    if not session.posargs and "FORCE_COLOR" in os.environ:
        args.insert(0, "--color")

    session.install(".")
    session.install(*DOCS_DEPS)

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-build", *args)


@session(python=python_versions[0])
def docs(session: Session) -> None:
    """Build and serve the documentation with live reloading on file changes."""
    args = session.posargs or ["--open-browser", "docs", "docs/_build"]

    session.install(".")
    session.install("sphinx-autobuild", *DOCS_DEPS)

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-autobuild", *args)
