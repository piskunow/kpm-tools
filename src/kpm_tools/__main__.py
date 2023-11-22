"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """KPM Tools."""


if __name__ == "__main__":
    main(prog_name="kpm-tools")  # pragma: no cover
