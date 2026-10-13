"""One command dispatcher; numerical dependencies load only for selected runs."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from importlib import import_module

from .registry import COMMANDS


def main(argv: Sequence[str] | None = None) -> int:
    """Run a named experiment without modifying the caller's argument list."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Config-driven manta LMPC and heterogeneous harbor experiments.",
        usage="%(prog)s [command] [options]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Commands:\n"
        + "\n".join(
            f"  {name:<40} {command.description}" for name, command in COMMANDS.items()
        )
        + "\n\nUse COMMAND --help for its options. With no command, runs manta.",
    )
    if arguments and arguments[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    if not arguments or arguments[0].startswith("-"):
        command_name = "manta"
    else:
        command_name = arguments.pop(0)
    command = COMMANDS.get(command_name)
    if command is None:
        parser.error(f"unknown command {command_name!r}; use --help to list commands")
    experiment = import_module(command.module)
    experiment.main([*arguments, *command.preset])
    return 0
