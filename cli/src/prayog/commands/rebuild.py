"""Focused entry point for reproducing a previous project run."""
import argparse

from prayog.commands.run import run as submit_run
from prayog.context import Context


def setup_rebuild(p):
    p.description = (
        "Reproduce a finished run using its original version, entrypoint, hyperparameters, "
        "and dataset fingerprints. Supply the same dataset files with --data; the server "
        "rejects a rebuild if their fingerprints differ. Compare the result with "
        "prayog project compare after it finishes."
    )
    p.add_argument("name", help="the project's name")
    p.add_argument("run", metavar="VERSION/N", help="the original run, such as A/1")
    p.add_argument("--gpus", type=int, default=1, help="number of GPUs on one node (default: 1)")
    p.add_argument("--walltime", required=True, metavar="HH:MM",
                   help="the longest the rebuild may take, such as 04:00")
    p.add_argument("--data", action="append", default=[], metavar="PATH",
                   help="the original dataset file or folder (repeatable; fingerprints must match)")
    p.set_defaults(func=rebuild)


def rebuild(ctx: Context, args) -> None:
    # Share the existing submission path so data checks, retries, and server validation
    # cannot drift from the established run --rebuild implementation.
    submit_run(ctx, argparse.Namespace(
        name=args.name,
        gpus=args.gpus,
        walltime=args.walltime,
        version=None,
        data=args.data,
        entrypoint=None,
        hp=[],
        rebuild=args.run,
    ))
