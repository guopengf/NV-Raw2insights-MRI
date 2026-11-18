"""Fresh-process entrypoint for comparison methods; never imports our model."""

import json

from ..common.execution import execute, parser


def main():
    args = parser().parse_args()
    cfg = json.loads(args.config.read_text())
    if cfg["paper"]["method"] in {"cg_sense", "cs_tv"}:
        from .solvers import execute_solver

        execute_solver(args)
    else:
        from .learned import factory

        execute(args, factory)


if __name__ == "__main__":
    main()
