#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import inspect
import tempfile
from pathlib import Path


TEST_FILES = (
    Path("tests/test_joint_4dflow_encodings.py"),
    Path("tests/test_flowvn_3d_backbone.py"),
)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load test module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    passed = 0
    for test_file in TEST_FILES:
        module = load_module(test_file)
        tests = sorted(
            (name, value)
            for name, value in vars(module).items()
            if name.startswith("test_") and callable(value)
        )
        for name, test in tests:
            parameters = tuple(inspect.signature(test).parameters)
            if not parameters:
                test()
            elif parameters == ("tmp_path",):
                with tempfile.TemporaryDirectory(prefix=f"{name}-") as tmp_dir:
                    test(Path(tmp_dir))
            else:
                raise RuntimeError(f"Unsupported direct-test parameters for {name}: {parameters}")
            passed += 1
            print(f"PASS {test_file.name}::{name}", flush=True)
    print(f"DIRECT_TESTS_PASSED={passed}", flush=True)


if __name__ == "__main__":
    main()
