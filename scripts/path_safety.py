from __future__ import annotations

from pathlib import Path


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def assert_not_in_known_raw_data_path(path: str | Path, what: str = "output") -> Path:
    candidate = _resolve(path)
    names = set(candidate.parts)
    raw_markers = {"Aorta", "ValidationSet", "TrainingSet"}
    if any(part.startswith("ChallengeData") for part in candidate.parts) or names.intersection(raw_markers):
        raise ValueError(
            f"Refusing to write {what} under a path that looks like raw challenge data: {candidate}\n"
            "Choose a workspace, scratch, or experiment output directory instead."
        )
    return candidate


def infer_protected_data_roots(paths: list[str | Path] | tuple[str | Path, ...]) -> list[Path]:
    """
    Infer read-only raw-data roots from input data paths.

    For 4D flow ChallengeData paths, protect the Aorta folder if present;
    otherwise protect the nearest ChallengeData ancestor. As a conservative
    fallback, protect each input directory or each input file's parent.
    """
    roots: list[Path] = []
    for item in paths:
        p = _resolve(item)
        base = p if p.suffix == "" else p.parent

        protected = None
        for ancestor in [base, *base.parents]:
            if ancestor.name == "Aorta":
                protected = ancestor
                break
        if protected is None:
            for ancestor in [base, *base.parents]:
                if ancestor.name.startswith("ChallengeData"):
                    protected = ancestor
                    break
        roots.append(protected or base)

    unique: list[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


def assert_not_within_roots(path: str | Path, protected_roots: list[str | Path] | tuple[str | Path, ...], what: str = "output") -> Path:
    candidate = _resolve(path)
    assert_not_in_known_raw_data_path(candidate, what=what)
    for root in protected_roots:
        protected = _resolve(root)
        if candidate == protected or _is_relative_to(candidate, protected):
            raise ValueError(
                f"Refusing to write {what} inside raw data path: {candidate}\n"
                f"Protected raw data root: {protected}\n"
                "Choose a workspace, scratch, or experiment output directory instead."
            )
    return candidate


def assert_outputs_not_in_data(
    output_paths: list[str | Path] | tuple[str | Path, ...],
    input_data_paths: list[str | Path] | tuple[str | Path, ...],
) -> list[Path]:
    protected_roots = infer_protected_data_roots(input_data_paths)
    return [assert_not_within_roots(path, protected_roots, what="output path") for path in output_paths]
