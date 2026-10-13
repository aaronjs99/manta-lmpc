"""Reserve one output directory without replacing unrelated saved runs."""

from contextlib import contextmanager
import os
from pathlib import Path


@contextmanager
def claim_run_directory(root: Path, *, resume_path: Path | None = None):
    """Protect a run from simultaneous writers and accidental output reuse."""
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".manta-run.lock"
    try:
        handle = lock_path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise ValueError(
            f"output directory is locked: {root}. Another run may be active. "
            "If it was killed, confirm it has stopped before removing .manta-run.lock."
        ) from exc
    try:
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        existing = [path for path in root.iterdir() if path != lock_path]
        checkpoint_path = root / "checkpoint.npz"
        in_place_resume = (
            resume_path is not None
            and root.resolve() == resume_path.parent.resolve()
            and (
                not checkpoint_path.exists()
                or checkpoint_path.resolve() == resume_path.resolve()
            )
        )
        if existing and not in_place_resume:
            raise ValueError(
                f"output directory already contains files: {root}. "
                "Use --resume for that run, or choose an empty --output-dir."
            )
        yield root
    finally:
        handle.close()
        lock_path.unlink(missing_ok=True)
