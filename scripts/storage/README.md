# Manta checkpoints

`save_manta_checkpoint(path, result, config)` writes one `checkpoint.npz` archive.
`load_manta_checkpoint(path)` accepts that file or its containing run directory
and returns a `MantaCheckpoint` with `config` and `result` fields.

Schema version 1 stores effective configuration and step statuses in a JSON
Unicode scalar named `metadata`. State histories, LMPC controls, slack telemetry,
and APF seed controls are separate real numeric arrays. Every completed attempt
is retained, including attempts that were not admitted for learning. Loading
recomputes safety, goal, learning, and selected-iteration verdicts from this data.
Missing members, incompatible versions, malformed metadata, invalid shapes,
nonfinite states or controls, and object arrays are rejected. Slack NaNs are
preserved because held or fallback steps have no solver telemetry.

Saving validates the snapshot, writes a unique temporary file in the destination
directory, flushes and synchronizes it, then atomically replaces the checkpoint.
A failed write or interrupted replacement preserves the previous checkpoint and
removes that save's temporary file. NumPy loading always uses
`allow_pickle=False`; this format adds no dependencies and never loads pickle.

Checkpoints represent completed iteration boundaries. An interrupted iteration's
partial states and the optimizer's internal state are not part of the archive.
