# Command-Line Interface

`registry.py` declares names, descriptions, implementation modules and enforced
presets. `dispatcher.py` handles selection and global help. Numerical modules
load only when selected. `arguments/` owns per-command option parsing.

Every parser and experiment accepts an explicit `argv` list. Register additions
here rather than adding a top-level script or modifying `sys.argv`.
