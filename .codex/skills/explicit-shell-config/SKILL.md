---
name: explicit-shell-config
description: Create or revise user-facing runnable shell scripts so required paths, iteration counts, rollout settings, devices, seeds, outputs, and switches are directly editable in one explicit configuration block. Use for repository .sh launchers, experiment runners, inference scripts, export scripts, and other commands users are expected to tune by editing the file.
---

# Explicit Shell Config

- Put a clearly marked configuration block immediately after repository/path setup.
- Assign required and commonly tuned values directly, for example `ITERATIONS=20` or `DEVICE="cuda:0"`.
- Do not use `${ENV_VAR:-default}` for normal user configuration.
- Pass every declared setting explicitly to the underlying command.
- Include paths, task/model names, devices/providers, iteration/population sizes, seeds,
  rollout lengths, objective thresholds, output directories, and feature switches when relevant.
- Keep variable names descriptive and group related settings with short comments.
- Convert boolean variables to the command's explicit positive or negative flag.
- Retain `"$@"` only as an optional final temporary override unless the user requests a
  completely fixed command.
- Validate with `bash -n`, inspect `--help` for argument spelling, and run a cheap smoke test
  when safe.
