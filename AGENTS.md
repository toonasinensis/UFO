# Repository instructions for Codex

Read `MY_README.md` before preparing, launching, resuming, or documenting a training command.
In particular, every manifest-backed training launch must explicitly include
`--rebuild-motion-cache`, even though `run_train.sh` also enforces it as a safety net.

For user-facing runnable shell scripts, keep required and commonly tuned parameters as
plain, explicit assignments in a clearly marked configuration block near the top of the
script. Do not hide normal configuration behind `${ENV_VAR:-default}` expressions. Pass
the explicit variables to the command so paths, iteration counts, rollout lengths, seeds,
devices, output locations, and feature switches can be changed by editing one file.

Use the repository skill `.codex/skills/explicit-shell-config/SKILL.md` when creating or
substantially changing a runnable `.sh` script.
