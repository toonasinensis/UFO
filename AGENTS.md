# Repository instructions for Codex

Read `MY_README.md` before preparing, launching, resuming, or documenting a training command.
In particular, every manifest-backed training launch must explicitly include
`--rebuild-motion-cache`, even though `run_train.sh` also enforces it as a safety net.
