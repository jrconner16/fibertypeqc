# Research tooling

Scripts used to develop and evaluate models and to compare FiberTypeQC with other methods. They are
**not part of the supported workflow**: they may change or be removed without notice, many expect
private inputs that are not in this repository, and their outputs are not covered by the release
checks. The supported product code (`src/`, `fibertypeqc/`, `scripts/`) never imports from here.

Run a tool with `uv run python -m research.<name> --help`. `validation/` contains thin command
wrappers over some of these tools.
