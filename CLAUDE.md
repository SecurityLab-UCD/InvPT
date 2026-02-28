# CLAUDE.md

## Recent Updates

- Added per-operator robustness support for downstream evaluation.
- Downstream dataset augmentation scripts now accept an optional
  `--operator-key` argument to generate single-operator files:
  `aug_test_<operator_key>.jsonl`.
- `experiments_downstream/run_all_downstream.py` now includes a
  `per-operator` subcommand that runs `run_aug_test.sh` per operator.
- `experiments_downstream/parse_results.py` now supports
  `--per-operator` for operator breakdown tables.

## Operator Keys

- `localvarrenaming`
- `for2while`
- `while2for`
- `pp2addassignment`
- `addassignment2equalassignment`
- `reverseifelse`
