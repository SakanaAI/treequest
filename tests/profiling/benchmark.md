| Command | Mean [s] | Min [s] | Max [s] | Relative |
|:---|---:|---:|---:|---:|
| `uv run tests/profiling/ab_mcts_m.py -b 1` | 372.156 ± 37.808 | 342.916 | 414.852 | 6.98 ± 0.72 |
| `uv run tests/profiling/ab_mcts_m.py -b 2` | 202.324 ± 8.006 | 194.178 | 210.182 | 3.80 ± 0.16 |
| `uv run tests/profiling/ab_mcts_m.py -b 5` | 91.418 ± 2.391 | 90.026 | 94.179 | 1.71 ± 0.05 |
| `uv run tests/profiling/ab_mcts_m.py -b 10` | 53.307 ± 0.747 | 52.468 | 53.902 | 1.00 |
