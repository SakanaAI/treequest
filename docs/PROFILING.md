# Performance Investigation of `ask_batch` method
Here, we report the profiling result of `ask_batch` with different `batch_size`. As we increase `batch_size`, AB-MCTS runs are accelerated, while the search tree shapes get more wide.

I profiled TreeQuest on a 32-core x86-64 machine running Ubuntu 20.04. As for AB-MCTS-M, it runs 4 parallel MCMC chains per process, so I tested `batch_size` from 1 to 10.

## Profiling Commands

### AB-MCTS-M
```bash
hyperfine \
  'uv run tests/profiling/ab_mcts_m.py -b 1' \
  'uv run tests/profiling/ab_mcts_m.py -b 2' \
  'uv run tests/profiling/ab_mcts_m.py -b 5' \
  'uv run tests/profiling/ab_mcts_m.py -b 10' \
  -w 0 -r 3 --export-markdown tests/profiling/benchmark_m.md
```

### AB-MCTS-A
```bash
hyperfine \
  'uv run tests/profiling/ab_mcts_a.py -b 1' \
  'uv run tests/profiling/ab_mcts_a.py -b 2' \
  'uv run tests/profiling/ab_mcts_a.py -b 5' \
  'uv run tests/profiling/ab_mcts_a.py -b 10' \
  -w 0 -r 3 --export-markdown tests/profiling/benchmark_a.md
```

## Analysis for various `batch_size`
### AB-MCTS-M
#### Time Profiling
For a search tree with (# non-root nodes) = 50,
| Command | Mean [s] | Min [s] | Max [s] | Relative |
|:---|---:|---:|---:|---:|
| `uv run tests/profiling/ab_mcts_m.py -b 1` | 372.156 ± 37.808 | 342.916 | 414.852 | 6.98 ± 0.72 |
| `uv run tests/profiling/ab_mcts_m.py -b 2` | 202.324 ± 8.006 | 194.178 | 210.182 | 3.80 ± 0.16 |
| `uv run tests/profiling/ab_mcts_m.py -b 5` | 91.418 ± 2.391 | 90.026 | 94.179 | 1.71 ± 0.05 |
| `uv run tests/profiling/ab_mcts_m.py -b 10` | 53.307 ± 0.747 | 52.468 | 53.902 | 1.00 |

> ![NOTE]
> For more resource-constrained systems, we get more mild boost. I ran profiling for M3 MacBook Air with 4 Pcores and 4 Ecores:
> | Command | Mean [s] | Min [s] | Max [s] | Relative |
> |:---|---:|---:|---:|---:|
> | `uv run tests/profiling/ab_mcts_m.py -b 1` | 147.564 ± 4.527 | 144.297 | 152.731 | 1.90 ± 0.08 |
> | `uv run tests/profiling/ab_mcts_m.py -b 2` | 116.557 ± 3.015 | 113.391 | 119.395 | 1.50 ± 0.06 |
> | `uv run tests/profiling/ab_mcts_m.py -b 5` | 90.628 ± 1.760 | 89.299 | 92.624 | 1.17 ± 0.04 |
> | `uv run tests/profiling/ab_mcts_m.py -b 10` | 77.773 ± 2.050 | 75.819 | 79.908 | 1.00 |


We note that different batch size leads to different tree shape; The larger batch size generates wider search tree:

#### `batch_size=1`
![batch_size=1](../images/batch/abmcts_m_batch_size_1.jpg)

#### `batch_size=2`
![batch_size=2](../images/batch/abmcts_m_batch_size_2.jpg)

#### `batch_size=5`
![batch_size=5](../images/batch/abmcts_m_batch_size_5.jpg)

#### `batch_size=10`
![batch_size=10](../images/batch/abmcts_m_batch_size_10.jpg)

## Analysis for various `batch_size`
### AB-MCTS-A
#### Time Profiling
AB-MCTS-A is faster than AB-MCTS-M, so I tested (# non-root nodes) = 1000:
| Command | Mean [s] | Min [s] | Max [s] | Relative |
|:---|---:|---:|---:|---:|
| `uv run tests/profiling/ab_mcts_a.py -b 1` | 26.116 ± 0.656 | 25.677 | 26.871 | 1.00 |
| `uv run tests/profiling/ab_mcts_a.py -b 2` | 26.191 ± 0.709 | 25.599 | 26.976 | 1.00 ± 0.04 |
| `uv run tests/profiling/ab_mcts_a.py -b 5` | 26.404 ± 0.441 | 26.061 | 26.902 | 1.01 ± 0.03 |
| `uv run tests/profiling/ab_mcts_a.py -b 10` | 26.399 ± 0.364 | 25.998 | 26.709 | 1.01 ± 0.03 |


> [!NOTE] 
> I didn't parallelize AB-MCTS-A, so the command running time is the same for different batch sizes. 
> Intially I also implemented parallel run for AB-MCTS-A as well, but it didn't lead to much speedup:
> | Command | Mean [s] | Min [s] | Max [s] | Relative |
> |:---|---:|---:|---:|---:|
> | `uv run tests/profiling/ab_mcts_a.py -b 1` | 49.973 ± 2.909 | 47.645 | 53.235 | 2.79 ± 0.16 |
> | `uv run tests/profiling/ab_mcts_a.py -b 2` | 30.971 ± 0.546 | 30.528 | 31.581 | 1.73 ± 0.03 |
> | `uv run tests/profiling/ab_mcts_a.py -b 5` | 21.256 ± 0.706 | 20.732 | 22.058 | 1.18 ± 0.04 |
> | `uv run tests/profiling/ab_mcts_a.py -b 10` | 17.941 ± 0.110 | 17.867 | 18.068 | 1.00 |


We note that different batch size leads to different tree shape; The larger batch size generates wider search tree:

#### `batch_size=1`
![batch_size=1](../images/batch/abmcts_m_batch_size_1.jpg)

#### `batch_size=2`
![batch_size=2](../images/batch/abmcts_m_batch_size_2.jpg)

#### `batch_size=5`
![batch_size=5](../images/batch/abmcts_m_batch_size_5.jpg)

#### `batch_size=10`
![batch_size=10](../images/batch/abmcts_m_batch_size_10.jpg)