# Performance Investigation of `ask_batch` method
Here, we report the profiling result of `ask_batch` with different `batch_size`. As we can see, as we increase `batch_size`, AB-MCTS run is accelerated, while the search tree shape gets more skewed.

## Profiling Details

```bash
hyperfine \
  'uv run tests/profiling/ab_mcts_m.py -b 1' \
  'uv run tests/profiling/ab_mcts_m.py -b 2' \
  'uv run tests/profiling/ab_mcts_m.py -b 5' \
  'uv run tests/profiling/ab_mcts_m.py -b 10' \
  -w 0 -r 3 --export-markdown tests/profiling/benchmark.md
```