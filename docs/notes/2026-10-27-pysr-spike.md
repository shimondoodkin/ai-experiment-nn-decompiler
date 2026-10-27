# PySR feasibility spike (step 4, Task 1)

Date: 2026-10-27. Windows 11, Python 3.13, `uv sync --group pysr`.

- **Installed:** PySR 2.7.2 (Julia fetched by juliacall).
- **Warm-up:** the first `import pysr` took 8 min 16 s (Julia download, package resolve and
  precompile). It is not counted. Later imports take about 22 s.
- **`cond` operator:** available. The binary operators are `+ - * max min greater cond`, and the unary
  operators are `abs relu`.
- **Fit on `absdiff/hand`:** the design's settings (`maxsize=25`, `niterations=40`,
  `timeout_in_seconds=120`, `deterministic=True`, `parallelism="serial"`, `random_state=0`), 5,000
  samples of the net, took 74 s.
- **Best equation:** `abs(x1 - x0)` (sympy `Abs(x0 - x1)`), loss 0, complexity 4.

**Decision:** the spike passes, so B1/B2 stay in step 4. Run them with `uv run --group pysr`.
