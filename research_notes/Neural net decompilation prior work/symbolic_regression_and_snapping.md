# Symbolic regression baselines and "snapping" learned constants: prior work for decompiling small ReLU MLPs

Scope: SR tools that can serve as runnable baselines (formula from samples of a trained net), KAN auto-symbolic snapping, and evidence that rounding, quantizing or simplifying a network into a program recovers exact algorithms or generalizes better. Project tasks: |a−b|, max, median-of-3, tax brackets, |x|<0.5 window, |a−b|<0.2, XOR, all with 1 to 3 inputs.

Verification note: claims with a [Source] link were checked in this session (Oct 2026). A few items from background knowledge that I could not re-verify are marked "(unverified)" and listed under Gaps.

---

## Q1. PySR: capabilities (abs/min/max/conditionals), install, runtime

### Takeaway
PySR is the strongest single baseline for this project. Its built-in operator set covers every target task in closed form: `abs`, `max`, `min`, `cond`, `greater`, `logical_or`, `logical_and`, `sign`, `relu`, `round`, `floor`, `ceil`, `mod`. It installs with `pip install pysr`, and its Julia dependencies install at first import. The main costs are the Julia compile time on first run and the stochastic, multi-threaded search.

### Cited Findings
- Built-in binary operators include `+ - * / ^`, `max`, `min`, `mod`, plus these conditionals:
  - `cond` = `(x, y) -> x > 0 ? y : 0`
  - `greater` = `(x, y) -> x > y ? 1 : 0`
  - `logical_or`, `logical_and` (threshold at 0)

  [PySR operators.md (HF Space mirror of docs)](https://huggingface.co/spaces/MilesCranmer/PySR/raw/7763215ce65b491b19742a7cacce36735dd3cd95/docs/operators.md)
- Built-in unary operators include `neg`, `square`, `cube`, `abs`, `sign`, `relu`, `round`, `floor`, `ceil`, `exp`, `log`, `sqrt`, trig and hyperbolic functions, `erf` and `gamma`. Any Julia function that takes one or two scalars and returns one scalar can be added as a custom operator. Custom operators need a matching `extra_sympy_mappings` entry. They must not throw on any real input; they return NaN for invalid inputs instead. — [PySR operators.md](https://huggingface.co/spaces/MilesCranmer/PySR/raw/7763215ce65b491b19742a7cacce36735dd3cd95/docs/operators.md)
- Current docs list `max`, `min` and `cond` as binary operators and `abs` and `sign` as unary operators. — [PySR docs (operators, v1.5.9)](https://ai.damtp.cam.ac.uk/pysr/v1.5.9/operators)
- Install with `pip install pysr` or `conda install -c conda-forge pysr`. The README says "Julia dependencies will be installed at first import". Docker and Apptainer images are also available. — [PySR GitHub README](https://github.com/MilesCranmer/PySR)
- A third-party skill page says Julia is downloaded automatically on first import and that "Compiling Julia backend..." on first run is normal. — [tessl.io PySR skill (third-party)](https://tessl.io/registry/skills/github/astroautomata/PySR/pysr)
- Runtime and search controls:
  - The search runs as a multi-threaded Julia process.
  - The quickstart uses `niterations=40`, which the README describes as hundreds of thousands of mutations and evaluations.
  - `timeout_in_seconds` caps wall time.
  - The search can be stopped early with `q` then Enter.

  [PySR GitHub README](https://github.com/MilesCranmer/PySR)
- Citation is arXiv:2305.01582 ("Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl"). — [PySR GitHub README](https://github.com/MilesCranmer/PySR)
- Older PySR setups required a manual Julia install plus the Optim and SpecialFunctions packages. Install docs found on old mirrors are therefore outdated. — [PySR on PyPI, v0.2.1 (old)](https://pypi.org/project/pysr/0.2.1)

### Inferences
- Every task has a short exact PySR expression:
  - |a−b| = `abs(a-b)`
  - max = `max(a,b)`
  - median3 = `max(min(a,b), min(max(a,b),c))`
  - |x|<0.5 = `greater(0.5, abs(x))`
  - |a−b|<0.2 = `greater(0.2, abs(a-b))`
  - XOR on {0,1} = `abs(a-b)`, or `logical_*` combinations
  - Tax brackets = sum of rate·`relu(x−threshold)` terms, or nested `min`/`max`

  Because exact formulas exist, PySR is a fair "black-box SR on network samples" baseline against the repo's AST tool. It also gives a direct comparison of whether PySR's fitted constants come out blurry (0.198) or exact (0.2).
- Restricting `binary_operators` and `unary_operators` to the piecewise set (`+ - * max min abs greater cond relu`) should keep runs on 1 to 3 inputs short. For reproducible baselines, use `deterministic=True` with single-threaded settings and a fixed `random_state`. Check those parameter names against the installed version.
- PySR's model selection trades loss against complexity, so it does not snap constants by default. A fair comparison applies the repo's snapping step to PySR's output as well.

### Gaps
- I could not fetch the live pysr.ai operators page (the redirect returned empty content), so the current release number and date were not confirmed. The operator list above comes from a docs snapshot.
- I found no published wall-clock timings for PySR on 1 to 3 variable problems. Measure locally. My unverified expectation is under a minute after Julia compiles, with a first-run compile of a minute or more.

---

## Q2. gplearn, AI Feynman (1.0/2.0), Operon, SRBench: what works with pip in 2026

### Takeaway
- **gplearn** (pure Python, scikit-learn API) is the easiest secondary baseline. It has no built-in abs/min/max, but custom functions can be added. Maintenance status is uncertain.
- **AI Feynman** needs a Fortran compiler and runs only on Linux/Mac. Treat it as impractical on this Windows laptop except under WSL.
- **Operon** (pyoperon) is fast, C++, and a top SRBench method. Its PyPI build has lagged behind its GitHub wheels.
- **SRBench** is a benchmark harness, not a solver. Its 2025 roster includes PySR and is the recommended baseline set.

### Cited Findings
- **gplearn:**
  - Installs with pip after scikit-learn, and needs numpy and scipy. — [gplearn installation docs](https://gplearn.readthedocs.io/en/stable/installation.html)
  - It was one of the 14 SR methods in SRBench 2021. — [SRBench site](https://cavalab.org/srbench)
- **AI Feynman (pip name `aifeynman`):**
  - Supported only on Linux and Mac.
  - Installs with pip, but a Fortran compiler such as gfortran must be on PATH.

  [AI Feynman installation docs](https://ai-feynman.readthedocs.io/en/latest/installation.html)
- AI Feynman 2.0 was among the SRBench 2021 methods. — [SRBench site](https://cavalab.org/srbench)
- MIPS uses AI Feynman's brute-force search for its nonlinear integer symbolic regression step:
  - Expression templates up to length 6
  - Unary operators {>, <, ∼, H, D, A}
  - Binary operators {+, −, ∗, %}

  So AI Feynman-style brute force with comparison operators is a precedent for this project's tasks. — [MIPS, arXiv:2402.05110](https://arxiv.org/html/2402.05110)
- **Operon / pyoperon:**
  - The PyPI page says the PyPI version is outdated because of ABI incompatibility. Wheels are released on GitHub, and the build route is clone then `pip install .` in a conda or micromamba environment. — [pyoperon on PyPI](https://pypi.org/project/pyoperon)
  - piwheels lists pyoperon 0.6.0 (Feb 2026) and 0.6.1 (2026-03-01), so newer builds exist. — [piwheels pyoperon](https://www.piwheels.org/project/pyoperon)
- **SRBench 2021** (La Cava et al., NeurIPS 2021 Datasets & Benchmarks):
  - 14 SR methods and 7 ML methods on 252 problems.
  - Several methods performed similarly at exact-solution recovery under noise.
  - The best real-world methods combine genetic algorithms with parameter estimation and/or semantic search.

  [La Cava et al. 2021 PDF](https://cavalab.org/assets/papers/La%20Cava%20et%20al.%20-%202021%20-%20Contemporary%20Symbolic%20Regression%20Methods%20and%20their.pdf)
- **SRBench 2025:**
  - 25 SR methods in Docker containers, 30 runs per dataset, 24 PMLB datasets plus first-principles problems.
  - The roster adds PySR, AFP_fe, Bingo, Brush, E2E, EQL, NeSymRes, QLattice, TPSR, uDSR and others.
  - The site recommends the 2025 roster for new baselines.
  - The minimum requirement for adding a method is a scikit-learn compatible API.

  [SRBench site](https://cavalab.org/srbench)
- The SRBench 2025 paper ("Call for Action", GECCO 2025 SR workshop) concludes that no single algorithm dominates all datasets. It adds complexity, accuracy and energy trade-off analysis. — [arXiv:2505.03977](https://arxiv.org/abs/2505.03977)
- A Dec 2025 review covers open challenges in SR optimization, model selection, simplification and benchmarking. — [arXiv:2512.01682](https://arxiv.org/pdf/2512.01682)
- **Feyn / QLattice** (Abzu) is a separate commercial-origin SR tool, not AI Feynman. It supports Python 3.8 to 3.12 and is in the SRBench 2025 roster as "Qlattice". — [Feyn install docs](https://docs.abzu.ai/docs/guides/installation.html); [SRBench site](https://cavalab.org/srbench)

### Inferences
- Practical ranking for a Windows laptop CPU in 2026:
  1. **PySR**: pip, all needed operators built in
  2. **gplearn**: pure pip; add abs/max/min/greater through `make_function`
  3. **pyoperon**: fast, but install from GitHub wheels; its operator set is mostly arithmetic and transcendental, so piecewise tasks are likely awkward (unverified)
  4. **AI Feynman**: WSL only; its dimensional-analysis and separability tricks add little for 1 to 3 inputs
- SRBench is useful as a citation and a method roster rather than something to run. Running it means Docker and Git LFS, which is heavy for this repo.

### Gaps
- Not confirmed:
  - gplearn's latest release and Python 3.12/3.13 compatibility. My unverified recollection is that the last release, 0.4.2, dates from around 2022.
  - aifeynman's maintenance status.
  - pyoperon's built-in operator list, in particular whether it has abs/min/max.
- No per-method SRBench rankings were retrieved. AI Feynman's strong noiseless ground-truth recovery and Operon's strong black-box accuracy in 2021 are my recollection and are unverified.
- No year, venue or URL was checked for AI Feynman 1.0 (Udrescu & Tegmark, Science Advances 2020) or AI Feynman 2.0 (NeurIPS 2020). Both are from memory and unverified.

---

## Q3. KANs and auto-symbolic snapping (pykan)

### Takeaway
KAN symbolification is the closest published analogue of this repo's "snapping":
1. Sparsify and prune the network.
2. Replace each learned 1-D edge spline with the best-fitting library function c·f(a·x+b)+d, chosen by R².
3. Retrain only the affine constants.
4. Read off a SymPy formula.

The library includes `abs` and `sgn`, so |a−b| and max(a,b) = (a+b+|a−b|)/2 are reachable. Comparisons and step-like functions are only approximated through `sgn` and `tanh`.

### Cited Findings
- **Paper:** "KAN: Kolmogorov–Arnold Networks", Liu et al., arXiv:2404.19756 (Apr 2024, rev. Feb 2025), published at ICLR 2025 (poster). — [arXiv](https://arxiv.org/abs/2404.19756); [ICLR 2025 poster page](https://iclr.cc/virtual/2025/poster/29784)
- **Follow-up:** "KAN 2.0: Kolmogorov-Arnold Networks Meet Science", arXiv:2408.10205. — [pykan GitHub](https://github.com/KindXiaoming/pykan); [arXiv 2408.10205](https://arxiv.org/html/2408.10205v1)
- **Symbolification pipeline from the paper:**
  1. Sparsify with an L1 penalty on activation magnitudes plus an entropy penalty. L1 alone was not sufficient.
  2. Prune nodes whose largest incoming and outgoing activation norm falls below θ = 10⁻².
  3. `fix_symbolic(l,i,j,f)` sets an edge to function f. It fits y ≈ c·f(a·x+b)+d using iterative grid search over (a,b) and linear regression for (c,d).
  4. `suggest_symbolic` proposes candidate functions.
  5. Once every edge is symbolic, only the affine parameters are retrained. Reaching "machine precision" loss is taken as the signal that the formula is correct.
  6. SymPy produces the final expression.

  [KAN paper](https://arxiv.org/html/2404.19756)
- **Example:** f = exp(sin(πx)+y²) was recovered as 1.0·exp(1.0·y² + 1.0·sin(3.14x)). This is snapping-like recovery of exact coefficients. — [KAN paper](https://arxiv.org/html/2404.19756)
- **pykan `SYMBOLIC_LIB` entries:**
  - Powers and roots: `x`, `x^2` … `x^5`, `1/x` … `1/x^5`, `sqrt`, `x^0.5`, `x^1.5`, `1/sqrt(x)`
  - Exponential and log: `exp`, `log`
  - Piecewise: `abs`, `sgn`
  - Trig and hyperbolic: `sin`, `cos`, `tan`, `tanh`, `arcsin`, `arccos`, `arctan`, `arctanh`
  - Other: `0`, `gaussian`
  - `relu`, `sigmoid` and `cosh` are present but commented out.

  [pykan kan/utils.py](https://raw.githubusercontent.com/KindXiaoming/pykan/master/kan/utils.py)
- **`fit_params` details:**
  - Grid search over (a,b) on a 101×101 grid in [−10,10] by default, scored by squared-correlation R².
  - 3 zoom-in iterations.
  - c and d come from sklearn `LinearRegression`.
  - It warns when R² < 0.9.

  [pykan kan/utils.py](https://raw.githubusercontent.com/KindXiaoming/pykan/master/kan/utils.py)
- `auto_symbolic` replaces each activation with the best `suggest_symbolic` candidate. Its parameters include `min_r2`, the library `lib`, and the a/b sweep ranges. This is described in a pykan-derived package's docs. — [CoxKAN docs](https://coxkan.readthedocs.io/en/latest/coxkan.html)
- Recommended workflow: train, prune, then `auto_symbolic()`, then retrain affine parameters, then `symbolic_formula()`. — [pykan docs index (Algolia)](https://docsearch.algolia.com/mcp/docs/repo/kindxiaoming/pykan)
- **Install and runtime:**
  - `pip install pykan`, Python ≥ 3.9.7.
  - Pinned requirements: torch==2.2.2, numpy==1.24.4, sympy==1.11.1.
  - Tutorials run on a single CPU in under 10 minutes.
  - Call `model.speed()` to turn off the slow symbolic branch when training without symbolic use.

  [pykan GitHub](https://github.com/KindXiaoming/pykan)
- One applied paper (transistor modeling) notes that the whole-network formula can become too convoluted for complex patterns. — [arXiv:2503.15209](https://arxiv.org/pdf/2503.15209)

### Inferences
- **KAN is a different model class:** It snaps learned spline activations, not ReLU MLP weights. As a baseline, it means "train a KAN on network samples or the task, then symbolify". It cannot be applied directly to the project's existing MLP weights.
- **What its library can express:**
  - |a−b| and max fit naturally, using a hidden node that sums a and −b, then `abs`.
  - Median-of-3 needs nested abs/max and will be fragile.
  - Window, near-equality and XOR-as-threshold need a step function. `sgn` exists, but fitting it by R² grid search on a discontinuous target is likely unreliable.
  - Adding `relu`, or a custom step function, to the library is possible but non-default.
- **Pinned dependencies:** The old torch, numpy and sympy pins may conflict with a modern environment, so use a separate venv.
- **Analogy to this repo's snapping:** KAN's "choose function by R², then refit constants" is functional-form snapping. The repo's 0.198→0.2 is constant snapping. KAN does not round the fitted (a,b,c,d) to nice values by default, although the paper's displayed 1.0 coefficients show they often land there.

### Gaps
- I did not read the paper's discussion of discontinuous or piecewise targets (the second half of the HTML was not fetched).
- No independent benchmark of KAN symbolification on piecewise or boolean tasks was found.
- The current pykan PyPI version number was not confirmed.

---

## Q4. Rounding, quantizing or simplifying weights to recover exact algorithms, and whether the simplified program generalizes better

### Takeaway
- **Snapping as a pipeline step:** The clearest precedent is MIPS (Michaud, Tegmark et al. 2024). Its pipeline explicitly rounds any weight or bias within 0.01 of an integer, after symmetry normalizations. It distills RNNs into Python and solves 32 of 62 algorithmic tasks.
- **Evidence of better generalization:** MIPS does not show that the snapped programs generalize better than the network. The strongest published evidence of "symbolic beats network out of distribution" is Cranmer et al. 2020.
- **Quantization literature:** Generic quantization-for-generalization work is mixed. It is not about exact algorithm recovery.

### Cited Findings
- **MIPS** ("Opening the AI black box: program synthesis via mechanistic interpretability", Michaud, Tegmark et al.):
  - arXiv:2402.05110 (Feb 2024).
  - Published in *Entropy*, Dec 2024, as "Opening the AI Black Box: Distilling Machine-Learned Algorithms into Code".
  - Pipeline: an integer autoencoder converts the RNN into a finite state machine, then Boolean or integer symbolic regression recovers the algorithm.
  - Solves 32 of 62 tasks, including 13 that GPT-4 does not solve. GPT-4 solves 30.

  [arXiv 2402.05110](https://arxiv.org/pdf/2402.05110); [PMC / Entropy](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11727218/)
- **MIPS "auto-simplification" normalizers**, applied in order:
  1. Whitening
  2. Jordan normal form of W
  3. Toeplitz simplification of V
  4. De-bias
  5. **Quantization**: round any weight or bias within ε = 0.01 of an integer to that integer

  Without the normalizers, the generated code has large arbitrary constants. With them, Sum_Last5 becomes a short sum of five state variables. — [MIPS HTML](https://arxiv.org/html/2402.05110)
- **MIPS integer lattice finder:**
  - A generalized-GCD lattice finder with a noise tolerance ε_gcd, swept over [10⁻³, 1].
  - Settings are picked by minimum description length.
  - Linear integer regression: fit a linear regression, round the coefficients to integers, then simplify.

  [MIPS HTML](https://arxiv.org/html/2402.05110)
- **MIPS code and data:** github.com/ejmichaud/neural-verification. — [MIPS HTML](https://arxiv.org/html/2402.05110)
- **MIPS evaluation scope:**
  - Programs are evaluated on the same test batches.
  - Loop lengths are fixed, e.g. `range(10)`.
  - A Dafny proof covers length-10 bit addition.
  - The paper does not demonstrate that distilled programs generalize better OOD than the RNN. It does note that continuous RNN computation fails on Majority tasks.

  [MIPS HTML](https://arxiv.org/html/2402.05110)
- **Cranmer et al. 2020** ("Discovering Symbolic Models from Deep Learning with Inductive Biases", NeurIPS 2020, arXiv:2006.11287):
  - Train a GNN with sparse latent representations, then run SR on its message and node functions.
  - Recovers force laws and Hamiltonians, and finds a new dark-matter concentration formula.
  - The abstract states: "The symbolic expressions extracted from the GNN using our technique also generalized to out-of-distribution-data better than the GNN itself."

  [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2020/hash/c9f2f917078bd2db12f23c3b413d9cba-Abstract.html); [arXiv abs](https://alphaxiv.org/abs/2006.11287)
- **Grokking** (Nanda et al., "Progress measures for grokking via mechanistic interpretability", arXiv:2301.05217):
  - The modular-addition algorithm is confirmed with Fourier-space ablations.
  - "Restricted loss" keeps only the constant plus 5 key frequencies. "Excluded loss" removes them.
  - Training is described as memorization, then circuit formation, then "cleanup" that removes memorizing components.

  [arXiv 2301.05217](https://arxiv.org/pdf/2301.05217)
- I could not verify the claim that restricting to key frequencies lowers loss below the full model's. The abstract page does not state it.
- **Quantization and generalization (generic, not algorithm recovery):**
  - NTK analysis of binary-weight networks suggests a smaller generalization gap, formalizing the "quantization as implicit regularizer" folklore (ICML 2023 workshop). — [arXiv:2206.05916](https://arxiv.org/pdf/2206.05916)
  - QT-DoG (2024) reports that quantization-aware training finds flatter minima and improves domain generalization. — [arXiv:2410.06020](https://arxiv.org/html/2410.06020v1)
  - Counterpoint: a 2025 paper finds standard QAT can significantly hurt OOD performance through sharp minima. Its flatness-oriented FQAT gains +5.24% OOD vs +2.61% ID in one 3-bit PACS setting. — [arXiv:2509.00859](https://arxiv.org/html/2509.00859v1)
- **Simplicity and length generalization:**
  - Zhou et al. (ICLR 2024) conjecture that transformers length-generalize when a short RASP-L program solves the task. — [ICLR 2024 proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/45ed1a72597594c097152ef9cc187762-Abstract-Conference.html)
  - A 2025 UW thesis testing hundreds of synthetic RASP-L tasks reports that the evidence does not support the conjecture. — [UW thesis](https://digital.lib.washington.edu/researchworks/items/7d479c1a-c15a-4cae-aac2-ea0595d57826/full)

### Inferences
- The repo's "snap blurry constants to intended values, then the program beats the network on the true task and OOD" has direct method precedent in MIPS (round within 0.01 of an integer) and KAN (snap to function forms, refit). Its outcome precedent is Cranmer 2020 (symbolic generalizes OOD better than the net).
- The repo's specific claim is not established in prior work as far as I found: rational or "intended-constant" snapping (0.198→0.2, not just integers) on ReLU MLPs for piecewise or boolean tasks, measured as beating the source network OOD. MIPS rounds only to integers and does not test OOD. Cranmer 2020 uses SR, not weight snapping. This is a plausible novelty claim, but word it carefully.
- **Why snapping helps on piecewise tasks:** A ReLU MLP's blurry breakpoint (|x|<0.497 instead of 0.5) and slope errors grow linearly with input magnitude. A snapped formula has zero such error, so its OOD advantage over the network is expected, not surprising. Cranmer 2020's OOD result has the same structure: an exact formula extrapolates and a fitted approximator does not.
- Generic quantization evidence (NTK, QAT) is about classification regularization and is mixed. Cite it only as context, not as support for exact-algorithm recovery.

### Gaps
- Friedman, Wettig & Chen, "Learning Transformer Programs" (NeurIPS 2023), which trains discrete transformers convertible to Python, was not verified (search failed). The same applies to Liu & Tegmark's BIMT ("Seeing is Believing", brain-inspired modular training).
- Lottery-ticket or pruning work that explicitly reveals algorithms was not searched, for lack of budget.
- I found no paper that rounds MLP weights to rationals or decimal "intended values" and reports OOD gains over the network.
- The MIPS Entropy version may contain an OOD or length-generalization analysis that the arXiv HTML lacks. Not checked.

---

## Q5. Summary table: year, venue, code, piecewise expressivity, fit for 1 to 3 inputs

### Takeaway
For runnable baselines on piecewise tasks with 1 to 3 inputs: use PySR first, gplearn with custom abs/max/greater functions second, and optionally pykan. Cite MIPS, Cranmer 2020 and KAN as the snapping and distillation precedents.

### Cited Findings

| Method | Year / venue | Code / install (2026) | Piecewise ops | 1 to 3 input suitability |
|---|---|---|---|---|
| PySR | 2023, arXiv:2305.01582 ([README](https://github.com/MilesCranmer/PySR)) | `pip install pysr`; Julia deps auto-install at first import ([README](https://github.com/MilesCranmer/PySR)) | abs, min, max, cond, greater, logical_and/or, sign, relu, round/floor/ceil, mod ([ops](https://huggingface.co/spaces/MilesCranmer/PySR/raw/7763215ce65b491b19742a7cacce36735dd3cd95/docs/operators.md)) | Excellent |
| gplearn | In SRBench 2021 roster ([SRBench](https://cavalab.org/srbench)) | pip + scikit-learn ([docs](https://gplearn.readthedocs.io/en/stable/installation.html)) | Custom functions only (my inference) | Good, pure Python |
| AI Feynman 2.0 | In SRBench 2021 roster ([SRBench](https://cavalab.org/srbench)) | pip + gfortran; Linux/Mac only ([docs](https://ai-feynman.readthedocs.io/en/latest/installation.html)) | Brute-force template search with comparison ops (as used in [MIPS](https://arxiv.org/html/2402.05110)) | Poor on Windows |
| Operon (pyoperon) | In SRBench 2021 roster ([SRBench](https://cavalab.org/srbench)) | PyPI build outdated; GitHub wheels ([PyPI](https://pypi.org/project/pyoperon)); 0.6.x in 2026 ([piwheels](https://www.piwheels.org/project/pyoperon)) | Not confirmed | Fast; install friction |
| SRBench | NeurIPS 2021 D&B; 2025 edition at GECCO 2025 workshop ([SRBench](https://cavalab.org/srbench), [arXiv:2505.03977](https://arxiv.org/abs/2505.03977)) | Docker, Git LFS ([SRBench](https://cavalab.org/srbench)) | n/a (harness) | Roster and citation only |
| KAN / pykan | ICLR 2025 ([ICLR](https://iclr.cc/virtual/2025/poster/29784)); KAN 2.0 arXiv:2408.10205 | `pip install pykan`, pinned old torch, numpy and sympy ([GitHub](https://github.com/KindXiaoming/pykan)) | abs, sgn in lib; relu commented out ([utils.py](https://raw.githubusercontent.com/KindXiaoming/pykan/master/kan/utils.py)) | Good for abs/max; weak for thresholds |
| MIPS | arXiv 2024; *Entropy* Dec 2024 ([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11727218/)) | github.com/ejmichaud/neural-verification ([MIPS](https://arxiv.org/html/2402.05110)) | Boolean DNF, integer SR with >, < ([MIPS](https://arxiv.org/html/2402.05110)) | Built for RNNs on integer and bit tasks; precedent, not a drop-in baseline |
| Cranmer et al. symbolic distillation | NeurIPS 2020 ([proc.](https://proceedings.neurips.cc/paper/2020/hash/c9f2f917078bd2db12f23c3b413d9cba-Abstract.html)) | Uses SR (PySR's predecessor ecosystem; unverified) | Depends on SR backend | Precedent for "symbolic beats net OOD" |

### Inferences
- The most defensible experimental setup has two parts:
  - **Baseline A:** PySR fitted on samples of the trained network, with operators restricted to the piecewise set.
  - **Baseline B:** PySR fitted on the same samples, followed by the repo's constant-snapping pass.

  Comparing both against the repo's AST tool on in-distribution and OOD accuracy separates "search quality" from "snapping benefit".
- KAN is optional. It answers a different question: does a model class designed for symbolification recover these tasks?

### Gaps
- No laptop CPU timings were measured for any tool in this session.
- No SRBench per-method numbers were retrieved.
