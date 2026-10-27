# Rule extraction from neural networks and compiling networks into branch code / trees

Scope: classic and modern rule extraction (pedagogical vs decompositional), decision-tree distillation, exact ReLU-to-tree / linear-region decomposition, and compiling networks or rules into branching code or lookup tables for speed, plus straight-line NN-to-C baselines. Focus: runnable open-source baselines for small 1-3 input ReLU MLPs.

Legend used below: **Exact** = the extracted artifact reproduces the network function exactly (up to float error); **Approx** = fidelity < 100%. **Pedagogical** = black-box/query-based (only input-output); **Decompositional** = uses weights / per-neuron structure; **Eclectic** = hybrid.

## Q1. Classic rule extraction (TREPAN, DeepRED, taxonomy, surveys) and maintained Python implementations

### Takeaway
The field's standard taxonomy (Andrews/Diederich/Tickle 1995) splits methods into pedagogical, decompositional and eclectic; classic methods (TREPAN, DeepRED, ECLAIRE) are all approximate rule/tree extractors aimed at classifiers. Python implementations exist but are research-grade and stale: `ruleex` (PyPI, 2019, HypInv/ANN-DT/DeepRED) and REMIX (GitHub, GPL-3, DeepRED/REM-D/ECLAIRE/PedC5.0, needs R + old Python 3.5-3.8). No maintained pip-installable TREPAN was found.

### Cited Findings
- Taxonomy: Andrews, Diederich & Tickle's survey classifies rule extractors by five criteria: rule representation language, translucency, rule quality, computational complexity and portability; translucency gives pedagogical (black box), decompositional (neuron-level) and eclectic (both) — as summarized in [El Mekkaoui et al., arXiv:2312.12878](https://arxiv.org/pdf/2312.12878) and [Hailesilassie 2016, arXiv:1610.05267](https://arxiv.org/abs/1610.05267). The Andrews et al. survey ("Survey and critique of techniques for extracting rules from trained artificial neural networks") appeared in Knowledge-Based Systems (ISSN 0950-7051) — [datalearner listing](https://datalearner.com/academic/journal-papers/0950-7051/volumes-and-issues/34/paper-detail/5254).
- The pedagogical/decompositional split is attributed to Craven & Shavlik; eclectic methods are a hybrid — [Knowledge Augmented ML survey, arXiv:2205.04712](https://arxiv.org/pdf/2205.04712); decompositional methods extract rules "at the level of each individual hidden and output unit" — [Eclectic Extraction of Propositional Rules, arXiv:1110.0214](https://arxiv.org/pdf/1110.0214).
- TREPAN (Craven & Shavlik) is pedagogical: it treats the trained network as an oracle answering queries and induces a decision tree approximating the network function — [X-TREPAN paper, arXiv:1508.07551](https://arxiv.org/pdf/1508.07551). X-TREPAN extends it to multi-class and regression — same source.
- Hailesilassie's 2016 DNN review organizes DNN rule extraction under the three headings and concludes there is limited study of rule extraction specifically for DNNs — [arXiv:1610.05267](https://arxiv.org/abs/1610.05267).
- Recent survey: El Mekkaoui, Benabbou & Berrado, "Rule-Extraction Methods From Feedforward Neural Networks: A Systematic Literature Review" (arXiv Dec 2023) — proposes an extended taxonomy mapping the old criteria onto modern XAI terms (model-specific vs model-agnostic, intrinsic vs post-hoc) and examines whether older techniques transfer to deep feedforward nets — [arXiv:2312.12878](https://arxiv.org/pdf/2312.12878).
- `ruleex` (PyPI): v0.4.1, released 29 Oct 2019, author Matěj Fanta, MIT license; implements HypInv, ANN-DT and DeepRED (slightly modified), rules stored in a tree structure; deps: TensorFlow (for DeepRED MLP), scikit-learn, numpy, graphviz, optional rpy2 + R (C5.0) and pyblique; repo github.com/fantamat/ruleex — [PyPI ruleex](https://pypi.org/project/ruleex/).
- REMIX (GitHub mateoespinosa/remix): implements DeepRED (C5.0 variant), REM-D, ECLAIRE, PedC5.0 (pedagogical, C5.0 fitted to DNN outputs) and REM-T; requires Python 3.5-3.8, R 4.x with C50/Cubist, TensorFlow; GPL-3.0; 14 commits, 49 stars — [REMIX GitHub](https://github.com/mateoespinosa/remix).
- ECLAIRE (Espinosa Zarlenga & Jamnik, NeurIPS 2021 workshop): decompositional, input = pre-trained DNN + unlabeled training instances, output = IF-THEN rule set; designed to scale better than DeepRED/REM-D — [arXiv:2111.12628](https://arxiv.org/pdf/2111.12628), [NeurIPS virtual page](https://neurips.cc/virtual/2021/32140).
- DeepRED (Zilke, Loza Mencía, Janssen 2016) is in Springer LNCS (Discovery Science 2016) — [Springer DOI link](https://link.springer.com/doi/10.1007/978-3-319-46307-0_29); only implementations found are inside `ruleex` and REMIX — [PyPI ruleex](https://pypi.org/project/ruleex/), [REMIX](https://github.com/mateoespinosa/remix).
- NN2Rules (arXiv 2207.12271) extracts rule lists from neural networks without assuming soft-binary hidden neurons — [arXiv:2207.12271](https://arxiv.org/pdf/2207.12271); no package link found.

### Inferences
- All classic methods are **approximate** and **classification-oriented** (discrete outputs, Boolean/threshold rules). They are poor fits as baselines for a decompiler that targets exact, real-valued regression programs from 1-3 input ReLU MLPs; at best they serve as "readability baseline" comparisons for classification tasks in the project's suite.
- Practical runnability today: `ruleex` (2019, TF1-era) and REMIX (Python <= 3.8 + R) will likely need a pinned old environment; neither is a drop-in baseline. A hand-written pedagogical baseline (sample the net densely on the 1-3D domain, fit an sklearn `DecisionTreeRegressor`/`Classifier`) is effectively TREPAN-lite and far easier to run.

### Gaps
- No maintained Python TREPAN found (searched PyPI/GitHub-oriented queries); original TREPAN reference (Craven & Shavlik, NIPS 1996) not fetched directly.
- Could not confirm the journal/publication year of the El Mekkaoui et al. review beyond the arXiv version, nor the exact year/volume of Andrews et al. (commonly cited as 1995, KBS 8(6); not verified here).
- Did not verify whether ruleex/REMIX install on Python 3.11+/TF2.

## Q2. Decision-tree distillation baselines (sklearn trees on network outputs; Frosst & Hinton soft trees)

### Takeaway
The practical standard pedagogical baseline is: query the network on sampled inputs, fit a CART tree (sklearn) or C5.0 to its outputs (this is what REMIX's "PedC5.0" does). Frosst & Hinton's soft decision tree (2017) is a different, approximate, distillation-by-training method with several unofficial PyTorch implementations; it is not an extraction of the given network's function.

### Cited Findings
- PedC5.0 in REMIX is "a pedagogical method that uses C5.0 to approximate the DNN's outputs from its inputs" — [REMIX GitHub](https://github.com/mateoespinosa/remix). `ruleex` uses scikit-learn for training initial decision trees — [PyPI ruleex](https://pypi.org/project/ruleex/).
- Soft decision tree ("Distilling a Neural Network Into a Soft Decision Tree", Frosst & Hinton) PyTorch implementations: xuyxu/soft-decision-tree (BSD-3, ~100 stars, last push 2024-02-22) — [ecosyste.ms entry](https://awesome.ecosyste.ms/projects/github.com%2Fxuyxu%2Fsoft-decision-tree); zeta1999/soft-decision-tree (Python 3.6.5, PyTorch 0.4.1; MNIST depth 5/7 test acc 94.15/94.38) — [GitHub](https://github.com/zeta1999/soft-decision-tree); robchenchen/Soft-Decision-Tree (depth 8 ~95% MNIST; inference averages leaf distributions weighted by path probability) — [GitHub](https://github.com/robchenchen/Soft-Decision-Tree).
- A Dec 2025 paper presents an "explainable and extendable PyTorch implementation" of the soft decision tree classifier — [arXiv:2512.11833](https://arxiv.org/pdf/2512.11833).
- Neural-Backed Decision Trees (nbdt, PyPI) take the opposite route: train the network with a tree-supervision loss then infer with embedded decision rules, rather than converting an existing net — [nbdt GitHub](https://github.com/alvinwan/neural-backed-decision-trees), [PyPI nbdt](https://pypi.org/project/nbdt/).

### Inferences
- For this project, the cheapest credible approximate baseline is `sklearn.tree.DecisionTreeRegressor(max_depth=k)` (or `max_leaf_nodes`) fit on dense samples of the MLP over its 1-3D input box, reporting fidelity (max/mean abs error vs the net) and leaf count; optionally `LinearTreeRegressor`/model trees for piecewise-linear leaves. Such trees compile to C trivially (m2cgen, emlearn, TL2cgen; see Q4).
- Soft decision trees evaluate every node (probabilistic routing), so they are not branchy code and do not yield speedups; they are an interpretability baseline only.

### Gaps
- No source found establishing a single "standard" distillation baseline in the literature; the claim that sklearn CART on network outputs is the de-facto baseline is an inference from tooling (PedC5.0, ruleex), not from a survey statement.
- Frosst & Hinton original paper (arXiv 1711.09784, 2017) not fetched directly; no official code found.

## Q3. Exact conversion of ReLU networks to decision trees / linear regions; region counting and enumeration tools

### Takeaway
A ReLU MLP is exactly a continuous piecewise-linear function; it can be rewritten as an oblique (multivariate) decision tree whose nodes test neuron pre-activation signs and whose leaves hold affine maps (Nguyen et al. 2020 EC-DT; Aytekin 2022; Chang et al. 2023; RENTT 2025). Naive trees grow exponentially, but in low input dimension the number of realized regions is small (Hanin & Rolnick 2019: ~ #neurons^input_dim on average). Runnable exact tools exist: `parx` (pip, Julia backend, exact region enumeration with H-representation and per-region affine maps), SplineCam (CVPR 2023, exact 2D-slice region computation), RENTT (GitHub, research code).

### Cited Findings
- Aytekin, "Neural Networks are Decision Trees", arXiv:2210.05189 (Oct 2022, sole author): any NN with piecewise-linear activations can be converted to an equivalent decision tree — equivalence, not approximation; later versions broaden to any activation — [arXiv abs](https://arxiv.org/abs/2210.05189v1), [v3 HTML](https://arxiv.org/html/2210.05189v3). No code repository found.
- Nguyen, Kasmarik & Abbass, "Towards Interpretable ANNs: An Exact Transformation to Multi-Class Multivariate Decision Trees", arXiv:2003.04675 (2020, rev. 2021): EC-DT (Exact-Convertible Decision Tree) represents exactly the decision boundaries of hidden layers; Extended C-Net also proposed; no venue or code stated — [arXiv:2003.04675](https://arxiv.org/abs/2003.04675).
- Chang, Dahlin, Jain & Nuzzo, "Equivalent and Compact Representations of Neural Network Controllers With Decision Trees", arXiv:2304.06049 (2023, v3 Jul 2025): exact yet efficient transformation of ReLU+argmax NN controllers into soft decision trees with redundant-branch pruning; reports verification speedups up to 21x (MountainCar) and 2x (CartPole) but no inference-speed comparison; no code — [arXiv:2304.06049](https://arxiv.org/abs/2304.06049).
- RENTT (Monke, Fresz, Bernreuther, Chen, Huber, arXiv:2511.09299, Nov 2025): exact multivariate-tree equivalent of NNs, extended to CNN/RNN, non-ReLU activations and biases; compared runtime/memory vs Nguyen et al. (2020); adds RENTT-FI feature importance; notes scalability issues may remain for complex nets — [arXiv:2511.09299](https://arxiv.org/html/2511.09299). Code: github.com/HelenaM23/RENTT, Python 3.10-3.12, `pip install -e .`, supports TF and PyTorch, compact (pruned) or complete trees; no license shown, 3 commits — [RENTT GitHub](https://github.com/HelenaM23/RENTT).
- Tree size grows exponentially with network size in naive conversion; efficient large-net implementation still missing — as summarized from [RENTT arXiv:2511.09299](https://arxiv.org/html/2511.09299) and [Aytekin](https://arxiv.org/pdf/2210.05189).
- Orfanos & Lelis, "Synthesizing Programmatic Policies with Actor-Critic Algorithms and ReLU Networks", arXiv:2308.02729 (2023): uses the ReLU-network / oblique-decision-tree connection to translate an actor-critic policy into a program with if-then-else, linear transformations of inputs and PID operations; "short and effective policies"; no code link; exactness not stated on abstract page — [arXiv:2308.02729](https://arxiv.org/abs/2308.02729). Its grammar is nested "if B then E else E" — [PDF](https://arxiv.org/pdf/2308.02729). (Closest prior work to "decompile a ReLU net into a short program".)
- Hanin & Rolnick, "Deep ReLU Networks Have Surprisingly Few Activation Patterns", NeurIPS 2019: average number of activation patterns at init is bounded by (total neurons)^(input dim), independent of depth, and tight at init and during training — [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2019/hash/9766527f2b5d3e95d4a733fcfb77bd7e-Abstract.html), [arXiv:1906.00904](https://arxiv.org/pdf/1906.00904).
- "Bounding and Counting Linear Regions of Deep Neural Networks" (arXiv:1711.02114): tighter upper/lower bounds on max linear regions of rectifier nets; upper bound exact for 1D inputs; uses MILP-based exact counting — [arXiv:1711.02114](https://arxiv.org/pdf/1711.02114.pdf), [summary](https://www.emergentmind.com/papers/1711.02114.md). Follow-up "Empirical Bounds on Linear Regions of Deep Rectifier Networks" (arXiv:1810.03370): MILP encoding, MIPBound probabilistic lower bound much faster than exact counting — [arXiv:1810.03370](https://arxiv.org/pdf/1810.03370).
- Counting linear regions is in general intractable (2025 complexity paper) — [arXiv:2505.16716](https://arxiv.org/html/2505.16716v1).
- "Parallel Algorithms for Exact Enumeration of Deep Neural Network Activation Regions" (arXiv:2403.00860, 2024) — finds all activation regions and sign vectors of a ReLU DNN — [arXiv HTML](https://arxiv.org/html/2403.00860v1); no library link found.
- `parx` (Polyhedral Affine Region eXplorer): `pip install parx`; Python 3.10+, Julia 1.10+ on PATH (juliacall, JuMP, HiGHS), PyTorch; exact enumeration (DFS + facet flip) and sparse forward-pass enumeration; per region outputs activation pattern, interior point, H-rep `D x <= g`, non-redundant facets, boundedness, and local affine map `f(x)=Ax+b`; point-to-region routing; sequential Linear->ReLU only; loads nn.Module/state_dict/.pth/.h5; MIT; alpha; 0 stars; Zenodo DOI 10.5281/zenodo.22696450 — [parx GitHub](https://github.com/Johanmkr/parx).
- SplineCam (Humayun, Balestriero, Balakrishnan, Baraniuk, CVPR 2023, pp. 3789-3798): exactly computes a CPWL network's partition/decision boundary on 2D input slices without sampling; supports ReLU, leaky ReLU, abs, maxout, max-pool; PyTorch + graph-tool — [CVPR open access](https://openaccess.thecvf.com/content/CVPR2023/html/Humayun_SplineCam_Exact_Visualization_and_Characterization_of_Deep_Network_Geometry_and_CVPR_2023_paper.html), [GitHub](https://github.com/ahmedimtiazprio/splinecam).
- Related equivalent representations cited by RENTT: Balestriero & Baraniuk 2020 (max-affine spline operators), Wang et al. 2019 (RNNs as piecewise affine splines), Sudjianto et al. 2020 and Chu et al. 2018 (local linear models) — [arXiv:2511.09299](https://arxiv.org/html/2511.09299).

### Inferences
- For 1-3 input MLPs, exact region enumeration is cheap: by Hanin & Rolnick the realized region count scales roughly as N^d (N neurons, d<=3), so a full region list plus affine maps is the natural "ground truth decompilation" baseline. `parx` produces exactly this artifact and is the most directly usable exact baseline (cost: Julia install). SplineCam covers 2D inputs exactly (and 1D/3D via slicing only), and needs graph-tool (conda-only on most platforms).
- The exact NN-to-tree papers (Aytekin, EC-DT, RENTT, Chang et al.) produce trees whose node tests are neuron sign tests, i.e. oblique hyperplanes in input space, with leaves = affine maps. These are exact but not "short readable programs": they encode the network's own neuron structure, not a minimal symbolic description. That distinguishes the project's goal (short programs, e.g. merging regions into a few formulas/splines) from this prior work.
- Of all prior work found, Orfanos & Lelis 2023 is closest to "ReLU net -> short if-then-else program", but for RL policies and without code.
- Attribution from memory (not confirmed by fetched pages): arXiv:1711.02114 is Serra, Tjandraatmadja & Ramalingam (ICML 2018); arXiv:1810.03370 is Serra & Ramalingam (AAAI 2020); Montufar, Pascanu, Cho & Bengio, "On the Number of Linear Regions of Deep Neural Networks", NeurIPS 2014.

### Gaps
- Montufar et al. 2014 not fetched; Serra et al. author lists not confirmed by fetched pages.
- RENTT's quantitative runtime/memory/tree-size numbers were not read (truncated); RENTT does not document exporting trees as code.
- No public code found for Aytekin 2022, Nguyen et al. 2020 EC-DT, Chang et al. 2023, or Orfanos & Lelis 2023 (searched arXiv pages and general web).
- Did not test parx installation on Windows.

## Q4. Compiling networks into branch code / trees / lookup tables for speed, and straight-line NN compilers as baselines

### Takeaway
I found no paper that converts a trained ReLU MLP into branching C code and benchmarks its inference speed against the straight-line matrix implementation; exact NN-to-tree work reports interpretability or verification gains, not CPU inference speedups. The nearest "compile NN to logic for speed" line is FPGA LUT-based nets (LogicNets 2020, NeuraLUT 2024), which need networks trained/quantized for it. For baselines: tree-to-C compilers (TL2cgen, successor of Treelite codegen; m2cgen; emlearn) for distilled trees, and onnx2c / TVM C target / emx-onnx-cgen for straight-line MLP C.

### Cited Findings
- No inference-speed comparison of the exact tree vs the NN in Chang et al. 2023 (only verification speedups 21x/2x) — [arXiv:2304.06049](https://arxiv.org/abs/2304.06049). Web search for ReLU->tree->if-else compilers returned papers giving the structure but no code generator — [search summary incl. arXiv:2308.02729, arXiv:2511.09299](https://arxiv.org/pdf/2308.02729).
- LogicNets (Xilinx, 2020): quantized neurons are equivalent to truth tables; networks trained with quantized I/O convert directly to a netlist of truth tables deployed as pipelined FPGA circuits; LUT cost grows exponentially with fan-in, so sparsity and low-bit activations are used — [arXiv:2004.03021](https://arxiv.org/pdf/2004.03021).
- NeuraLUT (Andronic & Constantinides, Imperial, 2024): maps whole sub-networks into a single LUT, so dense full-precision layers inside a partition cost nothing extra in table size — [arXiv:2403.00849](https://arxiv.org/pdf/2403.00849). Follow-ups: NeuraLUT-Assemble (2025) — [arXiv:2504.00592](https://arxiv.org/html/2504.00592v1); a 2026 paper notes exponential LUT size and sparse-connectivity as key challenges of LogicNets/PolyLUT/NeuraLUT — [arXiv:2601.09773](https://arxiv.org/html/2601.09773v1).
- TL2cgen (TreeLite 2 C GENerator): compiles decision-tree models (RF, GBM) to C99 (`tl2cgen.generate_c_code`, `export_lib`); since Treelite 4.0, C codegen moved out of Treelite into TL2cgen — [TL2cgen docs](https://tl2cgen.readthedocs.io), [migration notes](https://tl2cgen.readthedocs.io/en/latest/_sources/treelite-migration.rst.txt).
- m2cgen transpiles scikit-learn models to native code in several languages; a survey notes such general tools are not microcontroller-RAM-optimized — [IEEE CAS Magazine Q3 2020](https://nxtbook.com/nxtbooks/ieee/circuitsandsystems_2020q3/index.php?startid=10). emlearn targets any C99 device, no libc/dynamic allocation — [PyPI emlearn 0.12.0](https://pypi.org/project/emlearn/0.12.0/).
- onnx2c (kraiskil): ONNX -> single C file, compile-time buffers, no stdio/malloc, entry() function; optimizations: tensor unionization, Cast removal, AVR const placement; no unrolling or constant folding listed; 91/166 ONNX ops; one benchmark: sine model on STM32F411 ~20 us vs 490 us for STM32CubeAI; ~422 stars — [onnx2c GitHub](https://github.com/kraiskil/onnx2c).
- emmtrix emx-onnx-cgen: ONNX -> portable C, no dynamic memory/OS/runtime; v1.0.0 (Mar 2026) claims 100% coverage of ONNX opset 25 (vendor claim) — [emmtrix release](https://www.emmtrix.com/?p=31302), [emmtrix ONNX to generic C](https://www.emmtrix.com/news/onnx-to-generic-c-for-embedded-ai.html).
- TVM (Chen et al., OSDI 2018) is an end-to-end optimizing DL compiler — [arXiv:1802.04799](https://arxiv.org/pdf/1802.04799); its plain-C target (`Target("c")` + `export_library`) is used by some but forum threads question its maintenance status — [TVM forum](https://discuss.tvm.apache.org/t/whats-the-status-of-api-related-to-mlf-and-target-c/18336), [TVM forum C/ELF](https://discuss.tvm.apache.org/t/c-or-elf-generation/16054).
- OpTC (2024): toolchain for deploying NNs on AURIX TC3xx MCUs, generating C — [arXiv:2404.15833](https://arxiv.org/pdf/2404.15833).
- A Bernstein-polynomial "neurosymbolic compiler for edge deployment" (Bern2Edge) appeared in search results — [arXiv:2608.20497](https://arxiv.org/pdf/2608.20497) (not read; relevance unverified).

### Inferences
- The speed comparison the project plans (decompiled branch program in C vs straight-line compiled matrix code) appears to be largely unoccupied in the literature: exact NN-to-tree papers target interpretability/verification; LUT work targets FPGAs with co-designed quantized nets. This is a gap the project can claim, with appropriate hedging ("we found no prior benchmark").
- Recommended straight-line baselines for a tiny MLP: (1) hand-written / generated unrolled C with weights as `static const` and `-O3 -ffast-math` (the strongest baseline for tiny nets, since onnx2c does no unrolling/constant folding); (2) onnx2c output as an off-the-shelf tool; (3) optionally emlearn/m2cgen if the MLP is an sklearn `MLPRegressor`. TVM is heavy for 1-3 input nets and its C target maturity is uncertain.
- Recommended approximate-tree speed baseline: sklearn tree fitted to MLP outputs -> TL2cgen or m2cgen C. Recommended exact-region baseline: parx regions -> generated if-else over region H-reps or over neuron sign tests (a straightforward code generator the project would write itself).
- For small nets, branch-misprediction and the fact that a dense 3->H->H->1 MLP is only a few dozen FMAs mean straight-line code may win unless the decompiled program is much shorter; this is a hypothesis to measure, not a sourced finding.

### Gaps
- No source found that benchmarks CPU inference speed of an exact ReLU-to-tree/if-else compilation vs dense compiled MLP code.
- Did not find independent head-to-head benchmarks of onnx2c vs m2cgen vs emlearn vs TVM for small MLPs; onnx2c license not named on the README page fetched.
- Did not verify emlearn's MLP support details (page snippet truncated) or m2cgen's current maintenance status.

## Q5. Per-method summary (year, venue, exact/approx, pedagogical/decompositional, code, ease on a 1-3 input MLP)

### Takeaway
Easiest runnable baselines today: sklearn tree distillation (approx, pedagogical, trivial), parx (exact, decompositional, pip + Julia), SplineCam (exact, 2D, needs graph-tool). Classic rule extractors (ruleex, REMIX) are runnable only in old environments and are approximate classification-rule methods.

### Cited Findings
| Method | Year / venue | Exact? | Type | Code | Ease on small MLP |
|---|---|---|---|---|---|
| TREPAN (Craven & Shavlik) | 1996, NIPS | Approx | Pedagogical (queries oracle, builds tree) | None maintained found — [X-TREPAN](https://arxiv.org/pdf/1508.07551) | Re-implement or approximate with sklearn |
| DeepRED (Zilke et al.) | 2016, Discovery Science (LNCS) — [Springer](https://link.springer.com/doi/10.1007/978-3-319-46307-0_29) | Approx | Decompositional (layer-wise C4.5/C5.0) | ruleex (2019, MIT) — [PyPI](https://pypi.org/project/ruleex/); REMIX (GPL-3) — [GitHub](https://github.com/mateoespinosa/remix) | Old TF / Py<=3.8 + R; classification only |
| HypInv, ANN-DT | (in ruleex) | Approx | Pedagogical-ish | ruleex — [PyPI](https://pypi.org/project/ruleex/) | Same env issues |
| ECLAIRE | 2021, NeurIPS workshop — [arXiv](https://arxiv.org/pdf/2111.12628) | Approx | Decompositional | REMIX — [GitHub](https://github.com/mateoespinosa/remix) | Py 3.5-3.8 + R |
| PedC5.0 | (in REMIX) | Approx | Pedagogical | REMIX — [GitHub](https://github.com/mateoespinosa/remix) | As above; sklearn CART is a modern substitute |
| Soft decision tree (Frosst & Hinton) | 2017, arXiv | Approx (new model trained by distillation) | Pedagogical | Unofficial PyTorch: [xuyxu](https://awesome.ecosyste.ms/projects/github.com%2Fxuyxu%2Fsoft-decision-tree), [zeta1999](https://github.com/zeta1999/soft-decision-tree) | Easy, but not branchy code |
| EC-DT (Nguyen et al.) | 2020, arXiv — [arXiv](https://arxiv.org/abs/2003.04675) | Exact | Decompositional | None found | Re-implement |
| NNs are Decision Trees (Aytekin) | 2022, arXiv — [arXiv](https://arxiv.org/abs/2210.05189v1) | Exact | Decompositional | None found | Re-implement (simple) |
| Chang et al. NN controllers -> DT | 2023-25, arXiv — [arXiv](https://arxiv.org/abs/2304.06049) | Exact (pruned) | Decompositional | None found | Re-implement |
| Orfanos & Lelis programmatic policies | 2023, arXiv — [arXiv](https://arxiv.org/abs/2308.02729) | Not stated | Decompositional | None found | n/a |
| RENTT | 2025, arXiv — [arXiv](https://arxiv.org/html/2511.09299) | Exact | Decompositional | [GitHub](https://github.com/HelenaM23/RENTT) (research code, no license shown) | Py 3.10-3.12, TF/PyTorch; plausible |
| Region counting (MILP) | 2018 — [arXiv:1711.02114](https://arxiv.org/pdf/1711.02114.pdf) | Exact count | Decompositional | Not checked | n/a |
| Hanin & Rolnick | 2019, NeurIPS — [proc.](https://proceedings.neurips.cc/paper/2019/hash/9766527f2b5d3e95d4a733fcfb77bd7e-Abstract.html) | Theory | — | — | Justifies small region counts in low d |
| SplineCam | 2023, CVPR — [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Humayun_SplineCam_Exact_Visualization_and_Characterization_of_Deep_Network_Geometry_and_CVPR_2023_paper.html) | Exact (2D slices) | Decompositional | [GitHub](https://github.com/ahmedimtiazprio/splinecam) | Good for 2-input nets; graph-tool dependency |
| parx | ~2025-26, software (Zenodo) — [GitHub](https://github.com/Johanmkr/parx) | Exact (and sparse mode) | Decompositional | pip, MIT, alpha | Good fit: Linear->ReLU stacks, outputs regions + affine maps; needs Julia |
| LogicNets / NeuraLUT | 2020 / 2024 — [LogicNets](https://arxiv.org/pdf/2004.03021), [NeuraLUT](https://arxiv.org/pdf/2403.00849) | Exact for quantized nets (truth tables) | Decompositional, co-designed | Not checked | FPGA-oriented; requires quantized training |
| TL2cgen / m2cgen / emlearn | tree->C compilers — [TL2cgen](https://tl2cgen.readthedocs.io), [emlearn](https://pypi.org/project/emlearn/0.12.0/) | n/a | — | pip | Easy for compiling distilled trees |
| onnx2c / TVM / emx-onnx-cgen | NN->straight-line C — [onnx2c](https://github.com/kraiskil/onnx2c), [TVM](https://arxiv.org/pdf/1802.04799), [emmtrix](https://www.emmtrix.com/?p=31302) | Exact | — | open source | onnx2c easiest; TVM heavy |

### Inferences
- The project's niche: exact (or verified-tolerance) decompilation into *short* readable programs, plus C speed comparison. Prior exact methods yield neuron-structured trees (size ~ number of regions), prior short-program methods are approximate rule extractors; neither benchmarks CPU speed. parx + a custom if-else generator and sklearn-tree + TL2cgen give one exact and one approximate baseline with low effort.

### Gaps
- Venue/year of TREPAN (NIPS 1996) and Frosst & Hinton (arXiv 2017) are from general knowledge; their primary pages were not fetched in this session.
- parx's first release date not visible on the GitHub page fetched.
