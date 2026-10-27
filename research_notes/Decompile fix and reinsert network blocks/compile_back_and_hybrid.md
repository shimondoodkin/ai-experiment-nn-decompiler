# Putting a decompiled/fixed program back: compiling into weights (a) and hybrid neural/CPU-code inference (b)

Verification legend: items marked **[verified]** were checked this session against a primary page or search result. Items marked **[unverified]** are from background knowledge and were NOT fetched this session; the report writer should flag them or check them before citing.

## Q1. Compiling programs into network weights (exact vs approximate)

### Takeaway
There is a mature line of "program -> transformer weights" compilers (RASP -> Tracr -> ALTA, plus the reverse direction, Learning Transformer Programs) and an older line of "rules/decision trees -> MLP" constructions (KBANN, entropy nets, DJINN, neural random forests). For ReLU MLPs, compilation is exact only for continuous piecewise-linear (CPWL) programs (relu/min/max/abs/linear). Hard if/else thresholds are discontinuous, so a ReLU net can only reproduce them exactly outside a small margin around the threshold. Tracr makes the same split: categorical maps compile to exact lookup tables, numerical maps to approximations by discretisation.

### Cited Findings
**Theory: what ReLU nets can represent exactly**
- A function R^n -> R is computable by a ReLU network **exactly iff it is continuous piecewise linear**. Every CPWL function on R^n needs depth at most ceil(log2(n+1))+1, and for R -> R one hidden layer suffices, with tight size bounds. Arora, Basu, Mianjy, Mukherjee, "Understanding Deep Neural Networks with Rectified Linear Units", ICLR 2018 **[verified]**. [arXiv 1611.01491](https://arxiv.org/abs/1611.01491v3); [ICLR poster](https://iclr.cc/virtual/2018/poster/155)

**Transformer compilers**
- **RASP**: Weiss, Goldberg, Yahav, "Thinking Like Transformers", ICML 2021. It is a sequence-processing language whose primitives map to attention (select/aggregate) and MLP (elementwise map). Code: github.com/tech-srl/RASP (an interpreter, not a weight compiler). **[unverified this session]** [arXiv 2106.06981](https://arxiv.org/abs/2106.06981)
- **Tracr**: Lindner, Kramár, Farquhar, Rahtz, McGrath, Mikulik. NeurIPS 2023 Spotlight. It compiles RASP into the weights of a standard decoder-only transformer. Code: https://github.com/google-deepmind/tracr (JAX/Haiku) **[verified]**. [arXiv 2301.05062](https://arxiv.org/abs/2301.05062)
- Tracr MLP compilation: categorical Maps "act as a lookup table", i.e. exact. Numerical MLPs are built as "universal function approximators": the first layer discretises the input and the second maps each bucket to an output, with discretisation points chosen around known input values. Attention uses a large inverse temperature (default 100), so it is near-hard but not exactly hard **[verified]**. [ar5iv Tracr](https://ar5iv.labs.arxiv.org/html/2301.05062)
- Tracr's stated limitations:
  - All variables sit in orthogonal residual subspaces, and the hand-coded matrices are "both unrealistic and inefficient".
  - The authors say "we will likely never compile fully featured language models in Tracr".
  - Gradient-descent compression of the residual stream works, e.g. frac_prevs goes from D=14 to d=6 "without hurting performance". However, the compressed sort_unique model reaches near-perfect accuracy with only about 0.8 cosine similarity to the compiled one, encoding a variable numerically instead of one-hot. So retraining can preserve behaviour while changing the mechanism **[verified]**. [ar5iv Tracr](https://ar5iv.labs.arxiv.org/html/2301.05062)
- **ALTA**: Shaw, Cohan, Eisenstein, Lee, Berant, Toutanova. TMLR 2025; earlier at the NeurIPS 2024 Sys2-Reasoning workshop. It provides a symbolic interpreter plus a compiler from ALTA programs to Transformer weights, and is inspired by RASP/Tracr. Code: https://github.com/google-deepmind/alta **[verified]**. [arXiv 2410.18077](https://arxiv.org/abs/2410.18077v2); [TMLR listing](https://mlanthology.org/tmlr/2025/shaw2025tmlr-alta/)
- **Learning Transformer Programs**: Friedman, Wettig, Chen. NeurIPS 2023 oral. This is the reverse direction: it trains constrained transformers that map deterministically to Python, which can then be debugged with standard Python tools. Code: princeton-nlp/TransformerPrograms (repo named in the paper; README not opened) **[verified: paper, venue]**. [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/995f693b73050f90977ed2828202645c-Abstract.html); [arXiv 2306.01128](https://arxiv.org/abs/2306.01128v1)
- **Percepta, "Can LLMs Be Computers?"** (blog, March 2026, Christos Tzamos's team):
  - It analytically compiles an interpreter into transformer weights (no gradient descent). The program is supplied as input and executed token by token in the forward pass.
  - Secondary sources report >30k tokens/s on CPU, a correct Sudoku solver and Hungarian-algorithm matching. The repo is "transformer-vm".
  - **[blog and secondary coverage only, not peer reviewed]** [getmaxim summary](https://www.getmaxim.ai/blog/computation-beyond-tool-use-executing-programs-inside-a-transformer/); [Spivack commentary](https://www.novaspivack.com/business/the-internalization-of-computation-what-perceptas-transformer-computer-actually-signals-and-why-the-implications-are-far-stranger-than-the-demo); [repo listing](https://gittrend.io/repo/Percepta-Core/transformer-vm)
- Looped Transformers as Programmable Computers: Giannou et al., ICML 2023. Hard-codes weights so that a looped transformer emulates a simple instruction-set computer **[unverified this session]**. arXiv 2301.13196

**Rules / decision trees -> MLP (classic exact or near-exact conversions)**
- **KBANN**: Towell & Shavlik, "Knowledge-Based Artificial Neural Networks", Artificial Intelligence 69/70, 1994; Towell PhD thesis 1991 "Symbolic Knowledge and Neural Networks: Insertion, Refinement and Extraction". It has three stages:
  1. Approximately-correct rules set the structure and initial weights.
  2. Standard backprop refines them.
  3. Rules are extracted again with the NOFM algorithm ("Extracting Refined Rules from Knowledge-Based Neural Networks", Machine Learning 13, 1993). The extracted rules "closely reproduce the accuracy of the network".

  This is exactly the user's insert/refine/extract cycle, done with propositional rules **[verified]**. [Towell & Shavlik AIJ 1994 PDF](https://cs400-web.cs.wisc.edu/nchou/common/site/p/ml/ftp/towell.aij94.pdf); [thesis abstract](https://pages.cs.wisc.edu/~shavlik/abstracts/towell.thesis.abstract.html); [MLJ 1993](https://mlanthology.org/mlj/1993/towell1993mlj-extracting)
- **Entropy nets**: Sethi, "Entropy nets: from decision trees to neural networks", Proc. IEEE 78(10), 1990. It maps a tree to a 3-layer net: a node layer for split hyperplanes, an AND layer for leaves/paths and an OR layer for classes, using threshold units, then retrains **[unverified this session]**.
- **Neural random forests**: Biau, Scornet, Welbl, Sankhya A, 2019. It recasts each tree of a random forest as a 2-hidden-layer net (split layer + leaf layer) and relaxes the hard thresholds to tanh before training **[unverified this session]**. [arXiv 1604.07143](https://arxiv.org/abs/1604.07143)
- **DJINN**: Humbird, Peterson, McClarren, "Deep neural network initialization with decision trees". The arXiv preprint dates to 2017; a journal version appeared in IEEE TNNLS 2019 **[journal unverified]**.
  - Steps: (1) build a tree or ensemble, (2) map it to network architecture and initial weights, (3) fine-tune with backprop.
  - The mapping uses tree paths "as guidance" and does not reproduce the tree exactly. The networks start near tree performance and end above it.
  - Code: https://github.com/LLNL/DJINN (TensorFlow) and pip `djinnml` **[verified]**. [arXiv 1707.00784](https://arxiv.org/pdf/1707.00784); [GitHub](https://github.com/LLNL/DJINN); [PyPI](https://pypi.org/project/djinnml/)
- **Hummingbird** (Microsoft): Nakandala et al., "A Tensor Compiler for Unified Machine Learning Prediction Serving", OSDI 2020.
  - Compiles trained trees, ensembles and featurizers to tensor ops (PyTorch/TorchScript/ONNX), with one strategy implementing tree traversal as GEMM. This is exact, but it is a tensor program, not a ReLU MLP.
  - One third-party note says traversal is level-by-level, so cost grows linearly with depth and deep trees run slower than XGBoost **[verified]**. [USENIX](https://www.usenix.org/conference/osdi20/presentation/nakandala); [MSR](https://www.microsoft.com/en-us/research/publication/a-tensor-compiler-for-unified-machine-learning-prediction-serving/); [PyPI](https://pypi.org/project/hummingbird-ml/0.0.6); [OctoML note](https://octo.ai/blog/compiling-classical-ml-for-performance-gains-up-to-30x-and-hardware-portability)
- "CompiledNN" (B-Human) is a JIT compiler for running NN inference fast on CPU. It is **not** a program-to-network compiler **[unverified this session]**. github.com/bhuman/CompiledNN

### Inferences
- These are standard identities, derived here and not taken from one cited source:
  - max(a,b) = a + relu(b-a)
  - min(a,b) = a - relu(a-b)
  - |x| = relu(x) + relu(-x)
  - linear tests are linear layers

  So a decompiled program using **only** relu/min/max/abs/linear compiles back to a ReLU MLP **exactly**, as Arora et al.'s CPWL characterisation guarantees.
- `if w.x > t: A else B` is exact only when both branches agree at the boundary, i.e. the result is continuous. In that case it is a CPWL function: for example out = B + (A-B)*... is CPWL only when written as min/max combinations.
- A genuine jump needs a steep ramp: step(z) ≈ relu(z/ε) - relu(z/ε - 1). This is exact for z <= 0 and z >= ε, and wrong only on the margin (0, ε). Exactness can therefore be achieved on a finite or quantised input set, or wherever the data has a margin around t. That matches how Tracr treats numerical MLPs (discretise near known input values) and why it uses a large attention temperature.
- With a margin, steep slopes (1/ε) inflate weight norms, which may hurt later fine-tuning. A Tracr fine-tuning study found compiled weights unstable to train (see Q3).
- KBANN is the closest classic precedent for the whole loop: extract readable rules, edit them, reinsert them as weights, train, and re-extract.

### Gaps
- I did not fetch Sethi 1990, Biau et al. 2019 or RASP this session. Their constructions are described from memory, and the exact layer counts and thresholds should be checked.
- Tracr's numerical-MLP bucket count and hidden size were not stated in the fetched text.
- I found no paper that specifically benchmarks exact-vs-margin compilation of if/else into ReLU MLPs as a fidelity study.

## Q2. Replacing components of trained networks with code, interpretable approximations, or lookup tables (hybrid inference)

### Takeaway
Interpretability "replacement" work (transcoders, cross-layer transcoders) swaps MLPs for wider sparse MLPs and reports a sizeable fidelity gap. In Anthropic's 2025 work the replacement matches the top-1 next token only about 50% of the time unless per-prompt error terms are added. Systems that splice compiled or symbolic modules into a working model and keep or improve accuracy exist mainly for narrow symbolic subtasks (Neural Comprehension/CoNN) or via architectural lookup tables (MoLE).

### Cited Findings
- **Transcoders**: Dunefsky, Chlenski, Nanda, NeurIPS 2024.
  - A wide, sparse one-hidden-layer ReLU MLP is trained to approximate an MLP sublayer's input-output map, with an L1 penalty. It performs at least on par with SAEs on sparsity, faithfulness and interpretability across 120M-1.4B models.
  - The paper applies it to the GPT-2-small greater-than circuit.
  - Code: https://github.com/jacobdunefsky/transcoder_circuits **[verified]**. [NeurIPS poster](https://neurips.cc/virtual/2024/poster/95736); [arXiv 2406.11944](https://www.arxiv.org/abs/2406.11944)
- **Anthropic, "Circuit Tracing: Revealing Computational Graphs in Language Models"** (Transformer Circuits, 2025):
  - The "replacement model" swaps MLPs for a cross-layer transcoder (CLT). The largest 18-layer CLT **matches the underlying model's next-token completion on ~50%** of diverse pretraining-style prompts.
  - Normalized mean reconstruction error is ~11.5% at L0≈88 (18L, 10M features) and 21.7% at L0≈235 (Claude 3.5 Haiku, 30M features).
  - Exact agreement on a given prompt needs a "local replacement model". It adds per-(token, layer) error terms equal to true MLP output minus CLT output and freezes attention patterns and normalisation denominators. The authors caution that the replacement may use different mechanisms.
  - Released: the attribution-graph frontend, github.com/anthropics/attribution-graphs-frontend **[verified]**. [Methods page](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)
  - A separate open-source "circuit-tracer" library (Anthropic + Decode, May 2025) for open-weight models **[unverified this session]**.
  - Critique: LessWrong argues CLTs are incentivised to learn unfaithful mechanisms. [LessWrong](https://www.lesswrong.com/posts/6CS2NDmoLCFcEJMor/cross-layer-transcoders-are-incentivized-to-learn-unfaithful)
- **Neural Comprehension / CoNN**: Weng et al., ICLR 2024. The arXiv title is now "Mastering Symbolic Operations: Augmenting Language Models with Compiled Neural Networks".
  - Tracr-compiled "Compiled Neural Networks" with hand-set attention weights for rules (copy, add-with-carry, etc.) are converted to PyTorch and combined with a pretrained LM. Each generated token comes from either the LM or a CoNN.
  - This is the closest precedent for splicing a compiled program module into a trained model.
  - Code: github.com/WENGSYX/Neural-Comprehension, plus Hugging Face CoNN checkpoints **[verified]**. [arXiv 2304.01665](https://arxiv.org/abs/2304.01665v3); [ICLR 2024](https://iclr.cc/virtual/2024/poster/19283); [HF model card](https://huggingface-gw-kz.just-ai.com/WENGSYX/CoNN_Add_Carry)
- **Mixture of Lookup Experts (MoLE)**: Jie et al., ICML 2025 oral, PMLR 267.
  - Experts are FFNs during training but take embedding-layer output as input. Before inference they are converted into **lookup tables** keyed by token ID and offloaded to storage, so inference is only a retrieval.
  - Reported speed is comparable to dense models and much faster than MoE with expert offloading, at matched FLOPs and VRAM, with MoE-level quality.
  - Code: github.com/JieShibo/MoLE **[verified]**. [PMLR](https://proceedings.mlr.press/v267/jie25b.html); [ICML](https://icml.cc/virtual/2025/poster/43620)
- Older differentiable-program hybrids, all **[unverified this session]**:
  - Neural Programmer (Neelakantan et al., ICLR 2016)
  - Neural Programmer-Interpreters (Reed & de Freitas, ICLR 2016)
  - Differentiable Forth / ∂4 (Bošnjak et al., ICML 2017)
  - TerpreT (Gaunt et al., 2016)
  - Neural Arithmetic Logic Units (Trask et al., NeurIPS 2018)

  These embed discrete operations or interpreters inside a differentiable model, but none splices extracted code back into a pretrained net.
- KAN (Liu et al., ICLR 2025, github KindXiaoming/pykan) lets per-edge learned activations be "snapped" to symbolic formulas and the remaining affine parameters retrained. This is a symbolic-replace-then-fine-tune workflow within one architecture **[unverified this session]**.

### Inferences
- Replacing an MLP with something it was not trained jointly with (e.g. a transcoder) costs noticeable fidelity. For the user's case of a small block, exact decompilation plus a verified fix should not have that gap on the block's own I/O. The literature's gap comes from approximate replacements.
- Option (b), code on CPU in place of the block, has few direct precedents for *extracted* code. The nearest are CoNN (a compiled module routed to at token level), MoLE (a precomputed table off-GPU) and tool use. The user's idea is a fairly unoccupied niche, but this rests on an absence of hits, which is weak evidence.

### Gaps
- I did not find a paper that decompiles a block of a *trained* network into code, edits it and runs it on CPU inside the original model, reporting end-to-end accuracy.
- I did not get CE-loss-recovered numbers for the transcoders paper.

## Q3. Fine-tuning after reinsertion (compiled or hand-written module spliced in, then training)

### Takeaway
Reinsert-then-train is established for rules and trees (KBANN 1991-94, entropy nets, DJINN, neural random forests). For compiled transformers there is evidence it works, with caveats. Tracr weights are sparse and train unstably unless small noise and SGD are used. Retraining can keep accuracy while changing the internal mechanism, which matters if the goal is to keep the fixed algorithm intact.

### Cited Findings
- **Fine-tuning a Tracr-compiled model**: Jain et al., "Mechanistically analyzing the effects of fine-tuning on procedurally defined tasks" (arXiv 2311.12786; ICLR 2024 venue **unverified**).
  - Tracr gives "a rather sparse parameterization, which often yields unstable training dynamics (e.g., gradient spikes)", especially with adaptive optimizers.
  - Fix: add small Gaussian noise to the compiled weights, and use SGD with momentum at learning rates from 1e-1 down to 1e-4 **[verified via search snippet of the paper]**. [arXiv 2311.12786](https://arxiv.org/pdf/2311.12786)
- In the Tracr paper's own experiments, gradient-descent compression keeps accuracy, but in one case the trained model diverges in representation (cosine ≈0.8, numeric instead of one-hot) **[verified]**. [ar5iv Tracr](https://ar5iv.labs.arxiv.org/html/2301.05062)
- **DJINN**: tree-to-net initialisation followed by backprop fine-tuning beats the original tree **[verified]**. [arXiv 1707.00784](https://arxiv.org/pdf/1707.00784)
- **KBANN**: rule-initialised networks refined by backprop "generalize better to future examples than do standard networks", and refined rules can be re-extracted **[verified]**. [AIJ 1994](https://cs400-web.cs.wisc.edu/nchou/common/site/p/ml/ftp/towell.aij94.pdf)
- InterpBench (Gupta et al., NeurIPS 2024 Datasets & Benchmarks) trains realistic transformers to implement Tracr circuits via Strict Interchange Intervention Training **[unverified this session]**.

### Inferences
- For the user's plan (a), compile the fixed program exactly and freeze it, or fine-tune only the surrounding layers. If the block itself is fine-tuned, use a small LR, SGD and a little noise, and re-decompile afterwards to check that the fix survived. The Tracr compression result shows accuracy can be kept while the mechanism drifts.

### Gaps
- I did not find a paper that freezes a compiled or hand-written block inside a pretrained network and fine-tunes only the rest, reporting recovery curves.

## Q4. Speed: replacing small MLP blocks with branchy CPU code, trees or lookup tables

### Takeaway
Conditional or tree-structured and table-based execution gives large CPU speedups in reported work:
- UltraFastBERT: about 78x on the FFN layer, using 12 of 4095 neurons.
- Logic-gate networks: over 1M MNIST images/s on one core.
- MoLE: dense-level speed with experts as tables.

Speedups are layer-level and implementation-dependent. Tree-as-tensor compilers like Hummingbird can be slower than native tree code for deep trees.

### Cited Findings
- **Fast Feedforward / UltraFastBERT**: Belcak & Wattenhofer (ETH), "Exponentially Faster Language Modelling", arXiv 2311.10770.
  - FFF neurons form a balanced binary tree, and only one root-to-leaf path is executed. That is 12 of 4095 neurons per layer (about 0.3%).
  - High-level CPU code reaches **78x** over an optimized baseline FFN, and the PyTorch implementation about 40x. The authors say this is not fully efficient.
  - It keeps at least 96.0% of BERT-base GLUE performance.
  - Code and weights released (github.com/pbelcak/UltraFastBERT) **[verified: claims; repo URL from memory]**. [arXiv 2311.10770](https://arxiv.org/abs/2311.10770v2); [HF papers](https://huggingface.co/papers/2311.10770); [Synced](https://syncedreview.com/2023/11/24/eth-zurichs-ultrafastbert-realizes-78x-speedup-for-language-models/)
  - The earlier FFF paper is "Fast Feedforward Networks", arXiv 2308.14711 **[unverified]**.
- **Deep Differentiable Logic Gate Networks**: Petersen et al., NeurIPS 2022. Relaxed logic gates are trained by gradient descent, then discretised. They run at over 1M MNIST images/s on a single CPU core with more than 97.5% accuracy. Code: github.com/Felix-Petersen/difflogic **[repo unverified]**. [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0d3496dd0cec77a999c98d35003203ca-Abstract.html); [arXiv 2210.08277](https://arxiv.org/pdf/2210.08277)
- **MoLE**: lookup-table experts on storage, at dense-model speed **[verified]**. [ICML 2025](https://icml.cc/virtual/2025/poster/43620)
- **Hummingbird**: trees run as GEMM tensors. The MSR summary says it is competitive with hand-crafted kernels, but the OctoML note reports it is slower than XGBoost for deep trees **[verified]**. [MSR](https://www.microsoft.com/en-us/research/publication/a-tensor-compiler-for-unified-machine-learning-prediction-serving/); [OctoML](https://octo.ai/blog/compiling-classical-ml-for-performance-gains-up-to-30x-and-hardware-portability)
- Other LUT-inference work, all **[unverified this session]**:
  - MADDNESS (Blalock & Guttag, ICML 2021): approximate matmul by hashing and table lookup, reporting about 100x over exact matmul at some accuracy cost
  - LUT-NN (MobiCom 2023)
  - T-MAC (Microsoft, EuroSys 2025): LUT kernels for low-bit LLMs on CPU, github microsoft/T-MAC
  - LogicNets / PolyLUT (FPGA)

### Inferences
- For a tiny decompiled block (a few relu/min/max/if ops), straight-line or branchy CPU code is effectively a hand-written FFF path. The big CPU wins reported above come from skipping most neurons.
- The win depends on batch size and data movement. A host round-trip from GPU to CPU per layer would likely erase gains unless the whole model runs on CPU. No source here measures that splice cost, so this is an inference.

### Gaps
- I found no published benchmark of "decompiled MLP block as CPU code vs the original block" latency.
- I did not retrieve Hummingbird's headline speedup figures or the MADDNESS numbers.
