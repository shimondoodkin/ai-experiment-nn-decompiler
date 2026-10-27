# LLMs that decompile, explain or translate trained neural networks into code or formulas (2023–2026)

Research date: 2026-10-27. About 17 search/fetch calls. Several items are dated 2026 and were found only through search snippets or arXiv abstract pages. Their details are thinner than for the 2023–2024 papers, and that is flagged where it applies.

## Which papers and projects use LLMs to explain or convert neural network internals into code or formulas?

### Takeaway
The LLM-based work falls into three groups:
- **LLMs that explain units in natural language.** Examples are OpenAI's neuron explainer and InterpAgent.
- **Agents that experiment on a black box.** Examples are FIND/AIA and MAIA. The agent probes the function and describes it in language or code.
- **LLM-guided symbolic regression from data.** Examples are LLM-SR and LaSR.

The closest non-LLM work turns weights into code with a fixed pipeline. Examples are MIPS (RNN to Python), the D-RASP transformer decompiler, Weights to Code (Discrete Transformer) and neural decompilers for Tracr. None of these has an LLM agent refactor a decompiled network.

### Cited Findings

**OpenAI, "Language models can explain neurons in language models" (Bills et al., May 9, 2023)**
- Authors are Bills, Cammarata, Mossing, Tillman, Gao, Goh, Sutskever, Leike, Wu and Saunders, all at OpenAI. Published May 9, 2023. GPT-4 explains the neurons. — [OpenAI paper page](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)
- The method has three stages: Explanation, then Simulation, then Scoring. It also covers ablation scoring and human scoring. The page notes that an "open source version" was released. — [OpenAI paper page](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)
- What the LLM sees: the fetched page section did not state this. As commonly described, GPT-4 sees text excerpts with per-token activations, not weights. This is behavioral (activation) data. Its output is a natural-language explanation, scored by how well a simulator reproduces the activations. (Inference, not verified in this session; see Gaps.)

**FIND: Function INterpretation and Description benchmark, with the Automated Interpretability Agent (AIA) (Schwettmann et al., NeurIPS 2023 Datasets & Benchmarks, arXiv 2309.03886)**
- FIND is a benchmark suite for testing the building blocks of automated interpretability methods. Its functions are built procedurally in textual and numeric domains. They include noise, composition, approximation and bias. — [NeurIPS 2023 poster](https://neurips.cc/virtual/2023/poster/73478); [DeepAI listing](https://deepai.org/publication/a-function-interpretation-benchmark-for-evaluating-interpretability-methods)
- The paper evaluates methods that "use language models (LMs) to produce code-based and language descriptions of function behavior." An off-the-shelf LM with only black-box access to the functions can sometimes infer their structure. — [DeepAI listing](https://deepai.org/publication/a-function-interpretation-benchmark-for-evaluating-interpretability-methods)
- Main limitation reported: LM descriptions tend to capture global function behavior and miss local corruptions. — [MIT news](https://computing.mit.edu/news/ai-agents-help-explain-other-ai-systems/)
- **Relevance:** FIND is the closest benchmark to this project's task style, which is recovering a numeric function as code. But the agent sees only the function's input/output behavior, never weights. Its targets are hand-written functions, not trained networks.

**MAIA: A Multimodal Automated Interpretability Agent (Rott Shaham, Schwettmann, Wang, Rajaram, Hernandez, Andreas, Torralba; ICML 2024, PMLR 235; arXiv 2404.14394)**
- MAIA gives a pretrained vision-language model tools for running experiments on parts of other models. The tools synthesize and edit inputs, find maximally activating exemplars and summarize results. MAIA forms hypotheses, runs experiments to test them, observes the outcomes and revises the hypotheses. It builds on the AIA paradigm from Schwettmann et al. 2023. — [MAIA project page](https://multimodal-interpretability.csail.mit.edu/maia); [ICML 2024](https://icml.cc/virtual/2024/poster/33183)
- One AI-generated secondary review names GPT-4V as the backbone. Treat that as unconfirmed. — [Moonlight review](https://www.themoonlight.io/review/a-multimodal-automated-interpretability-agent)
- What MAIA sees: inputs and activations from experiments. Its output is natural language.

**InterpAgent (arXiv 2605.01555, 2026)**
- A multi-agent framework with two loops. One refines explanations through competing hypotheses and tests. The other discovers features. It is also applied to raw MLP neuron activations in weight-sparse transformers, where it recovers neurons previously found by hand. — [arXiv 2605.01555](https://arxiv.org/html/2605.01555v1)
- What it sees: activations. Its output is natural language.

**LLM-SR (Shojaee, Meidani, Gupta, Barati Farimani, Reddy; ICLR 2025 oral; arXiv 2404.18400)**
- The LLM proposes equation skeletons as programs. Their parameters are fitted to data inside an evolutionary search over equation programs. It outperforms symbolic-regression baselines, especially out of domain. Official code is on GitHub (deep-symbolic-mathematics). — [ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/28df8e730c054c5331855fd4d5403ba9-Abstract-Conference.html); [arXiv](https://arxiv.org/abs/2404.18400v3)
- What it sees: data samples (x, y). Its output is a Python equation program. Verification is the fit error on the data.

**LaSR: Symbolic Regression with a Learned Concept Library (Grayeli, Sehgal, Costilla-Reyes, Cranmer, Chaudhuri; NeurIPS 2024; arXiv 2409.09359)**
- LaSR keeps a library of abstract text concepts that LLM queries evolve from high-performing hypotheses. It mixes standard evolutionary steps with LLM-guided ones and is validated on the Feynman equations. — [arXiv 2409.09359](https://arxiv.org/pdf/2409.09359); [NeurIPS 2024](https://nips.cc/virtual/2024/poster/96212)
- What it sees: data. Its output is symbolic formulas.
- LLM-SRBench (arXiv 2504.10415) benchmarks LLM equation-discovery methods and uses LaSR as a baseline. — [LLM-SRBench](https://arxiv.org/pdf/2504.10415)

**Symbolic Regression with Multimodal LLMs and KANs (Harvey, Ruehle, Fraser-Taliente, Halverson; arXiv 2505.07956, 2025)**
- Vision-capable LLMs propose formula ansätze for univariate functions. For the multivariate case, a KAN is trained first and the LLM fits the univariate function on each KAN edge. A language model then simplifies the combined expression. — [arXiv 2505.07956](https://arxiv.org/html/2505.07956v1); [IAIFI summary](https://research.iaifi.org/posts/symbolic-regression-with-multimodal-large-language-models-and-kolmogorov-arnold-)
- **Relevance:** the LLM looks at internal parts of a trained network (plots of KAN edge functions), not just end-to-end input/output behavior. This is one of the few cases where an LLM turns the inside of a trained network into a formula. It needs a KAN architecture, though, and the LLM sees plots, not weights as code.

**Interpretable Automated Mechanism Design with LLMs (arXiv 2502.12203, 2025)**
- LLM-driven evolutionary search produces a Python code segment. The code is an interpretable approximation of a trained black-box mechanism, RegretNet. — [arXiv 2502.12203](https://arxiv.org/pdf/2502.12203)
- What it sees: input/output behavior or the objective. The output is code that approximates the network. It is not decompiled from weights.

**Jane Street puzzle, "Can you reverse engineer our neural network?"**
- The network was hand-built with integer weights, so it was not trained. The solver treated it as an integer linear program and used a constraint solver. Gemini wrote an md5 implementation, which was matched against the neurons. This was a human-led effort with LLM help, not an autonomous agent. — [Jane Street blog](https://blog.janestreet.com/can-you-reverse-engineer-our-neural-network/)

**Non-LLM weights-to-code baselines (the nearest work on recovering algorithms)**
- **MIPS** (Michaud, Liao, Lad, Liu et al., MIT/IAIFI; arXiv 2402.05110 Feb 2024; journal version in *Entropy*, Dec 2024, DOI 10.3390/e26121046). It distills trained RNNs into Python. An integer autoencoder turns the RNN into a finite state machine, then Boolean or integer symbolic regression recovers the algorithm. On 62 algorithmic tasks, MIPS solves 32, including 13 that GPT-4 does not solve. GPT-4 solves 30. — [arXiv 2402.05110](https://arxiv.org/pdf/2402.05110); [Entropy/PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11727218/)
- **Discovering Interpretable Algorithms by Decompiling Transformers to RASP** (arXiv 2602.08857, ICML 2026). Small GPT-2-style transformers trained on algorithmic tasks are rewritten exactly into a D-RASP program, assuming a linearized LayerNorm. The program is then simplified in three ways: pruning parts that causal interventions show are irrelevant, replacing matrices with primitives from a library, and replacing MLPs with known primitives. The resulting programs keep more than 90% of the model's performance. Models that generalize to longer inputs can usually be decompiled; the others cannot. No LLM is used. Code: github.com/lacoco-lab/decompiling_transformers. — [Neuroexplicit blog, 2026-06-10](https://www.neuroexplicit.org/blog/2026-06-10-decompiling-transformers/)
- **Weights to Code: Extracting Interpretable Algorithms from the Discrete Transformer** (Zhang, Bi, Zhang, Jin, Fu, Jin; arXiv 2601.05770, Jan 2026, v3 May 2026). A custom Discrete Transformer uses temperature-annealed sampling. Hypothesis testing and symbolic regression then extract readable programs. It reports performance comparable to MIPS and extends to continuous intermediate computations. The abstract does not mention an LLM. — [arXiv 2601.05770](https://arxiv.org/abs/2601.05770)
- **Neural Decompiling of Tracr Transformers** (Thurnherr & Riesen, Univ. Bern; arXiv 2410.00061, 2024). A model is trained on pairs of Tracr-compiled weights and RASP source, and learns to map weights back to RASP. About 30% of outputs match exactly and more than 70% are functionally equivalent. — [arXiv 2410.00061](https://arxiv.org/pdf/2410.00061); [Alignment Forum post](https://www.alignmentforum.org/posts/ktCrb6utgsGLuBtNy/decompiling-tracr-transformers-an-interpretability), which reports 73% correct translation, likely measured differently.

### Inferences
- In this literature, LLMs explain networks almost entirely from behavior: activations, input/output probes or (x, y) data. The tools that read weights and emit code (MIPS, D-RASP decompiler, Weights to Code, Tracr decompilers) are fixed pipelines or trained decompilers, not LLM agents.
- This project combines both strands. A fixed decompilation into code (one line per neuron) feeds an LLM agent that refactors and abstracts it, checked against the net. Neither strand alone does this.

### Gaps
- The OpenAI 2023 methods text, which shows exactly what GPT-4 sees, was not fetched. The page reached had only the table of contents and credits. "Activations on text excerpts" is from background knowledge.
- Langosco et al. 2024, "Towards meta-models for automated interpretability" (a transformer meta-model reading Tracr weights and outputting RASP), and Friedman et al., "Learning Transformer Programs" (NeurIPS 2023), are known to the researcher but were not verified in this session.
- Code or pip availability was confirmed only for MIPS (github.com/ejmichaud/neural-verification), the D-RASP decompiler (lacoco-lab/decompiling_transformers) and LLM-SR (deep-symbolic-mathematics on GitHub). Nothing found is pip-installable as a "net to program" tool. pip availability was not checked systematically.
- arXiv 2603.20101, "Pitfalls in Evaluating Interpretability Agents", came up but its content was not retrieved.

## For each: year, venue, what the LLM sees, what it outputs, whether outputs are verified, and whether code is open

### Takeaway
Across the LLM-based work, the LLM sees behavior, not weights. Outputs are natural language (OpenAI, MAIA, InterpAgent) or code and formulas (FIND/AIA, LLM-SR, LaSR, mechanism design). Verification is statistical (simulation score, fit error, held-out accuracy), not exact equivalence with a specific trained network.

### Cited Findings
| Work | Year / venue | LLM sees | LLM outputs | Verified? | Code |
|---|---|---|---|---|---|
| OpenAI neuron explainer (Bills et al.) | 2023, OpenAI report | Activation records (method not fetched; see Gaps) | Natural language | Simulation and scoring, ablation, human scoring — [OpenAI](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html) | "Open source version" mentioned — same source |
| FIND + AIA (Schwettmann et al.) | 2023, NeurIPS D&B | Black-box input/output queries | Natural language and code descriptions | Scored against ground-truth functions — [NeurIPS](https://neurips.cc/virtual/2023/poster/73478) | Benchmark released (repo URL not fetched) |
| MAIA (Rott Shaham et al.) | 2024, ICML | Experiment results, exemplars, edited inputs | Natural language | Experimental evaluation — [MAIA](https://multimodal-interpretability.csail.mit.edu/maia) | Project page |
| LLM-SR (Shojaee et al.) | 2025, ICLR oral | (x, y) data | Python equation skeletons with fitted parameters | Data fit, out-of-domain tests — [ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/28df8e730c054c5331855fd4d5403ba9-Abstract-Conference.html) | GitHub |
| LaSR (Grayeli et al.) | 2024, NeurIPS | Data and concept library | Symbolic formulas | Feynman benchmark — [arXiv](https://arxiv.org/pdf/2409.09359) | Not checked |
| Multimodal LLM + KAN (Harvey et al.) | 2025, arXiv | Plots of trained KAN edge functions | Formulas | Fit — [arXiv](https://arxiv.org/html/2505.07956v1) | Not checked |
| InterpAgent | 2026, arXiv | Activations | Natural language | Hypothesis tests — [arXiv](https://arxiv.org/html/2605.01555v1) | Not checked |
| Weight Oracles (Kabra, Venhoff, Schroeder de Witt) | 2026, NeurIPS 2026 workshop (NeuralArtifacts) spotlight | Raw weights of small transformers | Forward-pass simulation; natural-language answers about anomalies and backdoors | 99% held-out simulation accuracy; AUROC 0.93 and 0.81 — [arXiv 2610.07334](https://arxiv.org/abs/2610.07334v1) | No repo listed |
| MIPS (GPT-4 Turbo baseline inside it) | 2024, arXiv / Entropy | GPT-4 sees two length-10 input/output lists | Formula, then Python | 100% accuracy on test data; Dafny for one program — [ar5iv](https://ar5iv.labs.arxiv.org/html/2402.05110) | github.com/ejmichaud/neural-verification |

- MIPS's GPT-4 Turbo baseline worked as follows. It was given two length-10 lists from the RNN training set and asked for a formula, then Python code, which it tested on those lists. It had three attempts at default temperature and counted as solved if any attempt reached 100% accuracy. An external script extracted and ran the code. — [ar5iv 2402.05110](https://ar5iv.labs.arxiv.org/html/2402.05110)

### Inferences
- MIPS's GPT-4 comparison is the closest published head-to-head of "LLM from input/output" against "algorithm from weights" on small trained networks. Each solved tasks the other missed (MIPS solved 13 that GPT-4 did not). That suggests giving the LLM the weights as code, as this project does, could combine both strengths. No source tests this directly.

### Gaps
- Repo URLs for FIND, MAIA and LaSR were not fetched.
- Weight Oracles' abstract page does not say whether weights are fed as text tokens or as continuous embeddings. One search snippet said "continuous tokens" — not confirmed from the page.

## Is there work where the LLM reads the network's weights directly (as code or text), not only input/output behavior?

### Takeaway
Very little, and none of it in this project's form. The only explicit "LLM reads raw weights" paper found is Weight Oracles (Oct 2026). It fine-tunes LMs to simulate small transformers and to detect backdoors, not to emit programs. The other weights-to-code systems do not use LLMs as the reader. No paper or repo was found in which an off-the-shelf LLM agent reads a decompiled trained MLP as source code and refactors it into a short verified program.

### Cited Findings
- **Weight Oracles** fine-tunes language models to "diagnose properties of a target network by reading its raw weights directly, without behavioural testing." Phase I: simulate small transformers' forward passes from weights, with 99% holdout accuracy. Phase II: answer natural-language diagnostic questions, such as backdoor detection, with AUROC 0.93 on attention-routed backdoors and 0.81 overall. It uses an external chain of computation that hands parameter-free operations to deterministic code. Scaling to realistic model sizes is the main open challenge. — [arXiv 2610.07334](https://arxiv.org/abs/2610.07334v1); [HTML](https://arxiv.org/html/2610.07334)
- Neural Decompiling of Tracr Transformers reads weights, but with a purpose-trained transformer, not a general LLM. It works only on compiled (Tracr) networks, not trained ones. — [arXiv 2410.00061](https://arxiv.org/pdf/2410.00061)
- The D-RASP decompiler and Weights to Code read trained weights but use deterministic rewriting, pruning, primitive matching and symbolic regression. No LLM is involved. — [Neuroexplicit blog](https://www.neuroexplicit.org/blog/2026-06-10-decompiling-transformers/); [arXiv 2601.05770](https://arxiv.org/abs/2601.05770)
- Harvey et al. feed an LLM plots of a trained KAN's per-edge learned functions. That is internal structure, but as images of 1-D functions, not weights as code. — [arXiv 2505.07956](https://arxiv.org/html/2505.07956v1)
- In the Jane Street puzzle, humans read the weights and an LLM helped only by writing a reference implementation (md5) to compare against. — [Jane Street blog](https://blog.janestreet.com/can-you-reverse-engineer-our-neural-network/)
- A broad search for LLM agents reverse-engineering network weights into Python returned no such agent. The nearest results were behavioral distillation and DNN-executable decompilation (NeuroDeX, arXiv 2509.06402, which decompiles compiled DNN binaries — a different problem). — [NeuroDeX](https://arxiv.org/pdf/2509.06402)

### Inferences
- This project's setup appears to be new: a trained ReLU MLP dumped as plain Python, one line per neuron, and handed to a general-purpose coding agent (Claude Code) to refactor blind, with no task name. The nearest neighbors are:
  1. MIPS, which does weights to Python with snapping but no LLM.
  2. The D-RASP decompiler, which does exact reparametrization then simplification, again with no LLM.
  3. Weight Oracles, where an LLM reads weights but outputs simulations and diagnoses, not code.
  4. FIND/AIA, where an LLM writes code describing a function, but from black-box queries only.
- Distilling a deterministic tool from the LLM's notes, as this project's step 3 did, also has no direct match in the sources found.

### Gaps
- Unreviewed blog posts, LessWrong/Alignment Forum posts or GitHub experiments may exist where someone pasted a small network's weights into ChatGPT or Claude. Searches did not surface any. This absence is a weak signal, not proof.
- Weight Oracles' input format (text versus embeddings) is unconfirmed.

## Is there anything specifically about recovering exact algorithms and "snapping" constants with an LLM?

### Takeaway
Snapping learned constants to "intended" values is established in non-LLM pipelines. MIPS rounds weights within 0.01 of an integer and uses an integer/lattice autoencoder. KAN's symbolic step replaces learned splines with closed-form functions. LLM-SR fits constants to data rather than snapping them. No source was found where an LLM decides which learned constants to snap (e.g., 0.198 to 0.2) while checking equivalence against the original network.

### Cited Findings
- MIPS's quantization normalizer rounds weights and biases within ε = 0.01 of an integer to that integer. Its integer autoencoder maps hidden states to integer tuples using a GCD-based lattice finder (with noise thresholds) or a linear lattice finder, then fits symbolic formulas to the resulting lookup tables. Hidden-state clusters are assigned to bitstrings. — [ar5iv 2402.05110](https://ar5iv.labs.arxiv.org/html/2402.05110)
- MIPS networks are RNNs whose transition and output are ReLU MLPs with a linear final layer. Search covered hidden sizes 1–128, widths up to 256 and depths 1–3. Sixteen tasks were solved with hidden size 1. Only the binary-addition program was formally verified (Dafny, with manual annotations). The others were checked as 100% accurate on test data. — [ar5iv 2402.05110](https://ar5iv.labs.arxiv.org/html/2402.05110)
- In the D-RASP decompiler, the rewrite is exact by construction under a linearized LayerNorm. Simplification swaps matrices and MLPs for library primitives — a structural analogue of snapping — and the result keeps more than 90% of performance, not exact equivalence. — [Neuroexplicit blog](https://www.neuroexplicit.org/blog/2026-06-10-decompiling-transformers/)
- LLM-SR has the LLM propose program skeletons whose numeric parameters are then optimized against data. Constants are fitted, not snapped to round values. — [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/28df8e730c054c5331855fd4d5403ba9-Abstract-Conference.html)
- A 2026 KAN paper (Sovrano et al.) notes that each KAN edge can in principle be replaced by a symbolic operator. It reports that greedy in-context symbolic regression reduces median OFAT test MSE by up to 99.8%. — [arXiv 2603.15250](https://arxiv.org/html/2603.15250v1) (search snippet only; not fetched in full)

### Inferences
- This project's main differences from the nearest work are:
  - An LLM, not a fixed threshold, chooses snap targets semantically (e.g., 0.2 as a tolerance, tax-bracket edges). This lets it recover non-integer "intended" constants that MIPS's integer rounding would miss.
  - A checker verifies the result against the net itself, rather than against task test data.
  - It targets continuous-input feed-forward ReLU MLPs, not RNNs over discrete or integer sequences (MIPS) or transformers (RASP work).
- MIPS's GPT-4 Turbo baseline is the strongest published evidence that LLMs alone, from input/output data, recover many but not all small-network algorithms (30 of 62).

### Gaps
- No source on LLM-driven constant snapping was found. The pykan `auto_symbolic` snapping procedure was not confirmed from a primary source in this session.
- Whether Weights to Code snaps constants is not stated in its abstract; the full text was not read.
