# Mechanistic interpretability work that recovers exact algorithms or programs from small trained networks

Scope note: the repo being positioned decompiles small trained ReLU MLPs (1–9 inputs, 1–2 hidden layers; XOR, |a−b|, median-of-3, tax brackets) into short readable programs, snaps learned constants to intended values, and verifies the program against the network. Everything below is judged against that. Research date: 2026-10-27. Items from 2024–2026 are marked **[2024]**, **[2025]**, **[2026]**.

## Q1. Grokking / modular-addition algorithm recovery (Nanda et al.) and follow-ups, including "The Clock and the Pizza"

### Takeaway
This line of work recovers *one specific named algorithm* (Fourier "clock", "pizza") from small transformers or MLPs trained on modular arithmetic. It does this by hand-guided analysis, and the output is a prose and mathematical description, not an automatically emitted program. It shows that small networks implement clean, recoverable algorithms, and that the same task can produce *different* algorithms depending on seed and hyperparameters. It does not offer a general, automated decompiler.

### Cited Findings
- Nanda, Chan, Lieberum, Smith, Steinhardt reverse-engineer small transformers trained on modular addition. The learned algorithm "uses discrete Fourier transforms and trigonometric identities to turn addition into rotation around a circle." They define progress measures that split training into memorization, circuit formation and cleanup. — [arXiv 2301.05217](https://arxiv.org/abs/2301.05217)
- The arXiv page for Nanda et al. lists no venue (it was v3 in Oct 2023) and no code link. The ICLR 2023 venue comes from the task brief and was not confirmed from a primary page in this session. — [arXiv 2301.05217](https://arxiv.org/abs/2301.05217)
- "The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks" (Zhong, Liu, Tegmark, Andreas) appeared at NeurIPS 2023 (oral) and is arXiv 2306.17844. Small changes to hyperparameters or initialization lead networks to learn different modular-addition algorithms: the known "Clock" algorithm, a new "Pizza" algorithm, more complex variants, and sometimes several imperfect Pizza copies running in parallel. — [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/56cbfbf49937a0873d451343ddc8c57d-Abstract.html); [arXiv 2306.17844](https://arxiv.org/abs/2306.17844)
- Code for Clock/Pizza is at https://github.com/fjzzq2002/pizza. — [search result summarizing the paper](https://arxiv.org/abs/2306.17844)
- Gromov (2023), "Grokking modular arithmetic": two-layer fully connected networks (MLPs) trained with vanilla GD. The paper gives *analytic* weight expressions that solve a large class of modular arithmetic tasks and argues that trained networks find these feature maps. — [arXiv 2301.02679](https://arxiv.org/pdf/2301.02679)
- A follow-up by Doshi et al. ("Grokking Modular Polynomials") **[2024]** reportedly extends the analytic solutions to modular multiplication and general modular polynomials, and proposes a learnable vs. non-learnable classification. That classification is supported empirically but not proved. This comes from a secondary review site, not the primary paper. — [Moonlight review](https://www.themoonlight.io/en/review/grokking-modular-polynomials)
- Gross, Agrawal, Kwa et al., "Compact Proofs of Model Performance via Mechanistic Interpretability" **[2024]** (NeurIPS 2024), uses reverse-engineered mechanisms to *formally prove* accuracy lower bounds for 151 small transformers on Max-of-K, using 102 computer-assisted proof strategies. Shorter proofs need more mechanistic understanding. The main obstacle is "compounding structureless noise". The models are one-layer, attention-only transformers. — [arXiv 2406.11779](https://arxiv.org/abs/2406.11779v8); [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/90e73f3cf1a6c84c723a2e8b7fb2b2c1-Abstract.html); [Alignment Forum post](https://alignmentforum.org/posts/bRsKimQcPTX3tNNJZ/compact-proofs-of-model-performance-via-mechanistic)
- "Validating Mechanistic Interpretations: An Axiomatic Approach" **[2024, ICML 2025 per search snippet]** frames a mechanistic interpretation as a circuit that acts like a program over symbolic features. It names small models on algorithmic tasks such as modular addition as the most productive testbeds. — [arXiv 2407.13594](https://arxiv.org/pdf/2407.13594)

### Inferences
- Grokking work sets a precedent for "small network → exact algorithm". It is manual, one task at a time, and the output is a mathematical narrative such as "embed as cos/sin of k·a and use trig identities". It is not code, and it does not snap constants to intended values.
- Clock vs. Pizza matters for the repo: one task can admit several valid recovered programs. A decompiler that outputs one short program is choosing among functionally equivalent implementations. Verification against the network, rather than matching an expected program, is the right success criterion, and the repo already does this.
- The Compact Proofs line is the closest analogue to the repo's "verified against the network" step. Its verification is a proven bound on accuracy over a finite input domain, not equivalence of a program.

### Gaps
- I did not confirm the Nanda et al. model size or code repository from a primary page. The arXiv abstract page states neither. My recollection is that it was a 1-layer transformer with p=113 and a GitHub repo under "mechanistic-interpretability-grokking", but this was not verified, so I have not cited it.
- I did not retrieve Chughtai, Chan, Nanda, "A Toy Model of Universality" (ICML 2023, group operations) in this session.

## Q2. Compiled transformers and program extraction: Tracr, Learning Transformer Programs, RASP, and "decompile transformer to RASP"

### Takeaway
The "transformer ↔ RASP program" line has run in both directions. Tracr compiles programs into weights (2023). Learning Transformer Programs trains constrained transformers that convert *by construction* into discrete Python programs (2023). Neural Decompiling of Tracr Transformers learns a decompiler for Tracr-compiled weights (2024). Huang, Bakalova, Bhattamishra, Merrill, Hahn (ICML 2026) decompile *ordinary trained* small GPT-2-style transformers into D-RASP and prune them with causal interventions **[2026]**. That last paper is the transformer-side work closest in spirit to the repo.

### Cited Findings
- **Tracr** (Lindner, Kramár, Farquhar, Rahtz, McGrath, Mikulik; NeurIPS 2023 main track) compiles human-readable RASP programs into decoder-only transformers with known structure, as ground truth for evaluating interpretability methods. Examples include token frequency counting, sorting and parenthesis checking. Code: https://github.com/google-deepmind/tracr. — [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/771155abaae744e08576f1f3b4b7ac0d-Abstract.html)
- **Learning Transformer Programs** (Friedman, Wettig, Chen; NeurIPS 2023, oral per the virtual-site listing) trains a constrained, interpretable transformer with gradient descent and then converts it to a discrete, human-readable program, building on RASP. Code is in the princeton-nlp GitHub org as "TransformerPrograms". The repo was not opened. — [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/995f693b73050f90977ed2828202645c-Abstract.html); [arXiv 2306.01128](https://arxiv.org/abs/2306.01128v1); [NeurIPS oral page](https://neurips.cc/virtual/2023/oral/73853)
- **Neural Decompiling of Tracr Transformers** (Thurnherr, Riesen, Univ. Bern; arXiv 2410.00061, Sept 2024) **[2024]** generates (weights, RASP program) pairs with Tracr and trains a decompiler model on them. It reproduces about 30% of test programs exactly, and over 70% of its outputs are functionally equivalent. It works only on *Tracr-compiled* weights, not on SGD-trained ones. — [arXiv 2410.00061](https://arxiv.org/html/2410.00061v1); [Alignment Forum post](https://www.alignmentforum.org/posts/ktCrb6utgsGLuBtNy/decompiling-tracr-transformers-an-interpretability)
- **Discovering Interpretable Algorithms by Decompiling Transformers to RASP** (Huang, Bakalova, Bhattamishra, Merrill, Hahn; arXiv 2602.08857, v1 Feb 9 2026, v2 Jun 5 2026; "Accepted for publication at ICML 2026") **[2026]** re-parameterizes a trained transformer as a RASP program and then uses causal interventions to find a small sufficient sub-program. On small transformers trained on algorithmic and formal-language tasks, it "often recovers simple, interpretable programs" from models that generalize to longer inputs. — [arXiv 2602.08857](https://arxiv.org/abs/2602.08857)
  - The project blog gives these details. The dialect is D-RASP. Under a LayerNorm linearization assumption, the conversion of a GPT-2-style transformer "exactly preserves its input-output behavior". Pruning removes causally irrelevant components, simplifies matrices by matching library primitives, and replaces MLPs with known primitives where possible. Tasks are most-frequent character, copying unique tokens, sorting integers and Dyck-4. The pruned programs keep ">90%" of the original model's performance, so they are approximate, not exact. Code: https://github.com/lacoco-lab/decompiling_transformers. — [Neuroexplicit blog, 2026-06-10](https://www.neuroexplicit.org/blog/2026-06-10-decompiling-transformers/)
  - Venue conflict: an ICLR 2026 virtual-site listing with the same title also appeared in search ([iclr.cc](https://www.iclr.cc/virtual/2026/10016648)). The arXiv comment and the blog both say ICML 2026. It may be an ICLR workshop or an earlier submission. Unresolved.
- **Weights to Code: Extracting Interpretable Algorithms from the Discrete Transformer** (Zhang, Bi, Zhang, Jin, Fu, Jin; arXiv 2601.05770, v1 Jan 9 2026, v3 May 29 2026) **[2026]** proposes a "Discrete Transformer" that uses temperature-annealed sampling to fight superposition, so that hypothesis testing and symbolic regression can recover human-readable programs. It reports results comparable to the RNN-based MIPS baseline on shared discrete tasks and extends extraction to tasks with continuous-valued intermediates. No code link was found and no peer-reviewed venue is listed. — [arXiv 2601.05770](https://arxiv.org/abs/2601.05770)

### Inferences
- Learning Transformer Programs and Weights-to-Code both get programs by *changing the architecture or training* so that the network is discrete or programmable by construction. The repo decompiles *standard* ReLU MLPs after the fact, which is a different and harder setting. That is a defensible distinguishing point.
- Huang et al. (2026) is the closest transformer analogue: post-hoc on ordinary trained models, an exact reparameterization step followed by simplification to a short program. Its residual programs are approximate (>90%), while the repo emphasizes verification against the network. A comparison should note that the two differ in architecture (transformer vs MLP) and in output language (RASP vs general imperative code with snapped constants).
- None of the RASP work targets continuous-valued regression functions with meaningful constants (tax brackets, |a−b|). The repo's constant-snapping on real-valued outputs appears uncovered by this line.

### Gaps
- I did not read the full Huang et al. 2026 paper. I do not know their model parameter counts, how much of the pipeline is automatic, or whether they snap numeric constants.
- I did not verify the TransformerPrograms repository contents or licence.
- I did not check whether Weights-to-Code has a code release on an author GitHub.

## Q3. Automated circuit discovery (ACDC) and sparse autoencoders, insofar as they produce programs

### Takeaway
ACDC and SAE-based methods output *subgraphs* or *feature dictionaries*, not programs. A human still has to turn the circuit into an algorithm. They are adjacent tooling and should be cited as such, not as prior program-extraction work. OpenAI's weight-sparse circuits work (2025) aims for more tractable circuits but, as far as I saw, does not emit programs.

### Cited Findings
- ACDC (Conmy, Mavor-Parker, Lynch, Heimersheim, Garriga-Alonso; NeurIPS 2023 Spotlight) automates finding the circuit in a model's computational graph. It recovered the five component types of the GPT-2 Small Greater-Than circuit, selecting 68 of about 32,000 edges, all found earlier by manual analysis. The output is a subgraph of the computational graph. Code: https://github.com/ArthurConmy/Automatic-Circuit-Discovery. — [arXiv 2304.14997](https://arxiv.org/abs/2304.14997)
- OpenAI, "Understanding neural networks through sparse circuits" **[2025, Nov]**, pursues making networks easier to disentangle and trace through sparsity. From the snippet, it does not appear to extract executable programs. — [OpenAI](https://openai.com/index/understanding-neural-networks-through-sparse-circuits/)

### Inferences
- In the repo's terms, ACDC is closest to the "which hidden units matter" pruning step, not to program emission. SAEs target superposition in large models and are mostly irrelevant to networks with 1–2 hidden layers of a few units.

### Gaps
- I did not research SAE-to-program pipelines in depth. I found no source claiming that SAEs emit executable programs.

## Q4. "Neural networks are decision trees" (Aytekin 2022), exact ReLU-MLP-to-rule conversion, and tiny-MLP/toy-model interpretation

### Takeaway
Exact piecewise-linear structure in ReLU MLPs is well known. Aytekin (2022) shows any piecewise-linear network equals a decision tree. A separate literature, including Rolnick & Kording (ICML 2020), recovers ReLU network *weights and architecture* from queries. Neither produces short, readable, human-intended programs with snapped constants. The exact tree or region enumeration is a faithful but large and unreadable representation. I found no primary work that targets "decompile a tiny trained ReLU MLP into a short intended program" as its main goal, apart from MIPS (Q5), where the MLPs sit inside RNNs.

### Cited Findings
- Aytekin, "Neural Networks are Decision Trees" (arXiv 2210.05189, three versions Oct 11–25 2022) claims that any network with piecewise-linear activations can be represented *exactly* as a decision tree, a direct conversion rather than a distillation. Later versions narrow the computational-advantage claims to small networks, and the abstract wording varies between versions. — [arXiv 2210.05189](https://arxiv.org/abs/2210.05189v1); [ar5iv](https://ar5iv.labs.arxiv.org/html/2210.05189)
- Public discussion has questioned the novelty of Aytekin's claim, for example a podcast segment asking whether it is new and noting heavy social-media hype. I found no formal published rebuttal. — [player.fm podcast listing](https://sv.player.fm/1BSzscV)
- Rolnick & Kording, "Reverse-Engineering Deep ReLU Networks" (ICML 2020, PMLR v119), use the fact that linear regions change where a neuron switches on or off to recover architecture, weights and biases from input/output queries. This recovers *parameters*, not an algorithm. — [PMLR](https://proceedings.mlr.press/v119/rolnick20a.html)
- An optimization-based follow-up (arXiv 2312.04675) summarizes the Rolnick-Kording assumptions (noiseless outputs, integer weights per its summary) and cryptanalytic parameter extraction (Carlini et al. 2020). — [arXiv 2312.04675](https://arxiv.org/pdf/2312.04675)
- Gromov (2023) gives analytic weights for 2-layer MLPs on modular arithmetic (see Q1). It is the clearest "tiny MLP, exact algorithm" case in the grokking literature, but the algorithm is derived analytically rather than decompiled from arbitrary trained weights. — [arXiv 2301.02679](https://arxiv.org/pdf/2301.02679)
- KANs (Liu et al., arXiv 2404.19756) **[2024]** include an explicit "snap to symbolic" step. The pykan workflow is train, prune, `auto_symbolic()` to fix each activation to a symbolic function, retrain, then `symbolic_formula()`. This is the closest analogue to the repo's constant and function snapping, but it needs the KAN architecture, not a standard MLP. — [pykan docs](https://docsearch.algolia.com/mcp/docs/repo/kindxiaoming/pykan); [arXiv 2404.19756](https://alphaxiv.org/abs/2404.19756v4)
- Follow-ups criticize post-hoc KAN symbolification. S2KAN **[2025]** says trained activations often learn "pathological decompositions" with no correspondence to interpretable forms, and builds symbolic primitives into training with learnable gates. Symbolic-KAN **[2026]** uses discrete primitive selection to avoid post-hoc fitting. — [arXiv 2512.07875](https://arxiv.org/html/2512.07875v1); [arXiv 2603.23854](https://www.alphaxiv.org/abs/2603.23854.md)

### Inferences
- The repo can cite Aytekin and the region-enumeration view as the "exact but unreadable" baseline. Its contribution is choosing a *short* program among exact or near-exact descriptions and snapping constants to intended values, which the decision-tree view does not do.
- The KAN symbolification critiques ("pathological decompositions") parallel a risk for the repo. Post-hoc snapping on a standard MLP may find a network whose internal decomposition does not match the intended program even when the input/output behavior does. Verifying against the network, plus a held-out evaluation, is the honest answer to that risk.
- I found no prior work centred on interpreting tiny MLPs on everyday functions (XOR, |a−b|, median, tax brackets). This looks like an open niche, but the absence of evidence comes from a limited search.

### Gaps
- I did not find primary sources on mechanistic interpretation of tiny MLPs on XOR, abs or median specifically, such as LessWrong toy-model posts. The search returned mostly parameter-extraction work.
- I did not verify that the KAN paper was published at ICLR 2025 (claimed in the brief; not confirmed by the sources found).

## Q5. MIPS: "Opening the AI black box: program synthesis via mechanistic interpretability" (Michaud et al. 2024), in detail

### Takeaway
MIPS **[2024]** is the closest prior work. It is a fully automated pipeline that trains the smallest RNN able to solve a task, normalizes its weights, discretizes its hidden states to bits or an integer lattice, runs Boolean or integer symbolic regression (an AI-Feynman-style brute-force search), and emits Python. It also snaps near-integer weights. It solved 32 of 62 algorithmic tasks, against 30 for GPT-4 Turbo, and released code. Its domain is sequence tasks with integer or bit inputs and RNNs with MLP sub-blocks. It does not handle continuous-valued functions with real constants well, and it lists "continuous computation" as a failure mode.

### Cited Findings
- Authors: Eric J. Michaud, Isaac Liao, Vedang Lad, Ziming Liu, Anish Mudide, Chloe Loughridge, Zifan Carl Guo, Tara Rezaei Kheirkhah, Mateja Vukelić, Max Tegmark. arXiv 2402.05110, v1 Feb 7 2024, 24 pages. The arXiv page lists no venue. — [arXiv 2402.05110](https://arxiv.org/abs/2402.05110)
- Abstract claims: MIPS "auto-distills the learned algorithm into Python code". On 62 algorithmic tasks learnable by an RNN, MIPS solves 32, including 13 that GPT-4 does not solve. GPT-4 solves 30. It does not rely on human-written training data. — [arXiv 2402.05110](https://arxiv.org/abs/2402.05110)
- **Architecture:** RNN with h_i = f(h_{i−1}, x_i) and y_i = g(h_i). f and g are ReLU MLPs with a linear final layer, and h_0 = 0. The architecture is set by (n, w_f, d_f, w_g, d_g). — [arXiv HTML](https://arxiv.org/html/2402.05110v1)
- **AutoML minimal-size search:** n ∈ {1..128}, widths ∈ {1..256}, depths ∈ {1,2,3}, ranked so that smaller d_g, then d_f, n, w_g, w_f are preferred. Each candidate is trained with 5 seeds (Adam, lr 1e-3, batch 4096, 10k–20k steps, loss ½·log[1+(x−y)²]). A binary search finds the smallest configuration reaching 100% test accuracy. Sixteen tasks need only n = 1, and the largest is n = 81 (Dithering). — [arXiv HTML](https://arxiv.org/html/2402.05110v1)
- **Normalizers, in order:** whitening, then Jordan normal form of the recurrence matrix (with a unitary step to remove complex numbers), then Toeplitz transforms within Jordan blocks to simplify the input matrix, then de-biasing, then **quantization that snaps weights within 0.01 of an integer to that integer**. — [arXiv HTML](https://arxiv.org/html/2402.05110v1)
- **Discretization ("integer autoencoder"):** for bits, hidden states form 2^b clusters, and the method keeps the cluster-to-bitstring assignment that gives the shortest program. For integers, hidden states are treated as an affine-deformed, possibly sparse integer lattice. A "GCD lattice finder" recovers the basis from pairwise separations, with a noise tolerance ε_gcd and a minimum-description-length choice. If f is affine, a linear lattice finder reads the basis from the weights directly. — [arXiv HTML](https://arxiv.org/html/2402.05110v1)
- **Symbolic regression:** Boolean functions use DNF, or a function of the bit-sum if symmetric, keeping the shortest. Integer functions use linear regression with rounded coefficients if linear, else a brute-force search adapted from AI Feynman over expressions of up to 6 symbols (6 unary and 4 binary operators), checked on 100 points, with a fallback if nothing fits. The formulas are inserted into a template Python program around the RNN loop. — [arXiv HTML](https://arxiv.org/html/2402.05110v1)
- **Verification:** generated programs are checked against benchmark data, and only tasks where the RNN hit 100% accuracy are used. Formal verification is shown once: binary addition, translated to Dafny and checked for length-10 bitvectors. The authors say this should be automated in future. — [arXiv HTML](https://arxiv.org/html/2402.05110v1)
- **Inputs and tasks:** integer lists of length 10 or 20, with one or two lists per task. Bits are integers in {0,1}. Examples: Binary_Addition, Bitwise_Xor/Or/And/Not, Parity_Last2–4, Sum_Last2–7, Prev1–5, Abs_Current, Majority_0_1–3, Div_3/5/7, Dithering, Newton_Gravity/Spring/Magnetic. — [arXiv HTML](https://arxiv.org/html/2402.05110v1)
- **Failure modes:** (1) noise and nonlinearity break the lattice autoencoder, for example Diff_Abs_Values. (2) RNNs use continuous values, which breaks the finite-state assumption, for example the Majority_0_X tasks. — [arXiv HTML](https://arxiv.org/html/2402.05110v1)
- **Code:** the paper gives https://github.com/ejmichaud/neural-verification for the benchmark and code. The arXiv abstract page itself has no repository link. — [arXiv HTML](https://arxiv.org/html/2402.05110v1)
- **Successor:** "Weights to Code" **[2026]** uses MIPS as its baseline and reports comparable performance on shared discrete tasks with a Discrete Transformer, extending to continuous intermediates. — [arXiv 2601.05770](https://arxiv.org/abs/2601.05770)

### Inferences
- **Overlap with the repo:** MIPS also (a) targets small ReLU MLPs, as sub-blocks of an RNN, (b) is automated end to end, (c) snaps near-integer weights (0.01 tolerance), (d) prefers the shortest program (MDL), and (e) emits Python. The repo should cite MIPS as the primary prior work and not claim novelty on "automated weights to program for small nets" or "integer snapping".
- **Differences the repo can honestly claim**, if they hold in the repo's own results:
  - Feed-forward, real-valued functions with *non-integer intended constants* (tax brackets, thresholds) versus MIPS's integer and bit sequence tasks.
  - Piecewise-linear and branch structure (median, |a−b|, brackets) expressed as conditionals rather than lattice and FSM formulas. MIPS lists Diff_Abs_Values and continuous computation as failures, which is close to the repo's |a−b| task, though MIPS's version is a sequence task.
  - Decompiling a *given* trained network rather than co-designing the smallest net via AutoML. MIPS chooses architecture size to make extraction easy.
  - Systematic verification against the network itself, rather than only against benchmark data, plus a held-out task run.
- The repo should mention that MIPS reported 32/62 against GPT-4's 30/62 as a reference for typical success rates in this genre. The repo's 21/21 in-sample and 5/6 held-out should be framed with its much smaller task count.

### Gaps
- I did not inspect the neural-verification GitHub repo's licence, maintenance or runnability.
- I found no peer-reviewed venue for MIPS. It appears to remain an arXiv preprint (single v1).
- The precise per-task success list (which 32) was not extracted.

## Q6. Summary table: year, venue, network scale, automation, output form, code

### Takeaway
Only MIPS (2024), Weights-to-Code (2026) and Huang et al. (2026) are *automated* post-hoc extractors of readable programs from trained networks. MIPS is the only one built on ReLU MLPs, and none targets feed-forward real-valued functions with snapped non-integer constants.

### Cited Findings
| Work | Year / venue | Networks & scale | Automation | Output form | Code |
|---|---|---|---|---|---|
| Nanda et al., Progress measures for grokking | 2023; arXiv 2301.05217 (ICLR 2023 per brief, unverified here) | small transformer, modular addition ([arXiv](https://arxiv.org/abs/2301.05217)) | manual | math/prose algorithm (DFT + trig identities) | not on arXiv page |
| Gromov, Grokking modular arithmetic | 2023; arXiv | 2-layer MLP ([arXiv](https://arxiv.org/pdf/2301.02679)) | analytic | closed-form weights | n/a |
| Zhong et al., Clock and Pizza | 2023; NeurIPS ([proc.](https://proceedings.neurips.cc/paper_files/paper/2023/hash/56cbfbf49937a0873d451343ddc8c57d-Abstract.html)) | small transformers, modular addition | manual plus metrics | named algorithms (Clock/Pizza) | github.com/fjzzq2002/pizza |
| Tracr | 2023; NeurIPS ([proc.](https://proceedings.neurips.cc/paper_files/paper/2023/hash/771155abaae744e08576f1f3b4b7ac0d-Abstract.html)) | compiled transformers | automated *compiler* (program to weights) | weights | github.com/google-deepmind/tracr |
| Learning Transformer Programs | 2023; NeurIPS ([proc.](https://proceedings.neurips.cc/paper_files/paper/2023/hash/995f693b73050f90977ed2828202645c-Abstract.html)) | constrained transformers | automated (by construction) | discrete Python/RASP-like program | princeton-nlp/TransformerPrograms |
| ACDC | 2023; NeurIPS spotlight ([arXiv](https://arxiv.org/abs/2304.14997)) | GPT-2 Small etc. | automated | circuit subgraph | ArthurConmy/Automatic-Circuit-Discovery |
| MIPS **[2024]** | 2024; arXiv 2402.05110 ([arXiv](https://arxiv.org/abs/2402.05110)) | RNN with ReLU-MLP f,g; hidden n=1..81 | fully automated | Python program | ejmichaud/neural-verification |
| KAN **[2024]** | 2024; arXiv 2404.19756 | KAN architecture | semi-automated snap-to-symbolic | symbolic formula | pykan ([docs](https://docsearch.algolia.com/mcp/docs/repo/kindxiaoming/pykan)) |
| Compact Proofs **[2024]** | 2024; NeurIPS ([proc.](https://proceedings.neurips.cc/paper_files/paper/2024/hash/90e73f3cf1a6c84c723a2e8b7fb2b2c1-Abstract.html)) | 151 one-layer attn-only transformers, Max-of-K | computer-assisted | formal accuracy bounds | not checked |
| Neural Decompiling of Tracr **[2024]** | 2024; arXiv 2410.00061 ([arXiv](https://arxiv.org/html/2410.00061v1)) | Tracr-compiled transformers only | automated (learned decompiler) | RASP; 30% exact, >70% functionally equivalent | not checked |
| Weights to Code **[2026]** | 2026; arXiv 2601.05770 ([arXiv](https://arxiv.org/abs/2601.05770)) | Discrete Transformer (modified arch) | automated (hypothesis tests + symbolic regression) | programs | none found |
| Huang et al., Decompiling Transformers to RASP **[2026]** | 2026; ICML 2026 per arXiv ([arXiv](https://arxiv.org/abs/2602.08857)) | small GPT-2-style transformers | automated reparam + causal pruning | D-RASP; >90% fidelity after pruning ([blog](https://www.neuroexplicit.org/blog/2026-06-10-decompiling-transformers/)) | lacoco-lab/decompiling_transformers |
| Aytekin, NNs are Decision Trees | 2022; arXiv ([arXiv](https://arxiv.org/abs/2210.05189v1)) | any piecewise-linear NN | automated (exact conversion) | decision tree (large) | not checked |
| Rolnick & Kording | 2020; ICML ([PMLR](https://proceedings.mlr.press/v119/rolnick20a.html)) | deep ReLU nets | automated | recovered weights/architecture | not checked |

### Inferences
- For an honest positioning statement, the repo is "MIPS-style automated decompilation, applied to feed-forward ReLU MLPs on real-valued, piecewise tasks, with snapping to intended (not just integer) constants and verification against the network." Its prior-work section should list MIPS first, then KAN symbolification as the snapping analogue, then Huang et al. 2026 as the transformer-side counterpart, then the grokking case studies as manual precedent.

### Gaps
- Code availability for Compact Proofs, Neural Decompiling of Tracr, and Aytekin was not checked.
- Recent 2025–2026 LessWrong/Alignment Forum toy-MLP writeups were not surveyed, because the search surfaced none.
