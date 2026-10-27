# Neural network repair and model editing from failing inputs

Scope: methods that take a trained network plus inputs on which it is wrong, change part of it (weights, one layer, some neurons, an MLP block) so those inputs become correct, and try to keep other behaviour. Compared at the end with the repo's idea: decompile one block to Python, have an LLM agent fix the algorithm using failing cases, reinsert as CPU code or recompiled weights.

Verification note: items marked (V) were confirmed by a search or fetch in this session. Items marked (U) come from the researcher's prior knowledge and were NOT re-checked this session (mostly GitHub URLs and arXiv IDs). The report writer should treat (U) items as needing a check.

## Q1. Provable / verified DNN repair (PRDNN, REASSURE, APRNN, NNrepair, Goldberger et al.): what do they edit and what do they guarantee?

### Takeaway
This family edits weights only. They either solve a linear program over one layer (PRDNN, APRNN), add a local "patch network" per linear region (REASSURE), or solve constraints on weights picked by fault localization (NNrepair, Goldberger et al.). In exchange they guarantee that the repaired net provably meets the spec on the given failing points or on whole input polytopes. PRDNN and REASSURE also give minimality, and REASSURE gives locality. They are the closest prior art to "recompile the fixed block back into weights", and they report the same three numbers the repo would need: efficacy on the repair set, drawdown on held-out data, and generalization to unseen similar failures.

### Cited Findings
- **PRDNN, "Provable Repair of Deep Neural Networks"** (Sotoudeh and Thakur, PLDI 2021, arXiv 2104.04413) (V):
  - Its point repair handles a finite set of points and works with any activation function. Its polytope repair handles infinitely many inputs and needs piecewise-linear activations. — [arXiv 2104.04413](https://arxiv.org/abs/2104.04413v2); [PLDI 2021 page](https://pldi21.sigplan.org/details/pldi-2021-papers/39/Provable-Repair-of-Deep-Neural-Networks)
  - Its key device is the Decoupled DNN (DDNN). Any DNN converts trivially to an equivalent DDNN, and repairing a single layer of a DDNN becomes a linear program. Polytope repair reduces exactly to point repair on the vertices of the linear regions. — [arXiv 2104.04413](https://arxiv.org/pdf/2104.04413)
  - Guarantee: if the LP is feasible, the repaired net satisfies the spec and the change is a minimal single-layer repair. If it is infeasible, no single-layer DDNN repair exists for that layer. — [arXiv 2104.04413](https://arxiv.org/pdf/2104.04413)
  - What it edits: the parameters of one chosen layer, inside a decoupled copy of the network. A side effect is that the architecture changes, because the DDNN has two weight channels; APRNN was later motivated by this. — [APRNN page](https://thakur.cs.ucdavis.edu/bibliography/PLDI2023.html)
  - Code: github.com/95616ARG/PRDNN (U on exact URL; the search summary only confirmed that an official implementation named PRDNN is on GitHub). — [Thakur bibliography](https://thakur.cs.ucdavis.edu/bibliography/PLDI2021.html)
- **APRNN, "Architecture-Preserving Provable Repair of DNNs"** (Tao, Nawas, Mitchell and Thakur, PLDI 2023, PACMPL vol. 7, art. 124, DOI 10.1145/3591238) (V):
  - It modifies only the parameters and keeps the architecture. It repairs finitely many points or entire polytopes. It supports activations with linear pieces and fully-connected, convolutional, pooling and residual layers. — [Thakur PLDI 2023 page](https://thakur.cs.ucdavis.edu/bibliography/PLDI2023.html); [PLDI 2023 talk page](https://pldi23.sigplan.org/details/pldi-2023-pldi/19/Architecture-Preserving-Provable-Repair-of-Deep-Neural-Networks)
  - Code: [github.com/95616ARG/APRNN](https://github.com/95616ARG/APRNN). It ships with Docker, needs a Gurobi license, and compares against PRDNN and REASSURE on MNIST. Its result tables report drawdown and generalization. (V)
  - The arXiv listing 2304.03496 wrongly says "PLDI 2021"; the DOI and conference page say 2023. — [arXiv 2304.03496](https://arxiv.org/abs/2304.03496v1)
- **REASSURE, "Sound and Complete Neural Network Repair with Minimality and Locality Guarantees"** (Feisi Fu and Wenchao Li, Boston University, ICLR 2022 rather than 2023, arXiv 2110.07682) (V):
  - Method: it exploits the continuous piecewise-linear property of ReLU nets to synthesize a patch network tailored to the linear region of each buggy input. So it edits by adding a region-gated patch, not by changing the existing weights. — [arXiv 2110.07682](https://arxiv.org/pdf/2110.07682); [ICLR 2022 poster](https://iclr.cc/virtual/2022/poster/6686)
  - Guarantees: it is sound and complete, and outputs on inputs away from the repair region are unchanged (locality). It comes with a minimality lower bound and terminates in polynomial time in the network size and the number of buggy linear regions. — [arXiv 2110.07682](https://arxiv.org/pdf/2110.07682)
  - Code: GitHub BU-DEPEND-Lab/REASSURE. — [search summary of arXiv/ICLR](https://arxiv.org/abs/2110.07682v2)
- **NNrepair, "Constraint-based Repair of Neural Network Classifiers"** (Usman, Gopinath, Sun, Noller and Pasareanu, arXiv 2103.12535; Springer LNCS 978-3-030-81685-8, which is CAV 2021 by the researcher's knowledge, but the venue was not confirmed in-session) (V/U):
  - Method: fault localization first flags suspicious neurons in a layer and a subset of their incoming weights. Constraint solving then finds small changes to those weights. — [arXiv 2103.12535](https://arxiv.org/pdf/2103.12535)
  - Tooling, per a secondary review: it translates the Keras model to Java and uses Symbolic PathFinder concolic execution plus z3. It does not use MILP; the MILP/Gurobi formulation belongs to the separate QNNRepair (arXiv 2306.13793). — [velog review](https://velog.io/@thdalwh3867/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-NNrepair-Constraint-based-Repair-of-Neural-Network-Classifiers); [QNNRepair](https://arxiv.org/pdf/2306.13793)
  - Results: accuracy rises by 45.56 percentage points on poisoned data and 10.40 points on adversarial data, on MNIST and CIFAR-10. — [arXiv 2103.12535](https://arxiv.org/pdf/2103.12535)
  - Public code URL: not found in-session.
- **Goldberger, Adi, Keshet and Katz, "Minimal Modifications of Deep Neural Networks using Verification"** (LPAR-23, 2020, EPiC vol. 73, pp. 260–278, DOI 10.29007/699q) (V):
  - It uses DNN verification to change a trained net so that it meets requirements, with a provably minimal change and no retraining. It changes weights in a single layer; multi-layer changes are left as future work, since the problem is non-linear and non-convex. The backends are Marabou and Gurobi. Its use cases are watermark-removal resilience and repairing bugs. — [EasyChair](https://easychair.org/publications/paper/CWhF)
  - Code: github.com/jjgold012/MinimalDNNModificationMarabou (U).

### Inferences
- The guarantee that every provable method gives is that the network is correct on the stated failing set, or on the polytope. None of them guarantees correct behaviour on unseen failing inputs; generalization is measured empirically. That is the same epistemic position as an LLM fix to decompiled code, which can be proven correct on the failing set by execution but only tested on held-out data.
- PRDNN's single-layer LP repair is the closest formal analogue of "recompile the fixed block into weights". If the repaired code is a fixed function on a finite set of points, or on a ReLU polytope, PRDNN/APRNN can project that target behaviour back into one layer's weights with minimality. A hybrid is therefore possible: the LLM proposes the corrected function, and PRDNN/APRNN writes it into the weights as a provable point or polytope repair.
- REASSURE's patch network is in effect an "if input is in region R then add correction" program. That is structurally close to a code-level patch, for example an inserted `if` branch in decompiled ReLU code that covers one activation pattern.

### Gaps
- Exact repair, drawdown and generalization numbers for PRDNN, APRNN and REASSURE on their benchmarks (MNIST, ImageNet/SqueezeNet, ACAS Xu) were not extracted this session.
- The NNrepair code release and the CAV venue were not confirmed.

## Q2. Search-based / fault-localization repair (Arachne, CARE, MODE, Apricot, FSGMix): how do they localize faults from failing inputs?

### Takeaway
These SE-community methods share a two-step pattern: localize, then search. They first rank neurons or weights by sensitivity (Arachne), causal effect (CARE) or state differences between correct and failing inputs (MODE). They then tune only those parameters by PSO/DE (Arachne, CARE) or by retraining with chosen data (MODE, Apricot, FSGMix). None gives formal guarantees. Arachne is the paper that most explicitly measures generalization to unseen failing inputs.

### Cited Findings
- **Arachne** (Sohn, Kang and Yoo, TOSEM vol. 32, no. 4, art. 85, May 2023, DOI 10.1145/3563210; arXiv 1912.12463; ICSE 2023 journal-first) (V):
  - It repairs a pre-trained model by directly adjusting weights rather than retraining. Sensitivity-based fault localization limits the set of weights to change. — [arXiv 1912.12463](https://arxiv.org/pdf/1912.12463); [ICSE 2023 J1st](https://conf.researchr.org/details/icse-2023/icse-2023-journal-first-papers/4/Arachne-Search-Based-Repair-of-Deep-Neural-Networks)
  - The optimizer is PSO in the arXiv version and Differential Evolution in the journal version. — [arXiv 1912.12463](https://arxiv.org/pdf/1912.12463)
  - Its patches generalise to 61.3% of unseen misbehaviour, against 10.2% for the prior state-of-the-art repair technique. It is also used to debias a gender classifier. — [KAIST record](https://dspace.kaist.ac.kr/handle/10203/307190)
  - Localization in more detail (U, from prior knowledge of the paper): weights in the last layer(s) are scored by the gradient of the loss on failing inputs and by forward impact. Pareto-front selection then picks the candidates, and the fitness balances fixing the negative inputs against keeping the positive ones.
  - Code: github.com/coinse/arachne (U).
- **CARE, "Causality-based Neural Network Repair"** (Bing Sun, Jun Sun, Long H. Pham and Jie Shi, ICSE 2022, arXiv 2204.09274) (V):
  - Causality-based fault localization finds the "guilty" neurons by their causal effect on the misbehaviour; PSO then tunes those neurons' weights. It applies to all layers. — [arXiv 2204.09274](https://arxiv.org/pdf/2204.09274); [ICSE 2022 page](https://conf.researchr.org/details/icse-2022/icse-2022-papers/44/Causality-Based-Neural-Network-Repair)
  - Results: fairness improves by 61.91% on average. Backdoor attack success rate falls from over 98% to under 1%. The safety-property violation rate falls below 1%, and accuracy is preserved. It is built on the SOCRATES framework, and its code is public. — [arXiv 2204.09274](https://arxiv.org/pdf/2204.09274)
- **MODE** (Ma et al., FSE 2018): it uses state differential analysis to find the internal neurons responsible for misclassifications, then retrains with newly selected inputs. On simple applications such as digits it is reported to raise test accuracy from 75% to 93% on average. A later tool paper notes that no MODE implementation is available. — [MODE PDF](https://people.cs.umass.edu/~shiqingma/papers/FSE18.pdf); [Rutgers record](https://scholarship.libraries.rutgers.edu/esploro/outputs/conferencePaper/MODE-automated-neural-network-model-debugging/991031794683204646) (V via search summary)
- **Apricot** (Zhang et al., ASE 2019): it trains reduced models on subsets of the data, then moves the original weights toward the reduced models that classify the failing inputs correctly. Accuracy rises by 0.35–2.81%, 1.45% on average. A later tool paper reports that no public implementation exists. — [ASE 2019 page](https://2019.ase-conferences.org/details/ase-2019-papers/79/Apricot-A-Weight-Adaptation-Approach-to-Fixing-Deep-Learning-Models) (V via search summary)
- **FSGMix, "Few-Shot Guided Mix for DNN Repairing"** (Ren, Yu, Ma et al., ICSME 2020): it treats repair as data augmentation guided by the few failure examples, and retrains the whole model. Its follow-up, DeepRepair (style-guided augmentation), uses FSGMix-style methods as baselines. — [ICSME 2020 PDF](https://conferences.computer.org/icsme/pdfs/ICSME2020-1oOutvkGTwF4GyVvNtr3Mm/561900a717/561900a717.pdf); [DeepRepair arXiv 2011.09884](https://arxiv.org/pdf/2011.09884) (V)
- Related works found in passing (V on existence only):
  - **ArchRepair** (block-level, architecture-oriented repair; arXiv 2111.13330). Notable because it treats a block as the unit of repair. — [arXiv 2111.13330](https://arxiv.org/pdf/2111.13330)
  - "A Comprehensive Study of Deep Learning Model Fixing Approaches" (arXiv 2512.23745) is a recent empirical comparison. — [arXiv 2512.23745](https://arxiv.org/pdf/2512.23745)
  - "Element-Based Automated DNN Repair with Fine-Tuned Masked Language Model" (FSE 2025) uses a language model to propose repairs. — [FSE 2025](https://conf.researchr.org/details/fse-2025/fse-2025-research-papers/16/Element-Based-Automated-DNN-Repair-with-Fine-Tuned-Masked-Language-Model)

### Inferences
- The "localize from failing inputs, then edit only those parameters" structure maps directly onto the repo's pipeline. In decompiled code, localization becomes "which branch, threshold or neuron-expression is hit by failing inputs and not by passing ones". That is spectrum-based fault localization on code, which the SE community already applies to programs, and it is easier to read in code than in weights.
- Arachne's 61.3% vs 10.2% unseen-failure generalization gap suggests a key metric for the repo: hold out some failing inputs and report how many of them the LLM's code fix also corrects.

### Gaps
- The exact CARE GitHub URL was not confirmed.
- FSGMix's own numbers were not extracted.

## Q3. Transformer model editing that localizes and then edits an MLP block (ROME, MEMIT, MEND, knowledge neurons) and the critiques

### Takeaway
ROME and MEMIT treat a transformer MLP sublayer as a linear key-value memory. They write new associations by a closed-form rank-one update (ROME, one layer) or by spreading updates over several MLP layers (MEMIT). This is the closest existing analogue of "take an MLP block, fix it, put it back". Hase et al. (NeurIPS 2023) showed that causal-tracing localization does not predict which layer is best to edit, so "find the block responsible" and "the block where an edit works" can differ. The repo should test that empirically and not assume it.

### Cited Findings
- **ROME, "Locating and Editing Factual Associations in GPT"** (Meng, Bau, Andonian and Belinkov, NeurIPS 2022, arXiv 2202.05262):
  - It uses causal tracing to locate where facts are stored, then makes a rank-one modification of the feed-forward (MLP) weights of one layer. — [arXiv 2202.05262](https://arxiv.org/pdf/2202.05262); [NeurIPS 2022](https://neurips.cc/virtual/2022/poster/53864) (V)
  - It is evaluated on zsRE and on a new CounterFact dataset, where it keeps both specificity and generalization while other methods trade one away. Code is at [rome.baulab.info](https://rome.baulab.info) and github.com/kmeng01/rome (U on the GitHub path). (V for the claims)
- **MEMIT, "Mass-Editing Memory in a Transformer"** (Meng, Sen Sharma, Andonian, Belinkov and Bau, ICLR 2023 oral, arXiv 2210.07229): it scales to thousands of edits on GPT-J (6B) and GPT-NeoX (20B), editing MLP weights across a range of layers (layers 3–8 for GPT-J). — [memit.baulab.info](https://memit.baulab.info/); [ICLR 2023](https://iclr.cc/virtual/2023/oral/12726); code github.com/kmeng01/memit (V via gittrend listing). A third-party summary that claims MEMIT edits attention layers contradicts the authors' own description and should be ignored.
- **Hase, Bansal, Kim and Ghandeharioun, "Does Localization Inform Editing?"** (NeurIPS 2023, arXiv 2301.04213) (V):
  - The correlation between causal-tracing results and edit success is near zero for ROME, MEMIT and Adam finetuning. Which layer is edited predicts performance far better than localization does. The result generalizes across localization methods, editing methods, metrics, models and datasets.
  - ROME and MEMIT still work, even when tracing points elsewhere.
  - Code: [google/belief-localization](https://github.com/eltociear/belief-localization) (a mirror was found; the canonical repo is google/belief-localization).
  - Sources: [NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/hash/3927bbdcf0e8d1fa8aa23c26f358a281-Abstract-Conference.html); [arXiv 2301.04213](https://arxiv.org/pdf/2301.04213)
- **MEND, "Fast Model Editing at Scale"** (Mitchell, Lin, Bosselut, Finn and Manning, ICLR 2022, arXiv 2110.11309; code github.com/eric-mitchell/mend) (U): it trains small hypernetworks that turn the finetuning gradient on the edit example into a low-rank weight update, typically applied to the last few MLP layers. — [arXiv 2110.11309](https://arxiv.org/abs/2110.11309) (U)
- **Knowledge Neurons** (Dai, Dong, Hao, Sui, Chang and Wei, ACL 2022, arXiv 2104.08696; code github.com/Hunter-DDM/knowledge-neurons) (U): it attributes facts to individual FFN intermediate neurons using integrated gradients, then suppresses, amplifies or edits those neurons' value vectors. This is the neuron-level analogue. — [arXiv 2104.08696](https://arxiv.org/abs/2104.08696) (U). ROME compares against it as an interpretability-based editing baseline. — [ROME arXiv](https://arxiv.org/pdf/2202.05262) (V)
- **Editing image classifiers**: Santurkar, Tsipras, Elango, Bau, Torralba and Madry, "Editing a classifier by rewriting its prediction rules" (NeurIPS 2021, arXiv 2112.01008) (V):
  - It edits one layer's weights so that one concept's representation maps to another, for example "snowy road" to "road" in VGG16, or removing a CLIP typographic attack, while keeping unrelated-class accuracy.
  - It builds on Bau et al. 2020, "Rewriting a deep generative model".
  - Code: [MadryLab/EditingClassifiers](https://github.com/MadryLab/EditingClassifiers).
  - Sources: [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/c46489a2d5a9a9ecfc53b17610926ddd-Abstract.html)
- **Lifelong / adaptor editors (U on details)**: GRACE (Hartvigsen et al., NeurIPS 2023, arXiv 2211.11031) adds a discrete key-value codebook adaptor at one layer. It is in effect a lookup table inserted into the block, with a radius that decides when to apply the stored fix. SERAC (Mitchell et al., ICML 2022) uses an external memory plus a scope classifier. AlphaEdit (ICLR 2025, arXiv 2410.02355) projects edits into the null space of preserved knowledge. EasyEdit includes GRACE and WISE. — [EasyEdit GitHub](https://github.com/zjunlp/EasyEdit) (V for GRACE/WISE being listed)

### Inferences
- ROME's framing, "MLP = linear associative memory; solve for a weight delta so that key k* maps to value v* while other keys are preserved", is the weight-space version of "replace this block's algorithm". A code-level fix to a decompiled MLP block could be recompiled with the same least-squares or closed-form machinery: fit the delta so the block's outputs match the fixed program on the failing inputs, while constraining the outputs on a covariance or sample of normal inputs.
- GRACE and SERAC are close to "replace the block with code on CPU". They keep the original block and route some inputs to a stored correction, which is a hybrid model-plus-lookup-program. The repo's "code replacing the block" option is a more general form of this, with readable logic in place of a codebook.
- The Hase et al. result is a warning for the repo. Even if decompilation shows a failing case passing through block X, fixing a different block may work better. A useful experiment is to run the LLM code fix on several candidate blocks and compare.

### Gaps
- MEND, knowledge neurons, GRACE, SERAC and AlphaEdit details, venues and GitHub URLs were not re-verified this session (from prior knowledge).
- ROME/MEMIT CounterFact numbers were not extracted.

## Q4. Interpretability-guided fixing (SHIFT, fixing after finding the algorithm, LLM or human reading extracted logic)

### Takeaway
SHIFT (Marks et al., ICLR 2025) is the clearest published case of a human reading interpretable units of a network, judging some to be wrong (spurious), and editing the network by ablating them. It works at the level of features, not code. No published work was found in which an LLM agent reads a network block decompiled into source code and fixes its algorithm from failing inputs. That looks like an open niche, flagged as "not found" rather than "does not exist".

### Cited Findings
- **SHIFT (Sparse Human-Interpretable Feature Trimming)**, from Marks et al., "Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models" (ICLR 2025, arXiv 2403.19647) (V via secondary summaries):
  - Steps: discover a circuit of SAE features; humans label features as task-relevant or not; ablate the irrelevant ones. On Bias in Bios they ablated 55 features to remove a gender-spurious signal from a profession probe. SHIFT plus finetuning reportedly reached about 93% accuracy, comparable to training on an unbiased dataset.
  - Sources: [ICLR 2025 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/3ba4d47a83e498c2b1a0868cba20f6de-Paper-Conference.pdf); [IAIFI summary](https://research.iaifi.org/posts/sparse-feature-circuits-discovering-and-editing-interpretable-causal-graphs-in-l)
  - Critique: an Alignment Forum post argues that Bias in Bios is too easy. Ablating only 10 post-embedding (token-level) SAE features gives 85.6% out-of-sample accuracy, against 88.5% for 55 features. A LessWrong comment says that baselines with the same affordances match SHIFT. — [AF post](https://www.alignmentforum.org/posts/QdxwGz9AeDu5du4Rk/shift-relies-on-token-level-features-to-de-bias-bias-in-bios)
  - Follow-up: specialized SAEs applied to SHIFT report 90.2 in-distribution and 88.5 worst-group accuracy, against 75.5 for a neuron-skyline baseline. — [arXiv 2411.00743](https://arxiv.org/pdf/2411.00743)
  - Code: github.com/saprmarks/feature-circuits (U).
- **Santurkar et al. 2021**: a human picks the concept mapping to rewrite, for example snow to asphalt; this is interpretability-guided editing at the level of a single layer. — [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/c46489a2d5a9a9ecfc53b17610926ddd-Abstract.html) (V)
- **NNrepair** translates a Keras model into Java source and runs symbolic execution on it (per a secondary review). That is the closest found precedent for "network as program, analyzed with program tools, then weights repaired". The repair is still done by a constraint solver, not by reading or editing the code's logic. — [velog review](https://velog.io/@thdalwh3867/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-NNrepair-Constraint-based-Repair-of-Neural-Network-Classifiers) (V, secondary)
- A targeted search for "LLM agent repairs neural network by editing extracted program/rules" found only:
  - generic LLM program-repair agents, such as RepairAgent (ICSE 2025), which edit source code and not network logic;
  - two unconfirmed leads: "Neuron-Anchored Rule Extraction for LLMs via Contrastive Hierarchical Ablation" (arXiv 2605.03058) and "Gradient-based program repair: fixing bugs in continuous program spaces" (arXiv 2505.17703). Their content was not verified.
  - Sources: [RepairAgent](https://software-lab.org/publications/icse2025_RepairAgent.pdf); [arXiv 2605.03058](https://arxiv.org/pdf/2605.03058)

### Inferences
- The repo's proposal differs from all the above along one axis: the edit is made in a human- or LLM-readable program space, and correctness on failing inputs is checked by running the code. Prior repair works in weight space (PRDNN, Arachne, CARE, ROME), or in feature-ablation space with human labels (SHIFT). Code-space repair gives readable diffs and lets ordinary program-repair tools (spectrum-based fault localization, tests) be reused. It loses differentiability, and recompiling to weights has to be solved separately, where PRDNN/APRNN-style LP repair or ROME-style least squares are candidate back-ends.
- The SHIFT critiques suggest that the repo's benchmarks need baselines with equal affordances: plain finetuning, PRDNN, and Arachne on the same failing set and the same held-out data. Without them, a code fix's success cannot be told apart from task easiness.

### Gaps
- No paper found where an LLM edits decompiled network code to fix failing cases (search was limited; flag as not found).
- The two arXiv leads were not verified.

## Q5. Standard evaluation metrics and benchmarks; runnable code

### Takeaway
DNN-repair papers report efficacy (fix rate on the repair set), drawdown (accuracy drop on held-out normal data) and generalization (fix rate on unseen failing inputs of the same kind). Model-editing papers report reliability or efficacy, generalization (paraphrases), locality or specificity (neighbourhood prompts unchanged), and portability. The repo can report the same triple: repair rate on failing inputs, regression on passing inputs, and fix rate on held-out failing inputs.

### Cited Findings
- **APRNN** results tables report drawdown and generalization, comparing against PRDNN and REASSURE on MNIST. — [APRNN GitHub](https://github.com/95616ARG/APRNN) (V)
- **Arachne** reports generalization to unseen misbehaviour (61.3% against 10.2%). — [KAIST](https://dspace.kaist.ac.kr/handle/10203/307190) (V)
- **CARE** reports misbehaviour reduction (fairness, backdoor attack success rate, safety violations) plus accuracy preservation. — [arXiv 2204.09274](https://arxiv.org/pdf/2204.09274) (V)
- **EasyEdit** metric definitions:
  - Reliability: accuracy on the edit cases.
  - Generalization: accuracy on equivalent rephrasings.
  - Locality: out-of-scope inputs unchanged.
  - Portability: the edit carries over to related facts.
  - Also fluency and efficiency.
  - Benchmarks include KnowEdit. ROME, MEND, MEMIT, GRACE, WISE and others are implemented in one library.
  - Sources: [arXiv 2308.07269](https://arxiv.org/pdf/2308.07269); [GitHub zjunlp/EasyEdit](https://github.com/zjunlp/EasyEdit) (V)
- **ROME/MEMIT** benchmarks are zsRE and CounterFact (efficacy, paraphrase, specificity). — [arXiv 2202.05262](https://arxiv.org/pdf/2202.05262) (V)
- **Code availability summary** (V = confirmed in-session, U = from memory):

| Method | Year / venue | Edits | Guarantee | Code |
|---|---|---|---|---|
| Goldberger et al. | 2020 LPAR | weights of one layer | provably minimal change meeting spec (via Marabou/Gurobi) | jjgold012/MinimalDNNModificationMarabou (U) |
| PRDNN | 2021 PLDI | one layer's weights in a decoupled DNN (LP) | spec holds on points/polytopes; minimal single-layer; infeasible means no single-layer repair | 95616ARG/PRDNN (U path) |
| NNrepair | 2021 (CAV, U) | weights into flagged neurons | constraint-satisfied on repair set | not found |
| REASSURE | 2022 ICLR | adds patch networks per linear region | sound and complete, local, minimal, poly-time | BU-DEPEND-Lab/REASSURE (V) |
| APRNN | 2023 PLDI | parameters, architecture kept | provable point/polytope repair | 95616ARG/APRNN (V) |
| Arachne | 2023 TOSEM | localized weights (PSO/DE) | none (search) | coinse/arachne (U) |
| CARE | 2022 ICSE | weights of causally-guilty neurons (PSO) | none | public per paper (URL U) |
| MODE | 2018 FSE | retrain via selected inputs | none | none available (V) |
| Apricot | 2019 ASE | whole-model weight adaptation | none | none available (V) |
| FSGMix | 2020 ICSME | retrain with failure-guided augmentation | none | not checked |
| Santurkar et al. | 2021 NeurIPS | one conv layer (rank-one rewrite) | none | MadryLab/EditingClassifiers (V) |
| ROME | 2022 NeurIPS | one MLP layer, rank-one | none (empirical) | kmeng01/rome; rome.baulab.info (V site) |
| MEMIT | 2023 ICLR | several MLP layers | none | kmeng01/memit (V) |
| MEND | 2022 ICLR | low-rank deltas via hypernetwork | none | eric-mitchell/mend (U) |
| Knowledge neurons | 2022 ACL | individual FFN neurons | none | Hunter-DDM/knowledge-neurons (U) |
| GRACE | 2023 NeurIPS | codebook adaptor at a layer | none | in EasyEdit (V listed) |
| SHIFT | 2025 ICLR | ablate SAE features (human-labelled) | none | saprmarks/feature-circuits (U) |
| Hase et al. (critique) | 2023 NeurIPS | n/a | n/a | google/belief-localization (V mirror) |

### Inferences
- For the repo's iris/MLP experiments, PRDNN or APRNN (small ReLU MLPs, LP-based, Gurobi needed for APRNN) and Arachne are the natural weight-space baselines against an LLM code fix. Report three numbers for each: fix rate on failing train inputs, accuracy change on passing inputs (drawdown), and fix rate on held-out failing inputs (generalization).
- For transformer MLP blocks, EasyEdit gives ROME, MEMIT, MEND and GRACE baselines under one harness with standard locality and generalization metrics.

### Gaps
- No standard benchmark exists for "repair via decompiled code" (none found).
- The PRDNN and REASSURE benchmark numbers were not extracted.
