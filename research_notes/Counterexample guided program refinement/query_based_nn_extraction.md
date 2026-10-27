# Query-based and counterexample-guided extraction of rules, automata and programs from trained neural networks

Scope: prior work where something is extracted from a trained net using membership queries (MQ), equivalence queries (EQ) or counterexamples. The repo's loop is: an LLM proposes a Python program, a checker compares it with a small ReLU MLP on about 200k inputs (uniform, grid, corners, activation-boundary points) and returns counterexamples, and the LLM repairs the program. That loop is an approximate EQ oracle plus a repairing learner.

Verification status: [V] means confirmed this session against a primary or near-primary page (arXiv, proceedings, publisher, PyPI). [U] means from background knowledge and not re-checked this session; the report writer should treat [U] items as unverified.

---

## Q1. Angluin L*-style extraction from RNNs, and how the equivalence query is implemented against a net

### Takeaway
Weiss, Goldberg and Yahav (ICML 2018) is the canonical "learner proposes a hypothesis, net-backed oracle returns a counterexample" method. The RNN answers MQs directly. The EQ is approximated by comparing the L* DFA with a finite abstraction of the RNN's hidden-state space, and the abstraction is refined (SVM split) only when a disagreement is not a real counterexample. Later work replaced that EQ with random sampling (PAC-style, Mayr and Yovine), conformance testing (Muškardin et al., AALpy), or extended L* to weighted and probabilistic automata (Weiss et al. NeurIPS 2019; Okudono et al. AAAI 2020).

### Cited Findings
**Weiss, Goldberg, Yahav, "Extracting Automata from Recurrent Neural Networks Using Queries and Counterexamples", ICML 2018 (PMLR 80:5247-5256); extended version in Machine Learning 113:2877-2919 (2024), DOI 10.1007/s10994-022-06163-2** [V]
- L* is the learner and the trained RNN is the oracle. The RNN abstraction is refined only as far as needed to answer EQs. Disagreements between the abstraction and the L* hypothesis, checked against the RNN's true label, give either a counterexample for the DFA or a refinement of the abstraction, so the method never returns a wrong counterexample. — [arXiv 1711.09576](https://arxiv.org/abs/1711.09576); [MLJ 2024 extended version (mlanthology)](https://mlanthology.org/mlj/2024/weiss2024mlj-extracting)
- How the EQ works (from the paper text):
  - The partition of RNN state space starts maximally coarse, with all states in one abstract state. Two short labeled words (one accepted, one rejected) are also tried first as cheap counterexamples.
  - The abstraction is explored by BFS. Each abstract state is paired with the L* state reached by the same word.
  - Classification conflict: the RNN label differs from the DFA label, and the word is returned as a counterexample.
  - Clustering conflict: one abstract state is reached by words that lead to different L* states. A distinguishing suffix is appended and the RNN is queried. If the DFA is wrong on one of these words, that word is the counterexample; otherwise the partition is split.
  - Counterexamples are preferred over refinement, and the shortest one is returned.
  - The split is an RBF-kernel SVM (sklearn SVC, C=10^4) separating the conflicting RNN state from the others in its cell, and the splits are kept in a decision tree. An "aggressive" difference-based split (initial depth 10) is used for the first refinements.
  - Extraction stops on a time or size limit and returns the last L* hypothesis. The authors note that agreement between the abstraction and L* does not prove the RNN and DFA are equivalent.
  — [ar5iv 1711.09576](https://ar5iv.labs.arxiv.org/html/1711.09576)
- Experiments:
  - The 7 Tomita grammars on 2-layer GRUs with hidden size 100, each extracted correctly in under 2 s.
  - Random 10-state DFAs on GRU and LSTM nets with hidden sizes 50, 100 and 500, with a 30 s limit.
  - Balanced parentheses, emails and tokenized JSON.
  — [ar5iv 1711.09576](https://ar5iv.labs.arxiv.org/html/1711.09576)
- Counterexample source matters. On balanced parentheses, the abstraction-guided EQ reached 99.98% train-set agreement for both nets, against 87.12% (GRU) and 94.19% (LSTM) when random sampling was the EQ. It also captured nesting depth 8 versus 2-3. Omlin and Giles style quantization (q=2) did not finish in 1000 s and produced DFAs with more than 60,000 states, and k-means clustering only reached depth 3. — [ar5iv 1711.09576](https://ar5iv.labs.arxiv.org/html/1711.09576)
- A later source says the approach suffers from state-space explosion and does not scale to today's network sizes (search-summary claim, not checked in the primary text). — [alphaxiv 1711.09576 summary](https://alphaxiv.org/abs/1711.09576)
- Code: github.com/tech-srl/lstar_extraction (linked from the paper). — [ar5iv 1711.09576](https://ar5iv.labs.arxiv.org/html/1711.09576)

**Weiss, Goldberg, Yahav, "Learning Deterministic Weighted Automata with Queries and Counterexamples", NeurIPS 2019** [V]
- Adapts L* to reconstruct a PDFA from a black-box distribution over sequences, such as a language-model RNN. — [arXiv 1910.13895](https://arxiv.org/abs/1910.13895v1); [NeurIPS PDF](https://papers.NeurIPS.cc/paper_files/paper/2019/file/d3f93e7766e8e1b7ef66dfdd9a8be93b-Paper.pdf)
- Reports better WER and NDCG than spectral extraction of WFAs from the same networks, and notes that spectral methods need not return stochastic hypotheses. — [arXiv 1910.13895](https://arxiv.org/pdf/1910.13895v2)
- Code: github.com/tech-srl/weighted_lstar. — [arXiv 1910.13895 footnote](https://arxiv.org/pdf/1910.13895v2)

**Okudono, Waga, Sekiyama, Hasuo, "Weighted Automata Extraction from Recurrent Neural Networks via Regression on State Spaces", AAAI 2020** [V for existence and venue]
- Extracts a WFA from an RNN. In the WL* family, the EQ and state mapping are done by regression on the RNN state space. — [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/5977)
- Zhang et al., "Decision-Guided Weighted Automata Extraction from RNNs" (AAAI 2021) appeared in the search results ([AAAI OJS 17391](https://ojs.aaai.org/index.php/AAAI/article/view/17391/17198)); its content was not checked [U].

**Spectral extraction (Ayache, Eyraud, Goudian, "Explaining black boxes on sequential data using weighted automata", ICGI 2018)** [U]
- Builds a Hankel matrix from MQs (net outputs on prefix-suffix pairs) and runs spectral learning. There is no EQ or counterexample loop, so it is query-based but not counterexample-guided. Weiss et al. 2019 is the source that confirms spectral WFA extraction is the baseline. — [arXiv 1910.13895](https://arxiv.org/pdf/1910.13895v2)

**Mayr and Yovine, "Regular Inference on Artificial Neural Networks", CD-MAKE 2018 (LNCS 11015)** [V]
- Randomized L* variant where each EQ is replaced by random sampling. The number of samples per EQ is set from accuracy and confidence parameters and grows with the query index, which gives a PAC guarantee instead of exact identification. — [IFIP DL / HAL](https://dl.ifip.org/IFIP-LNCS-11015/hal-02060043); description in [arXiv 2504.02170](https://arxiv.org/pdf/2504.02170)
- Follow-ups by the same group include property checking for RNNs (2021) and congruence-based learning of PDFA from neural language models. — [arXiv 2412.09760](https://arxiv.org/pdf/2412.09760)

**Muškardin, Aichernig, Pill, Tappler, "Learning Finite State Models from Recurrent Neural Networks", iFM 2022 (pp. 229-248)** [V]
- Black-box L* where the EQ is model-guided conformance testing. The authors report that this learns FSMs that match the RNN more closely than other EQ strategies, because it catches imprecise generalizations they miss. — [TU Graz repository](https://graz.elsevierpure.com/de/publications/learning-finite-state-models-fromrecurrent-neural-networks); [PDF](https://repository.tugraz.at/publications/marc21/w60gb-s0s04/files/89363.pdf?download=1)
- Follow-up: Muškardin, Tappler, Aichernig, "Testing-based Black-box Extraction of Simple Models from RNNs and Transformers", ICGI 2023. — [dblp](https://dblp.uni-trier.de/pid/274/2463.html)
- Tooling: AALpy, a pure-Python library on PyPI (pip install aalpy). It provides L*, KV and RPNI for DFA, Mealy and Moore models, and the PyPI listing names "13 Equivalence Oracles" implementing conformance-testing strategies such as RandomWalkEqOracle. — [PyPI aalpy](https://www.pypi.org/project/aalpy)

### Inferences
- The structure is the same as the repo's: the hypothesis (DFA or program) is checked against the net, a disagreement input is returned, and the hypothesis is updated. The main difference is that L* updates its hypothesis by a deterministic table-filling procedure with guarantees, while the repo uses an LLM repair with no guarantee.
- Weiss et al. find counterexamples by looking where the net's internal structure (state clusters) disagrees with the hypothesis, not by sampling inputs. That is the RNN counterpart of the repo's activation-boundary scan. Their random-sampling baseline (87-94% vs 99.98%) is direct evidence that structure-aware counterexample search beats random EQs.
- For feed-forward ReLU MLPs the analogue of "refine the state abstraction" is splitting on activation patterns or linear regions. The repo's boundary-point sampling is a cheap version of that.

### Gaps
- No primary-source query counts were found for Weiss et al. (they report time, not MQ/EQ counts).
- AALpy's RNN-extraction examples were not confirmed. The search showed only the general library and its EQ oracles.
- The Ayache et al. ICGI 2018 details and Zhang et al. AAAI 2021 content were not verified.

---

## Q2. Query-based rule extraction and model extraction for feed-forward nets (TREPAN, active learning, model stealing, cryptanalytic extraction)

### Takeaway
There are two lineages.

1. Interpretable surrogate extraction (TREPAN, 1995/96) uses the net as a membership oracle to generate extra samples near decision nodes. It reports fidelity, but has no EQ or counterexample loop.
2. Model stealing and cryptanalytic extraction (Tramèr 2016, Milli 2019, Jagielski 2020, Rolnick and Kording 2020, Carlini 2020, Canales-Martínez et al. 2023/24, Carlini et al. 2025 hard-label) recovers exact ReLU weights by finding critical points (activation boundaries) with line or binary search.

Lineage 2 is closest to the repo's boundary-scan heuristic. However, it recovers weights, not a readable program, and it assumes the architecture is known.

### Cited Findings
**TREPAN: Craven and Shavlik, "Extracting Tree-Structured Representations of Trained Networks", NIPS 8 (conference 1995, proceedings 1996)** [V]
- Induces a decision tree that approximates the network, using the network as an oracle for queries. Queries can be constraints on feature values instead of full instances. The tree is grown best-first. The authors report high fidelity while staying comprehensible. — [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/1995/hash/45f31d16b1058d586fc3be7207b58053-Abstract.html); [Craven 1996 PhD thesis](https://ftp.cs.wisc.edu/machine-learning/shavlik-group/craven.thesis.pdf)
- X-TREPAN, a multi-class and regression extension, exists on arXiv. — [arXiv 1508.07551](https://arxiv.org/pdf/1508.07551)
- Background [U]: TREPAN draws extra samples from a model of the training distribution and labels them with the net, so that every node has enough examples (min_samples). That is MQs with no counterexamples. No maintained official code was found.

**Tramèr, Zhang, Juels, Reiter, Ristenpart, "Stealing Machine Learning Models via Prediction APIs", USENIX Security 2016 (pp. 601-618)** [V]
- Black-box extraction of logistic regression, neural networks and decision trees with near-perfect fidelity, demonstrated against BigML and Amazon ML. Two attack families:
  - equation-solving from returned confidences;
  - a path-finding algorithm for decision trees that uses confidences as quasi-identifiers of leaves, which the authors call the first practical "exact" decision-tree learning.
  Omitting confidences still leaves extraction possible. — [arXiv 1609.02943](https://arxiv.org/abs/1609.02943v1); [USENIX page](https://usenix.org/conference/usenixsecurity16/technical-sessions/presentation/tramer)
- Code: [U] github.com/ftramer/Steal-ML (background knowledge; the search did not confirm the repo name).

**Milli, Schmidt, Dragan, Hardt, "Model Reconstruction from Model Explanations", FAT\* 2019** [U for venue]
- Recovers 2-layer ReLU nets using gradient (explanation) queries. Jagielski et al. say they extend Milli et al.'s approach to 2-layer ReLU networks. — [Jagielski et al. arXiv 1909.01838](https://arxiv.org/pdf/1909.01838)

**Jagielski, Carlini, Berthelot, Kurakin, Papernot, "High Accuracy and High Fidelity Extraction of Neural Networks", USENIX Security 2020 (pp. 1345-1362)** [V]
- Distinguishes accuracy (task performance) from fidelity (agreement with the victim on any input). The limit case is functional equivalence, meaning agreement on all inputs.
- Argues that learning-based extraction cannot reach true high fidelity, because initialization, SGD batch randomness and GPU nondeterminism get in the way.
- Gives the first practical functionally-equivalent attack, which recovers 2-layer ReLU weights directly (without training) from double-precision outputs.
— [USENIX page](https://www.usenix.org/conference/usenixsecurity20/presentation/jagielski); [arXiv 1909.01838](https://arxiv.org/pdf/1909.01838)

**Rolnick and Kording, "Reverse-engineering deep ReLU networks", ICML 2020 (PMLR 119:8178-8187)** [V]
- Recovers architecture, weights and biases (up to isomorphism) from outputs alone. The method splits the set of linear-region boundaries into pieces belonging to individual neurons, locating boundary points by querying along line segments. — [PMLR](https://proceedings.mlr.press/v119/rolnick20a.html); [arXiv 1910.00744](https://arxiv.org/abs/1910.00744v2)
- A later paper says Rolnick and Kording is more efficient in time complexity and Carlini et al. in space complexity. — (search snippet citing [arXiv 2312.04675](https://arxiv.org/pdf/2312.04675v1); not checked in full)

**Carlini, Jagielski, Mironov, "Cryptanalytic Extraction of Neural Network Models", CRYPTO 2020 (arXiv 2003.04884)** [V; the CRYPTO venue comes from the search summary and the author's slides]
- Treats extraction as a differential cryptanalysis problem and recovers parameters "up to floating point precision", relying on ReLU piecewise linearity. Queries at critical points leak the weights. — [arXiv 2003.04884](https://arxiv.org/abs/2003.04884); [Carlini CRYPTO 2020 slides](https://nicholas.carlini.com/slides/2020_crypto_extraction.pdf)
- Query counts and precision:
  - 100,000-parameter MNIST net: about 2^21.5 queries, under 1 hour, worst-case error about 2^-25.
  - 4,000-parameter model: about 2^18.5 queries, error about 2^-40.4.
  - The authors claim 2^20 times more precision with 100x fewer queries than prior work.
  — [arXiv 2003.04884](https://arxiv.org/abs/2003.04884)
- Critical-point search:
  - The basic method picks a random line x0 + t·v and binary-searches for points where the gradient changes, searching an interval only if the derivative differs at its ends. This needs about p queries for p bits of precision.
  - The improved method takes two points. If they differ in exactly one neuron's sign, it computes the crossing analytically by intersecting the two linear pieces, using value and gradient at each end. If more than two pieces show up, it searches again.
  — [ar5iv 2003.04884](https://ar5iv.labs.arxiv.org/html/2003.04884)
- Weight recovery:
  - At a witness (critical point), finite-difference second derivatives give first-layer weight ratios, with bias b = -A·x*.
  - Signs are recovered by testing preimages of h ± e_i.
  - The method peels layer by layer, solving for deeper layers by least squares.
  - Precision refinement improves relative error from about 2^-15 to 2^-35.
  - It assumes float64, a known architecture, full (not top-label) outputs and ReLU activations. Expansive nets need a brute-force sign search that is exponential in width.
  - The authors show exact (ε,0)-equivalence checking is NP-hard, by reduction from Subset Sum.
  — [ar5iv 2003.04884](https://ar5iv.labs.arxiv.org/html/2003.04884)
- Code: github.com/google-research/cryptanalytic-model-extraction. — [ar5iv 2003.04884](https://ar5iv.labs.arxiv.org/html/2003.04884); [Papers with Code](https://cs.paperswithcode.com/paper/cryptanalytic-extraction-of-neural-network)

**Canales-Martínez, Chávez-Saab, Hambitzer, Rodríguez-Henríquez, Satpute, Shamir, "Polynomial Time Cryptanalytic Extraction of Neural Network Models" (arXiv 2310.08708, 2023)** [V for the content; Eurocrypt 2024 venue and author list are U]
- Fixes the exponential sign-recovery step of Carlini et al. All 8×256 neuron signs of a CIFAR-10 net were recovered in 32 minutes on a 256-core machine. — [arXiv 2310.08708](https://arxiv.org/pdf/2310.08708)

**Carlini, Chávez-Saab, Hambitzer, Rodríguez-Henríquez, Shamir, hard-label polynomial-time extraction, EUROCRYPT 2025 (ePrint 2024/1580)** [V]
- First hard-label (top-label only) extraction of ReLU nets with polynomially many queries and polynomial time. Recovered nearly 1M parameters of a CIFAR-10 net with 832 neurons in 4 hidden layers. Chen et al. (Asiacrypt 2024) did hard-label extraction earlier, but in exponential time and only on tiny nets. — [ePrint 2024/1580](https://eprint.iacr.org/2024/1580)
- Disputed by follow-ups:
  - Ito, Miura, Todo (ePrint 2025/1868, listed as CRYPTO 2026) say one almost-always-active neuron can block the attack, and that the cost grows exponentially with depth. — [ePrint 2025/1868](https://eprint.iacr.org/2025/1868); [arXiv 2510.06692](https://arxiv.org/pdf/2510.06692)
  - Tang et al. (ePrint 2026/2061) report that sign recovery can need about 2^52 guesses on a CIFAR-10 model, and propose a polynomial fix. — [ePrint 2026/2061](https://eprint.iacr.org/2026/2061)
- A 2026 MIT course report found that the original Carlini attack failed in 0 of 9 trials against output-rounded oracles (student work, low confidence). — [MIT 6.5610 report](https://65610.csail.mit.edu/2026/reports/cryptanalytic_nn_extract.pdf)

**Daniely and Granot, "An Exact Poly-Time Membership-Queries Algorithm for Extracting a Three-Layer ReLU Network", ICLR 2023 (arXiv 2105.09673)** [V]
- Learning theory: exact recovery of depth-2 and depth-3 ReLU nets from membership queries in polynomial time, under general-position assumptions. The authors say no poly-time result existed before for depth 3 or more. — [arXiv 2105.09673](https://arxiv.org/abs/2105.09673v5); [ICLR 2023 poster](https://iclr.cc/virtual/2023/poster/11830)

**Active learning and model extraction** [U]
- Chandrasekaran, Chaudhuri, Giacomelli, Jha, Yan, "Exploring Connections Between Active Learning and Model Extraction", USENIX Security 2020, frames extraction as query-synthesis active learning.
- Papernot et al. 2017 (Jacobian-based synthetic data augmentation) and Orekondy et al. 2019 (Knockoff Nets) are learning-based extraction.
- None of these was re-verified this session.

### Inferences
- The critical-point line search in Carlini et al. and Rolnick and Kording is essentially the repo's boundary-scan heuristic. For a 1-4 input MLP, it could locate every activation hyperplane exactly with few queries, and the result could be handed to the LLM as structural hints ("neuron k switches on the plane w·x+b=0") instead of only as counterexample points.
- For the repo's tiny nets (1-4 inputs), white-box access makes weight recovery unnecessary. The useful idea to borrow is the line-search and piecewise-linear intersection trick for generating exact boundary counterexamples cheaply.
- TREPAN shows the same move as the repo's grid and boundary sampling: querying the net to densify samples where the surrogate is undecided. It is old (1996), so it is prior art for "use the net as a membership oracle to grow an interpretable model".
- Jagielski et al.'s argument that learning-based extraction cannot reach functional equivalence supports the repo's choice of a checker plus exact program repair over distillation by training.

### Gaps
- Tramèr's code repository name was not confirmed.
- The exact per-critical-point query count in Carlini et al. was not stated in the fetched text.
- The Chandrasekaran et al. 2020 details were not verified.
- No pip-installable package was found for the cryptanalytic attacks; they appear to be research code only.

---

## Q3. Verification-in-the-loop extraction (NN verifiers proving extracted rule ≡ net or returning counterexamples)

### Takeaway
No paper was found that does exactly "extract an interpretable rule or program from a ReLU net, then use a complete NN verifier (Marabou, alpha-beta-CROWN, MILP) to prove equivalence or return a counterexample, then repair." The closest pieces are:
- CEGIS synthesis of programs or shields from neural policies with a verifier (Zhu et al. PLDI 2019);
- verifier-counterexample loops on ReLU nets for other properties (COMET, NeurIPS 2020);
- imitation-based policy-to-tree extraction with post-hoc verification (VIPER);
- standalone NN-vs-NN equivalence checkers (MILPEQUIV CP 2020, geometric path enumeration, ReluDiff/NeuroDiff, SMT equivalence).

Those equivalence checkers could serve as the repo's exact oracle, because a straight-line Python program over piecewise-linear operations can itself be encoded as MILP or SMT.

### Cited Findings
**Zhu, Xiong, Magill, Jagannathan, "An Inductive Synthesis Framework for Verifiable Reinforcement Learning", PLDI 2019 (Distinguished Paper)** [V]
- Verification of a neural RL policy is cast as counterexample-guided, syntax-guided inductive synthesis. A deterministic program that imitates the net and an inductive invariant are searched for together, in a CEGIS loop that removes safety counterexamples. Each counterexample initial state triggers synthesis of another program piece, and the pieces together cover the safe initial states.
- The algorithm is sound but may not terminate.
— [arXiv 1907.07273](https://arxiv.org/abs/1907.07273); [PLDI PDF](https://www.cs.purdue.edu/homes/suresh/papers/pldi19.pdf); [Rutgers award news](https://www.cs.rutgers.edu/news-events/news/news-item/he-zhu-receives-pldi-2019-distinguished-paper-award)
- Code [U]: background knowledge says it was released as "Revel"/"VRL" on GitHub; not confirmed.

**Bastani, Pu, Solar-Lezama, "Verifiable Reinforcement Learning via Policy Extraction" (VIPER), arXiv 1805.08328; NeurIPS 2018 [venue U]** [V for content]
- Distills a DNN policy into a decision tree with DAgger-style imitation, in which the DNN is queried on states the tree visits. The trees are about 10x smaller than DAgger's, which makes them verifiable. Verification found a counterexample to the tree policy's correctness, fixed by changing the environment (longer paddle). — [arXiv 1805.08328](https://arxiv.org/pdf/1805.08328)

**Sivaraman, Farnadi, Millstein, Van den Broeck, "Counterexample-Guided Learning of Monotonic Neural Networks" (COMET), NeurIPS 2020** [V]
- Uses SMT verification counterexamples on arbitrary ReLU nets to build a monotonic envelope at prediction time, and uses monotonicity as an inductive bias in training.
- Reviewers flagged the SMT cost and small datasets.
- Code: github.com/AishwaryaSivaraman/COMET.
— [NeurIPS abstract](https://proceedings.neurips.cc/paper/2020/hash/8ab70731b1553f17c11a3bbc87e0b605-Abstract.html); [arXiv 2006.08852](https://arxiv.org/pdf/2006.08852)

**Equivalence checking of neural networks**
- Kleine Büning, Kern, Sinz, "Verifying Equivalence Properties of Neural Networks with ReLU Activation Functions", CP 2020: an MILP encoding called MILPEQUIV. [V for existence and venue] — [Kern publications page](https://formal.kastel.kit.edu/~kern); [Teuber et al. describing MILPEQUIV](https://arxiv.org/abs/2112.06582v1)
- Teuber, Kleine Büning, Kern, Sinz, "Geometric Path Enumeration for Equivalence Verification of Neural Networks" (arXiv 2112.06582):
  - targets ε-equivalence and top-1 equivalence of structurally different nets;
  - outperforms MILPEQUIV in 4 of 5 ε-equivalence benchmarks and in counterexample finding;
  - notes that counterexamples from failed proofs can be used for further training.
  [V; venue ICTAI 2021 is U] — [arXiv 2112.06582](https://arxiv.org/pdf/2112.06582)
- Eleftheriadis et al., "On Neural Network Equivalence Checking using SMT Solvers" (arXiv 2203.11629). [V for existence] — [arXiv 2203.11629](https://arxiv.org/pdf/2203.11629)
- Paulsen, Wang, Wang, ReluDiff (ICSE 2020) and NeuroDiff (ASE 2020): differential verification of two nets by interval propagation. [U; Teuber et al. only confirm "the interval propagation approach of Paulsen et al."] — [arXiv 2112.06582](https://arxiv.org/pdf/2112.06582)
- A 2024 benchmark tests NN verifiers' soundness with hidden counterexamples. This is relevant if a verifier is used as the oracle, since verifiers can be unsound. — [arXiv 2412.03154](https://arxiv.org/html/2412.03154v1)

**LLM + CEGIS (adjacent, not net extraction)**
- LLMs combined with SAT/verifier oracles in CEGIS for plans and programs. — [arXiv 2309.16436](https://arxiv.org/pdf/2309.16436)
- LLM-synthesized abstract transformers for NN verification, guided by counterexamples and cost feedback. — [arXiv 2511.13663](https://www.alphaxiv.org/abs/2511.13663)
- ProbeGen: LLM-generated probes disprove equivalence for 18% of samples deemed equivalent by unit tests. — [arXiv 2502.18473](https://arxiv.org/pdf/2502.18473v1)
- No paper was found where an LLM decompiles a trained MLP into a program under a net-backed counterexample oracle.

### Inferences
- The repo could upgrade its sampling checker to a complete EQ on its tiny nets. Encode the net (MILP big-M or SMT over linear real arithmetic) together with the candidate program's piecewise-linear branches, and ask for x with |net(x) - prog(x)| > ε, or argmax disagreement. With 1-4 inputs and small widths this should be cheap, and MILPEQUIV-style encodings are prior art for it. A program with non-linear operations (e.g. multiplication of inputs) would need NRA/SMT or a fallback to sampling.
- Zhu et al. (PLDI 2019) is the closest published analogue of "propose program, get counterexample, repair". The synthesizer is enumerative or sketch-based rather than an LLM, and the oracle is a safety verifier rather than a functional-equivalence checker.
- Ito et al. and the verifier-soundness benchmark both show that "the oracle found no counterexample" is weaker evidence than it looks. That matches the repo's need to mix uniform, grid, corner and boundary sampling.

### Gaps
- No work was found that pairs Marabou or alpha-beta-CROWN with rule or program extraction for functional equivalence. The search may have missed it; queries such as "verified decision tree extraction neural network Marabou" are worth a follow-up.
- The venues for VIPER, Teuber et al. and ReluDiff are unverified.
- MILPEQUIV code availability was not confirmed.

---

## Q4. What these works measure (queries, fidelity) and how counterexample choice (random vs boundary/structural) changes results

### Takeaway
The metrics are:
- automata extraction: agreement with the net on held-out or random words, hypothesis size, wall time;
- cryptanalytic extraction: query count and max-error (log2 precision);
- model stealing: fidelity vs accuracy;
- TREPAN: fidelity and tree size.

The clearest direct evidence that counterexample choice matters comes from:
- Weiss et al. 2018: abstraction-guided counterexamples beat random-sampling EQ, 99.98% vs 87-94%;
- Muškardin et al. 2022: model-guided conformance testing beats other EQ strategies;
- Carlini et al.: structured critical-point queries make exact recovery possible with about 2^18-2^21 queries where learning-based methods cannot reach functional equivalence.

### Cited Findings
- Weiss et al. measure agreement with the RNN on train sets and on random words of length up to 1000, DFA size, extraction time and counterexample-generation time. Abstraction-guided EQ gave 99.98% vs 87.12% / 94.19% for random-sampling EQ on balanced parentheses (depth 8 vs 2-3). — [ar5iv 1711.09576](https://ar5iv.labs.arxiv.org/html/1711.09576)
- Mayr and Yovine bound EQ sample counts by PAC parameters (ε, δ), with the count growing with the query index. — [arXiv 2504.02170 description](https://arxiv.org/pdf/2504.02170)
- Muškardin et al.: model-guided conformance testing finds generalization errors that other EQ techniques miss and yields more accurate FSMs. — [TU Graz](https://graz.elsevierpure.com/de/publications/learning-finite-state-models-fromrecurrent-neural-networks)
- Weiss et al. 2019 measure WER and NDCG against the target LM. — [arXiv 1910.13895](https://arxiv.org/pdf/1910.13895v2)
- Carlini et al. measure queries (2^18.5 to 2^21.5), wall time, and worst-case output error (2^-25 to 2^-40.4) as a bound on (ε,0)-functional equivalence. Exact equivalence checking is NP-hard. — [arXiv 2003.04884](https://arxiv.org/abs/2003.04884); [ar5iv](https://ar5iv.labs.arxiv.org/html/2003.04884)
- Jagielski et al. define and separate accuracy from fidelity and argue that learning-based methods cannot reach functional equivalence. — [arXiv 1909.01838](https://arxiv.org/pdf/1909.01838)
- TREPAN reports fidelity to the net and comprehensibility (tree size). — [NeurIPS 1995](https://proceedings.neurips.cc/paper_files/paper/1995/hash/45f31d16b1058d586fc3be7207b58053-Abstract.html)
- Teuber et al. compare equivalence checkers on proof time and counterexample-finding success. — [arXiv 2112.06582](https://arxiv.org/pdf/2112.06582)

### Inferences
- The repo should report, per net:
  - checker queries until the first counterexample;
  - the number of repair rounds;
  - final disagreement rate on uniform samples vs on boundary samples, reported separately;
  - max |error| for regression.
  These match the literature's metrics (queries, fidelity, max error) and would let the repo test, the way Weiss et al. did, whether boundary counterexamples converge faster than uniform ones.
- An ablation is suggested by the literature but not run in it for MLP-to-program extraction: give the repairer only random counterexamples, only boundary counterexamples, or both.

### Gaps
- No study was found that measures random vs activation-boundary counterexamples for feed-forward MLP surrogate extraction specifically. This is a likely gap the repo can fill.

---

## Q5. Runnable open-source code and closeness to "propose program, counterexample from net, repair"

### Takeaway
Runnable research code exists for L* RNN extraction (tech-srl), weighted L* (tech-srl), cryptanalytic extraction (google-research) and COMET. AALpy is the one clearly pip-installable library (pip install aalpy) with pluggable EQ oracles. Nothing found is an LLM-driven program-repair loop against a net. The closest by loop structure are Weiss et al. 2018, Zhu et al. 2019 and AALpy-based extraction. The closest by "find activation boundaries along lines" are Carlini et al. 2020 and Rolnick and Kording 2020.

### Cited Findings
| Method | Year / venue | Code | Pip? | Closeness to the repo loop |
|---|---|---|---|---|
| TREPAN (Craven and Shavlik) | NIPS 8, 1995/96 [V] | none official found | no | Low-medium: MQs to grow a tree; no EQ or counterexamples |
| Tramèr et al. model stealing | USENIX Sec 2016 [V] | ftramer/Steal-ML [U] | no [U] | Low: equation-solving and path-finding; no hypothesis repair |
| Weiss et al. L* + abstraction EQ | ICML 2018 / MLJ 2024 [V] | [tech-srl/lstar_extraction](https://ar5iv.labs.arxiv.org/html/1711.09576) [V] | no [U] | High: hypothesis, EQ against the net, counterexample, update (DFA, not program) |
| Mayr and Yovine PAC L* | CD-MAKE 2018 [V] | not checked | ? | High in loop shape; random-sampling EQ, like the repo's uniform sampler |
| VIPER (Bastani et al.) | arXiv 1805.08328 / NeurIPS 2018 [U venue] | not checked | ? | Medium: queries the DNN on visited states; verification afterwards |
| Zhu et al. CEGIS shields | PLDI 2019 [V] | not confirmed [U] | no | High: program synthesis plus verifier counterexamples on a neural policy |
| Weiss et al. weighted L* | NeurIPS 2019 [V] | [tech-srl/weighted_lstar](https://arxiv.org/pdf/1910.13895v2) [V] | no [U] | High (probabilistic automata) |
| Okudono et al. WFA via regression | AAAI 2020 [V] | not checked | ? | Medium-high |
| Rolnick and Kording | ICML 2020 [V] | not checked | ? | Medium: boundary finding along lines; recovers weights, not programs |
| Carlini et al. cryptanalytic | CRYPTO 2020 [V] | [google-research/cryptanalytic-model-extraction](https://ar5iv.labs.arxiv.org/html/2003.04884) [V] | no [U] | Medium: same boundary line search as the repo, but the target is weights |
| Jagielski et al. | USENIX Sec 2020 [V] | not checked | ? | Low-medium: fidelity framing; functionally-equivalent 2-layer extraction |
| COMET (Sivaraman et al.) | NeurIPS 2020 [V] | [AishwaryaSivaraman/COMET](https://arxiv.org/pdf/2006.08852) [V via search summary] | no [U] | Medium: SMT counterexamples on a ReLU net, but for monotonicity |
| MILPEQUIV / geometric path enumeration | CP 2020 [V] / arXiv 2112.06582 [V] | not checked | ? | Oracle candidate: a complete EQ for net vs net or net vs program |
| Muškardin et al. + AALpy | iFM 2022, ICGI 2023 [V] | [AALpy on PyPI](https://www.pypi.org/project/aalpy) [V] | yes (aalpy) | High: L* with pluggable conformance-testing EQ oracles |
| Daniely and Granot | ICLR 2023 [V] | theory | n/a | Low (theory of MQ-exact recovery) |
| Canales-Martínez et al. | arXiv 2310.08708 (2023), Eurocrypt 2024 [U venue] | not checked | ? | Medium (polynomial-time sign recovery) |
| Carlini et al. hard-label | EUROCRYPT 2025 [V] | not checked | ? | Medium (label-only boundary queries, like the repo's argmax nets) |

### Inferences
- What is novel in the repo relative to this literature:
  - the hypothesis class is open-ended readable Python, not DFAs, trees or weights;
  - the learner is an LLM repairing from counterexamples, not L* or an enumerative synthesizer;
  - the target is a feed-forward ReLU MLP with real or integer inputs.
- The prior art for each component is strong: L*-style EQ loops, boundary line search, and MILP/SMT equivalence checking.
- Two practical adoptions to consider:
  1. Use a MILP or SMT equivalence query as the final certifying oracle, following MILPEQUIV.
  2. Use Carlini-style analytic intersection of linear pieces to generate exact boundary counterexamples with far fewer than 200k queries.

### Gaps
- Pip-installability was confirmed only for AALpy. The other repositories appear to be research code, and their install status and maintenance were not checked.
- The code links for Zhu et al., Okudono et al., Rolnick and Kording, Jagielski et al. and MILPEQUIV were not verified this session.
