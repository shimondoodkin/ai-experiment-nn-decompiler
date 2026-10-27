# The extract -> fix -> reinsert cycle: prior work on extracting knowledge from trained nets, correcting it, and putting it back (or deploying it instead)

Scope note: about 25 tool calls (search plus fetch). Items marked **[unverified this session]** come from background knowledge and were NOT confirmed by a source fetched in this session. The report writer should treat them as leads, not citations. Everything else has an inline source.

Repo context for the writer: the repo decompiles small ReLU MLPs into Python. In 16 of 29 formula-task runs the program beat the net on the true task. On Iris the program was 96.7% correct vs 95.3% for the net. The proposed next step is: decompile one block, run it on failing cases, have an LLM agent fix the algorithm, and reinsert the block.

## Q1. Knowledge-based neural networks (KBANN, Fu, neural-symbolic cycle): what did insert -> train -> extract -> refine find about extracted rules vs the net?

### Takeaway
The 1990s "theory refinement" line already ran the full cycle: put rules into a net, train it on data, pull refined rules back out. The main finding was that extracted M-of-N rules **roughly match the network's test accuracy (they are not reliably better)**. They beat purely symbolic rule-refiners and are comprehensible. The neural-symbolic line (Garcez et al.) formalized the extraction step, proving soundness and measuring fidelity. Both lines report two separate quantities, accuracy on held-out data and fidelity to the net, and that split is the core of the evaluation design.

### Cited Findings
- **Towell & Shavlik, "Extracting refined rules from knowledge-based neural networks", Machine Learning 13:71-101 (1993), doi 10.1007/BF00993103.** The paper frames the work as a three-step cycle: insert knowledge into a net, refine it with standard training, extract the refined knowledge. The paper addresses step 3 with the NofM algorithm. Reported results: the extracted rules (1) "closely reproduce the accuracy of the network", (2) beat rules from methods that refine symbolic rules directly, (3) beat earlier net-to-rule extraction methods, and (4) are human-comprehensible — [mlanthology abstract](https://mlanthology.org/mlj/1993/towell1993mlj-extracting); [datalearner record](https://datalearner.com/academic/journal-papers/0885-6125/volumes-and-issues/13/paper-detail/10396)
- The NofM precursor appeared at NIPS 1991 ("Interpretation of Artificial Neural Networks: Mapping Knowledge-Based Neural Networks into Rules"). It searches for rules of the form "if N of these M antecedents are true", motivated by the observation that SUBSET-extracted rule sets contained N-of-M concepts — [NeurIPS 1991](https://papers.neurips.cc/paper_files/paper/1991/hash/ed265bc903a5a097f61d3ec064d96d2e-Abstract.html)
- The KBANN insertion step was first used on DNA gene recognition (Noordewier, Towell & Shavlik, NIPS 1990). The network's learning algorithm "adjusts the domain theory so that it properly classifies these DNA sequences" — [NeurIPS 1990](https://papers.neurips.cc/paper/1990/hash/8efb100a295c0c690931222ff4467bb8-Abstract.html)
- **Towell & Shavlik, "Knowledge-based artificial neural networks", Artificial Intelligence 70 (1994).** The UW promoter-dataset README attributes KBANN 877/936 = **93.70%** test accuracy to this paper. The same README lists the later ADDEMUP at 908/936 = 97.01% — [UW promoters README](https://ftp.cs.wisc.edu/machine-learning/shavlik-group/datasets/promoters-large/README)
- Splice-junction benchmark, as reported on the DELVE dataset page (10-fold CV, 1000 examples): KBANN error rates were 4.62 (Neither), 7.56 (EI) and 8.47 (IE), alongside backprop and other baselines. Protocol may differ from the AIJ paper — [DELVE splice](https://www.cs.toronto.edu/~delve/data/splice/spliceDetail.html)
- Datasets and code: the UW Shavlik-group FTP holds the promoter datasets — [UW FTP](https://ftp.cs.wisc.edu/machine-learning/shavlik-group/datasets/promoters-large/README). I found no maintained runnable KBANN or NofM implementation.
- **Craven & Shavlik, ICML 1994, "Using Sampling and Queries to Extract Rules from Trained Neural Networks".** The method extracts conjunctive and M-of-N rules by sampling and querying the net. It is more efficient than search-based extraction such as NofM — [mlanthology](https://mlanthology.org/icml/1994/craven1994icml-using)
- **LiMin Fu.** "Knowledge-based connectionism for revising domain theories", IEEE Trans. SMC 23(1):173-182 (1993), and "Rule Generation from Neural Networks", IEEE Trans. SMC 24(8) (1994). A Mooney-group review says Fu's approach used backprop to revise **certainty factors** but "was unable to revise the rule-base structure" — [Mahoney & Mooney RAPTURE, Connection Science 1993](https://www.cs.utexas.edu/~ml/papers/rapture-connsci-94.pdf); [dblp Fu](https://dblp.org/pid/22/5350)
- **RAPTURE (Mahoney & Mooney, Connection Science 5:339-364, 1993)** is a hybrid refiner. It uses modified backprop for certainty factors plus ID3 information gain to **add new rules**, so the "fix" can change structure, not just weights — [UT Austin PDF](https://www.cs.utexas.edu/~ml/papers/rapture-connsci-94.pdf)
- **d'Avila Garcez, Broda & Gabbay, "Symbolic knowledge extraction from trained neural networks: A sound approach", Artificial Intelligence 125(1-2), 2001, doi 10.1016/S0004-3702(00)00077-1.**
  - The paper extracts nonmonotonic rules and proves the method sound. It is sound **and complete** for "regular" networks.
  - Non-regular nets are handled decompositionally, and the authors report "very high fidelity" between the rules and the net.
  - Page ranges differ between records (153-205 vs 155-207).
  - [City Research Online](https://openaccess.city.ac.uk/id/eprint/293/); [BIU record](https://cris.biu.ac.il/en/publications/symbolic-knowledge-extraction-from-trained-neural-networks-a-soun/)
- An Imperial College 1998 tech-report precursor keeps soundness but drops completeness — [Imperial Spiral](https://spiral.imperial.ac.uk/handle/10044/1/95413)
- **Andrews, Diederich & Tickle, "Survey and critique of techniques for extracting rules from trained artificial neural networks", Knowledge-Based Systems 8:373-389 (1995).**
  - It introduced the decompositional / eclectic / pedagogical taxonomy and "criteria for evaluating their efficacy" — [datalearner](https://datalearner.com/academic/journal-papers/0950-7051/volumes-and-issues/34/paper-detail/5254); [Connectionists 1995 TR announcement](https://mailman.srv.cs.cmu.edu/pipermail/connectionists/1995-January/016033.html)
  - The criteria are commonly listed as accuracy, fidelity, consistency and comprehensibility **[exact wording unverified this session]**.
- **Zhou (2004)** critique, as summarized in a 2023 systematic review: "rule extraction using neural networks" (goal: an accurate rule set) and "rule extraction for neural networks" (goal: explain the net) have different aims. So fidelity should be dropped from the evaluation in the first case and accuracy in the second — [arXiv 2312.12878 review](https://arxiv.org/pdf/2312.12878)
- **[unverified this session]** d'Avila Garcez & Zaverucha, "The connectionist inductive learning and logic programming system" (CIL2P), Applied Intelligence 1999. It describes the neural-symbolic cycle: a logic program is translated into a net, trained, and rules are re-extracted. The book "Neural-Symbolic Learning Systems" (Springer 2002) has a chapter "Knowledge Extraction from Trained Networks" — [Springer chapter](https://link.springer.com/chapter/10.1007/978-1-4471-0211-3_5)

### Inferences
- The repo's finding that the decompiled program is *more* correct than the net in 16/29 runs goes further than KBANN/NofM, which reported extracted rules roughly *matching* net accuracy. The likely mechanism is that a discrete, low-description-length program (snapping 0.1/0.2/0.4) acts as a regularizer, as M-of-N rules did. The repo should report this the way Zhou suggests: "extraction *using* nets" (judge by task accuracy) vs "extraction *for* nets" (judge by fidelity). The two numbers must be reported separately, because a fix deliberately lowers fidelity.
- The old cycle reinserted *rules* as an initialization (KBANN) and then retrained the whole net. The user's plan reinserts a *corrected block* without retraining, which is closer to model repair/editing (Q4) than to KBANN.

### Gaps
- I could not fetch the AIJ 1994 KBANN full text or the MLJ 1993 tables. The exact NofM-vs-net accuracy numbers per domain (promoter, splice, ribosome binding) and whether the rules ever beat the net are unconfirmed.
- No runnable maintained KBANN/NofM/CIL2P code was found.
- Fu's papers were confirmed only through secondary references; there is no full text.

## Q2. LLM-in-the-loop / human-in-the-loop correction of model logic (XIL, RRR, GAM Changer, LLM repairs of glass-box models)

### Takeaway
Human correction of model logic is established in two forms:
- **Explanation-level feedback turned into a training constraint** (RRR, XIL/CAIPI, Schramowski et al.).
- **Direct editing of an additive model's shape functions** (GAM Changer, KDD 2022). It comes with live validation metrics, subgroup slices and a git-like edit history, and the authors explicitly warn that the safeguards cannot prevent overfitting or injected bias.

LLM participation is recent. GPT-4 can read GAM shape graphs, flag anomalies that contradict domain knowledge and **suggest repairs** (Lengerich et al. 2023), but the authors treat these as suggestions, not validated fixes. I found no paper where an LLM agent rewrites a decompiled block of a neural net and reinserts it. That exact step appears to be open.

### Cited Findings
- **Ross, Hughes & Doshi-Velez, "Right for the Right Reasons: Training Differentiable Models by Constraining their Explanations", arXiv 1703.03717 (IJCAI 2017 [venue unverified this session]).**
  - Method: penalize input gradients on regions that experts annotate as irrelevant.
  - Result: faithful explanations, and models "generalize much better when conditions differ between training and test".
  - An unsupervised variant finds diverse models with qualitatively different decision boundaries.
  - [arXiv](https://arxiv.org/abs/1703.03717)
- **Teso & Kersting, "Explanatory Interactive Machine Learning", AIES 2019, pp. 239-245.** The learner explains each active-learning query, and the user can correct the explanation as well as the label. CAIPI is the model-agnostic, LIME-based instance — [TU Darmstadt PDF](https://www.aiml.informatik.tu-darmstadt.de/papers/teso2019aies_XIML.pdf); [UniTN record](https://iris.unitn.it/handle/11572/370228). The longer arXiv version says CAIPI handles models that are "right (wrong) for the wrong reasons" — [arXiv 1805.08578](https://arxiv.org/pdf/1805.08578)
- The "CAIPI in Practice" follow-up (2022) reports roughly 80% less labeling effort (a follow-up claim, not from the 2019 paper) — [arXiv 2204.02661](https://arxiv.org/abs/2204.02661v1)
- **Schramowski, Stammer, Teso, Brugger, Shao, Luigs, Mahlein & Kersting, "Making deep neural networks right for the right scientific reasons by interacting with their explanations", arXiv 2001.05371.** XIL is applied to a plant-phenotyping task, and the stated aim is avoiding "Clever Hans" behaviour — [arXiv](https://arxiv.org/abs/2001.05371). Published in Nature Machine Intelligence 2020 **[venue and numbers unverified this session; the arXiv page did not list them]**.
- **Wang et al., "Interpretability, Then What? Editing Machine Learning Models to Reflect Human Knowledge and Values", KDD 2022 (ADS), doi 10.1145/3534678.3539074.** The preprint "GAM Changer" is arXiv 2112.03245 (NeurIPS 2021 workshop) — [arXiv 2206.15465](https://ar5iv.arxiv.org/abs/2206.15465); [MSR page](https://www.microsoft.com/en-us/research/publication/gam-changer-editing-generalized-additive-models-with-interactive-visualization/)
  - **Code:** [github.com/interpretml/gam-changer](https://arxiv.org/pdf/2112.03245) (MIT license), pip package `gamchanger` for notebooks — [PyPI](https://pypi.org/project/gamchanger/0.1.12). Live demo at interpret.ml/gam-changer.
  - **Evaluation safeguards:**
    - A live metric panel on validation samples: confusion matrix, accuracy, balanced accuracy and AUC for classifiers; RMSE, MAE and MAPE for regressors.
    - Metric scopes: **Global**, **Selected region** and **Slice (subgroup)**, so users can see whether an edit is local and fair.
    - Correlated-feature highlighting to catch indirect effects. Example: an age edit disproportionately affected women.
    - Monotonicity edits via isotonic regression ("smallest change").
    - Undo/redo and a git-style commit history.
    - [ar5iv](https://ar5iv.arxiv.org/abs/2206.15465)
  - **User study:** 7 data scientists (finance 4, healthcare 2, media 1), about 1 hour each, editing a LendingClub loan-default model. Ease of use averaged 6.14/7 and understandability 5.86/7. Monotonicity and deletion were rated the top tools — [ar5iv](https://ar5iv.arxiv.org/abs/2206.15465)
  - **Stated risks:**
    - The safeguards "do not guarantee to prevent users from overfitting the model, injecting harmful bias, or maliciously manipulating model predictions."
    - Global metrics are insensitive to small edits in a few bins, so local scopes are needed.
    - Some edits rest on tiny samples; in the sepsis example only 19 patients were below the edited threshold.
    - Some participants worried editing is "too easy". P2 preferred retraining.
    - [ar5iv](https://ar5iv.arxiv.org/abs/2206.15465)
- **Lengerich, Bordt, Nori, Nunnally, Aphinyanaphongs, Kellis & Caruana, "LLMs Understand Glass-Box Models, Discover Surprises, and Suggest Repairs", arXiv 2308.01157 (2023).** GPT-4 can describe, interpret and debug univariate GAM component graphs. It can detect anomalies that contradict prior knowledge, explain possible causes and **suggest repairs** that remove them. Because each component is processed separately, the whole model never has to fit in context — [arXiv](https://arxiv.org/abs/2308.01157v1); [ar5iv](https://ar5iv.org/pdf/2308.01157)
- Follow-up **"Data Science with LLMs and Interpretable Models"** (Bordt et al., arXiv 2402.14474) cautions that LLM answers "should be seen as helpful suggestions rather than final answers" — [arXiv](https://arxiv.org/pdf/2402.14474). Code: the TalkToEBM package (github.com/interpretml/TalkToEBM) **[repo URL unverified this session]**.
- **Liu, Yu, Lee, Hung, Chen & Sun, "Synthesizing Programmatic RL Policies with LLM Guided Search" (LLM-GS), ICLR 2025, arXiv 2405.16450.**
  - The LLM writes Python, which is converted into the Karel DSL (the "Pythonic-DSL" strategy). Scheduled Hill Climbing then refines the program against environment reward.
  - It is more sample-efficient than prior programmatic-RL methods on Karel.
  - [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/dcf887f2bfe2776584e3bce80ed206ef-Abstract-Conference.html); [arXiv](https://arxiv.org/abs/2405.16450v2)
  - This is the nearest published case of an LLM proposing *and* search-refining a program against task feedback.
- **[unverified this session]** Koh et al., "Concept Bottleneck Models" (ICML 2020). It supports test-time human *intervention* on intermediate concept values, which amounts to correcting an internal block's output without retraining.

### Inferences
- GAM Changer's evaluation design maps directly onto "fix one decompiled block". For each candidate fix the repo should show:
  1. task metrics on held-out data, global and restricted to the region where behaviour changed;
  2. the size of the changed region (how many training points or how much input volume);
  3. subgroup or slice effects;
  4. a reversible, versioned edit log.
- Because LLM repairs of glass-box models are framed as suggestions by their own authors, an LLM-fixed block needs the same automated gate as a human edit: held-out accuracy, locality, and support count in the edited region.

### Gaps
- No source found where an LLM agent edits a *decompiled neural block* and the block is put back into the network. The repo's proposal appears novel; a broader 2025-2026 search could still turn up something.
- Quantitative XIL results (Schramowski 2020) and the CAIPI numbers from the 2019 paper were not retrieved.
- The TalkToEBM repo was not verified.

## Q3. Program synthesis as replacement or guide for networks (PIRL/NDPS, VIPER, PROPEL, deep distilling): program vs network on reward and robustness

### Takeaway
Programmatic-RL work shows the cycle in its most complete form. PROPEL (NeurIPS 2019) alternates:
- a neural gradient step on (program + neural residual) — "lift";
- program synthesis by imitation of that mixed policy — "project".

The **program is the deployed artifact**. Results:
- The program is often *slower* (lower reward) than the pure neural DDPG policy on tracks where DDPG succeeds.
- The program is far more *robust*: it crashes much less in training and generalizes to unseen tracks where DDPG crashes.
- It is cheaply verifiable (Z3 check in 0.49 s).

VIPER extracts decision trees that match the DNN, then verifies them. A counterexample found by verification led to a fix of the *environment*. Deep distilling (Blazek et al.) reports net-derived code that generalizes out-of-distribution to problems far larger than training.

### Cited Findings
- **Verma, Le, Yue & Chaudhuri, "Imitation-Projected Programmatic Reinforcement Learning" (PROPEL), NeurIPS 2019, arXiv 1907.05431** — [NeurIPS](https://proceedings.neurips.cc/paper/2019/hash/5a44a53b7d26bb1e54c05222f186dcfb-Abstract.html); [arXiv](https://arxiv.org/abs/1907.05431)
  - Method: functional mirror descent. UPDATE takes a policy-gradient step (DDPG/TRPO) on a neural residual f added to a fixed program pi (h = pi + lambda*f). PROJECT synthesizes a new program that imitates h. Theoretical convergence is shown for approximate projection — [arXiv PDF](https://arxiv.org/pdf/1907.05431)
  - **Code:** [bitbucket.org/averma8053/propel](https://arxiv.org/pdf/1907.05431) (URL given in footnote 6 of the paper).
  - **Evaluation:**
    - 5 TORCS tracks; 25 seeds; median lap time plus crash ratio; a crash counts as infinite lap time; "CR" means a crash in more than half the seeds.
    - Also MountainCar and Pendulum, scored as mean over 100 episodes.
    - Baselines: hand-written PID PRIOR, DDPG, NDPS (PIRL), VIPER.
  - **TORCS Table 1** (my reading of pdftotext output with garbled column layout; cross-check before quoting):
    - PRIOR finishes all tracks slowly (312.92 / 322.59 / 244.19 / 340.29 / 402.89 s, crash ratio 0).
    - DDPG is fast where it works (78.82 s G-Track, 89.71 s E-Road, 101.06 s Aalborg) but crashes in more than half the seeds on Ruudskogen and Alpine-2 (crash ratios 0.68 and 0.92).
    - PROPEL variants complete all tracks with crash ratios of about 0.04-0.36. Values include 93.67/119.17/147.28/124.58/256.59 s and 78.33/79.39/109.83/118.80/236.01 s; which row is PROPELPROG vs PROPELTREE is ambiguous in the extraction.
    - [arXiv PDF](https://arxiv.org/pdf/1907.05431)
  - **Paper's own summary:**
    - PROPELPROG and PROPELTREE beat NDPS and VIPER respectively.
    - "While DDPG outperforms PROPEL on some tracks, its volatility causes it to be unable to learn in some environments, and hence to crash the majority of the time."
    - It suffers "far fewer crashes while training".
    - [arXiv PDF](https://arxiv.org/pdf/1907.05431)
  - **Generalization (Table 2, train on one track and test on others):** "DDPG crashes significantly more often than PROPELPROG". For example, the policy trained on G-Track completed E-Road in 124 s while DDPG crashed (CR). PROPELTREE generalizes better than DDPG but worse than PROPELPROG. The authors concede generalization "still leaves much room for improvement" — [arXiv PDF](https://arxiv.org/pdf/1907.05431)
  - **Verifiability:** a smoothness property of the PROPELPROG policy was verified with Z3 in 0.49 s — [arXiv PDF](https://arxiv.org/pdf/1907.05431)
  - **Initialization matters:** PROPEL initialized from a DDPG neural policy also fails on multiple tracks. Initialized from a simple hand-crafted program it often finishes them. This is a cycle that starts from a human-written (or LLM-written) program — [arXiv PDF](https://arxiv.org/pdf/1907.05431)
- **PIRL / NDPS: Verma, Murali, Singh, Kohli & Chaudhuri, "Programmatically Interpretable Reinforcement Learning", ICML 2018, pp. 5052-5061.** Cited as ref [58] in PROPEL — [arXiv PDF ref list](https://arxiv.org/pdf/1907.05431). **[results unverified this session]**
- **Bastani, Pu & Solar-Lezama, "Verifiable Reinforcement Learning via Policy Extraction" (VIPER), NeurIPS 2018, arXiv 1805.08328** — [NeurIPS](https://proceedings.neurips.cc/paper/2018/hash/e6d8545daa42d5ced125a4bf747b3688-Abstract.html); [arXiv](https://arxiv.org/abs/1805.08328v2)
  - It trains decision trees from a DNN oracle and its Q-function, with Q-weighted DAgger.
  - The trees are under 1,000 nodes. They match the DNN's performance on symbolic Atari Pong and a toy Pong game, are provably robust (Pong), and provably never lose (toy).
  - Verification found a counterexample in the toy-game tree, fixed by "slightly lengthening the paddle".
  - Code: github.com/obastani/viper **[unverified this session]**.
- **Blazek, Venkatesh & Lin, "Deep Distilling: automated code generation using explainable deep learning", arXiv 2111.08275 (2021); Nature Computational Science 2024 [journal version unverified this session].**
  - Method: train "essence neural networks" (ENNs), then translate their parameters into concise executable code.
  - The code "generalizes out-of-distribution to solve problems orders-of-magnitude larger and more complex than the training data". Where a ground-truth rule set exists, the method recovers it exactly.
  - [arXiv](https://arxiv.org/abs/2111.08275v1)
  - Code: [github.com/pauljblazek/deepdistilling](https://github.com/pauljblazek/deepdistilling), whose README claims the code "outperform[s] human-designed algorithms" and that it tests the code on several test sets.
- Related 2026 toolkits (net block to symbolic or code, no LLM fixing):
  - **SymTorch** (arXiv 2602.21307) wraps a PyTorch module and runs PySR symbolic regression on its I/O — [arXiv](https://arxiv.org/html/2602.21307).
  - **"Weights to Code: Extracting Interpretable Algorithms from the Discrete Transformer"** (arXiv 2601.05770); method details not retrieved — [arXiv](https://arxiv.org/pdf/2601.05770).
  - **symbolic-circuit-distillation** (community GitHub) turns pruned circuits into executable programs. It enumerates templates, filters them by agreement with the circuit, and formally verifies the equivalence — [GitHub](https://github.com/neelsomani/symbolic-circuit-distillation).
- **LLM-GS** (ICLR 2025, Q2 above): an LLM proposes programs and hill-climbing on reward repairs them — [arXiv](https://arxiv.org/abs/2405.16450v2).

### Inferences
- PROPEL is the closest published design to the user's idea, with two differences. It uses *imitation* (fit the program to the current mixed policy) rather than *fixing on failing cases*, and it keeps a neural residual in the loop. An "LLM fixes the block" variant would replace PROJECT with "LLM edits the program, scored on task loss plus imitation of the net elsewhere". The neural residual could then be retrained around the fixed block, which is the "lift" step.
- The evaluation template is: task metric on the training environments, plus an out-of-distribution transfer matrix, plus failure/crash rate, plus a formal property check. It transfers directly to the repo: held-out accuracy, accuracy under shifted inputs, rate of catastrophic outputs, and properties such as monotonicity or bracket snapping checked symbolically.
- The repo's "program beats net on the true task" finding agrees with deep distilling and PROPEL. Discrete programs regularize and extrapolate differently than nets. They can also look worse on the in-distribution metric where the net is already good (as DDPG was on easy tracks).

### Gaps
- The exact row assignment in PROPEL Table 1 is uncertain (garbled PDF layout). Verify against the PDF before quoting per-track numbers.
- NDPS/PIRL and VIPER numeric results and code URLs were not fetched.
- The deep distilling journal version and its quantitative OOD results were not retrieved.

## Q4. Risks and evaluation: overfitting the fix, distribution shift, locality (agreement with the original net elsewhere), robustness

### Takeaway
The model-repair/editing literature provides a standard three-part metric: **efficacy** on the repair set, **generalization** to held-out inputs from the same failure class, and **drawdown** (accuracy lost on inputs the net got right). The editing literature adds **locality** (unchanged outputs on unrelated inputs). Provable Repair (PLDI 2021) shows the trade-off concretely:
- Fine-tuning to fix failures reaches 100% efficacy but loses 8-15% on ImageNet originally-correct inputs.
- Its layer-local LP repair achieves 0-5% drawdown.

GAM Changer and Zhou's critique add that fidelity and accuracy must be reported separately, that tiny-support edits are a red flag, and that global metrics hide local damage.

### Cited Findings
- **Sotoudeh & Thakur, "Provable Repair of Deep Neural Networks", PLDI 2021, arXiv 2104.04413** — [ar5iv](https://ar5iv.labs.arxiv.org/html/2104.04413); [PLDI page](https://pldi21.sigplan.org/details/pldi-2021-papers/39/Provable-Repair-of-Deep-Neural-Networks); [author page](https://thakur.cs.ucdavis.edu/bibliography/PLDI2021.html)
  - **Method:** repairs one layer of a "Decoupled DNN" by linear programming, so a specification holds on a finite point set or on convex polytopes (infinitely many points).
  - **Code:** [github.com/95616ARG/PRDNN](https://ar5iv.labs.arxiv.org/html/2104.04413).
  - **Metrics:**
    - Efficacy: accuracy on the repair set (guaranteed 100%).
    - Generalization: repaired minus buggy accuracy on a held-out set of similar buggy inputs.
    - Drawdown: buggy minus repaired accuracy on a test set of originally-correct inputs.
    - The authors define drawdown as "the network loses what it learned from the original, larger training set" and say fine-tuning carries a higher risk of it.
  - **ImageNet SqueezeNet** (natural adversarial examples; buggy accuracy 18.6%; repair sets of 100-752 images):
    - Provable Repair: 100% efficacy, 1.1-5.3% drawdown, 1.6-8.5 min.
    - Fine-tuning (FT): 100% efficacy, **8.2-15.4% drawdown**.
    - Modified FT (early-stopped and conservative): about 0% drawdown but at most 28% efficacy.
  - **MNIST ReLU-3-100** (fog-corrupted line segments):
    - Provable Repair: 2.4-5.9% drawdown, 42.9-46.0% generalization.
    - FT: up to 56% drawdown in some configs, but one FT config got 3.2% drawdown and 60% generalization. **Fine-tuning sometimes generalizes better.**
  - **ACAS Xu N2,9** (10 polytope slices):
    - Provable Repair: 100% efficacy, **0 drawdown** (all 5,466 points stayed correct), 94.69% generalization, 21 s.
    - FT broke 181 repair points (the original net got only 3 wrong), broke 650/5,466 previously-correct points, and timed out after 1h18m.
- **Sinitsin et al., "Editable Neural Networks", ICLR 2020, arXiv 2004.00345.**
  - "Editable Training" makes later single-mistake patches fast and local.
  - **Locality** means keeping predictions on unrelated inputs unchanged.
  - Drawdown is the reported side-effect metric — [ar5iv](https://ar5iv.arxiv.org/html/2004.00345); [mlanthology](https://mlanthology.org/iclr/2020/sinitsin2020iclr-editable)
  - Secondary-review figures (RMSProp editor about 1.77% drawdown, 100% success): **unverified against the paper's tables** — [review](https://liner.com/review/editable-neural-networks)
  - MEND (ICLR 2022) later measures locality on randomly sampled unrelated inputs — per the search summary of [MEND text](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/ICLR/2022/fast_model_editing_at_scale__934cb786.txt) **[not directly read]**.
- **Fidelity vs accuracy:** Zhou (2004, via review) argues that the two must be evaluated according to the goal. When the goal is a better model, fidelity to the net is not a quality criterion. When the goal is to explain the net, accuracy is not — [arXiv 2312.12878](https://arxiv.org/pdf/2312.12878)
- **Overfitting and small support (GAM Changer):**
  - The safeguards cannot prevent overfitting or bias injection.
  - Edits can rest on very few samples (19 patients in the sepsis example).
  - Edits can shift subgroups unevenly (age edit vs sex).
  - Global metrics barely move for local edits.
  - [ar5iv](https://ar5iv.arxiv.org/abs/2206.15465)
- **Generalization of programs vs nets out of distribution:**
  - PROPEL: cross-track transfer matrix; DDPG crashes far more on unseen tracks — [arXiv PDF](https://arxiv.org/pdf/1907.05431).
  - RRR: models constrained by explanations generalize better when train and test conditions differ — [arXiv](https://arxiv.org/abs/1703.03717).
  - Deep distilling: code generalizes to much larger instances — [arXiv](https://arxiv.org/abs/2111.08275v1).
- **Verification found the bug, and the fix was outside the model:** VIPER's counterexample was fixed by lengthening the paddle, i.e. by changing the environment — [search summary of arXiv full text](https://arxiv.org/abs/1805.08328v2).
- **[unverified this session]** Hein, Andriushchenko & Bitterwolf, "Why ReLU networks yield high-confidence predictions far away from the training data and how to mitigate the problem", CVPR 2019. ReLU nets are piecewise-linear, so far from the data they become arbitrarily confident. This bears directly on Iris nets "extrapolating arbitrarily" outside the data hull, and it is why agreement with the net should be measured only on the data support.

### Inferences
Proposed evaluation protocol for the repo's "fix one block" step, synthesized from the sources above:
1. **Split failing cases** into a repair set (shown to the LLM agent) and a held-out failure set (never shown). Report efficacy on the first and **generalization** on the second (Sotoudeh & Thakur).
2. **Drawdown / locality:** on inputs the original net got right, report the accuracy lost and the % of outputs changed. Compute it separately on (a) the training-data support (e.g. Iris convex hull / k-NN neighbourhood) and (b) off-support samples. Off-support disagreement with the net is not evidence against the fix, because the net's extrapolation is arbitrary (Hein et al. [unverified]). Report it, but don't penalize it.
3. **True-task accuracy on fresh held-out data** (new seeds or draws for formula tasks; repeated CV for Iris, since 150 samples means about 1.5 flowers is 1 percentage point). This is the "extraction *using* nets" criterion (Zhou).
4. **Support count of the edited region** (how many training points fall where the block's behaviour changed), following GAM Changer's small-sample warning. Flag fixes justified by fewer than about 10 examples.
5. **Description-length or complexity guard:** reject fixes that add case-by-case special-casing (if-input-equals-X branches). That is overfitting to the repair set. Compare program size before and after.
6. **Reinsertion check:** with the fixed block back in the net, compare end-to-end metrics against (i) the original net, (ii) the unfixed decompiled program, (iii) fine-tuning the net on the same repair set — the FT baseline used in Provable Repair, which can win on generalization. Optionally, PROPEL-style: retrain the downstream layers around the frozen fixed block and report again.
7. **Seeds/variance:** report the median over seeds plus failure counts, as PROPEL does (25 seeds, median, crash ratio). The repo already uses held-out seeds (iris seeds 3-5).
8. **Property checks** where possible: monotonicity, snapping to bracket values, class-region convexity. Do these symbolically or by exhaustive grid on the small input space, as in PROPEL's Z3 check and VIPER's verification.

The specific risk of an LLM-agent fixer: it sees the failing cases and the code, so it can memorize them. Steps 1, 4 and 5 are the defences. The LLM-GS and Lengerich works treat LLM output as proposals to be search-validated or human-validated, not as accepted fixes.

### Gaps
- No paper found that evaluates an LLM-edited, reinserted neural sub-block with these metrics. The protocol above is a synthesis, not an established standard.
- Editable NN / MEND tables were not read directly.
- Hein et al. 2019 was not fetched; its citation details are from memory.
- I found no empirical study of "fix overfitting" specific to rule or program edits beyond GAM Changer's qualitative warnings.
