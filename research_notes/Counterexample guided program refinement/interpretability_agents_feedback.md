# Interpretability agents and LLM explanation/program-induction methods with hypothesis-test-revise loops, and how the loop is ablated

Scope note: these notes were gathered on 2026-10-27 from arXiv HTML/abstract pages, proceedings listings and project blogs. Numbers marked "(extracted)" were read from paper HTML by an automated page reader and should be spot-checked against the PDF table before they are quoted in a paper. Venues marked "unverified" were not confirmed by a source fetched in this session.

## 1. FIND benchmark (Schwettmann et al. 2023): interactive agents versus one-shot description, and failing inputs

### Takeaway
FIND is the closest interpretability prior work to the repo's setup. A black-box function is given to an LLM "automated interpretability agent" (AIA), which runs it on inputs it chooses and writes a code or language description. The paper compares non-interactive description from precomputed exemplars, interactive-only AIA, and interactive AIA started from exemplars. It does not ablate giving the agent failing or counterexample inputs; the agent only sees outputs of its own queries. Interactivity helped most when combined with good initial exemplars. GPT-4 recovered exact code for only about a third of numeric functions.

### Cited Findings
- Authors: Sarah Schwettmann, Tamar Rott Shaham, Joanna Materzynska, Neil Chowdhury, Shuang Li, Jacob Andreas, David Bau, Antonio Torralba. arXiv:2309.03886 (v3, Dec 2023). Code: github.com/multimodal-interpretability/FIND (MIT). Data: Zenodo 10.5281/zenodo.8034162. — [arXiv HTML](https://arxiv.org/html/2309.03886). The venue (NeurIPS 2023 Datasets and Benchmarks, per the brief) was not confirmed by a page fetched here.
- Agent protocol: the AIA calls `PYTHON(function.py input)` on inputs it chooses, forms and tests hypotheses, and updates its description. Three settings: non-interactive "milan" (a description from fixed precomputed exemplars), AIA (no pre-selected inputs) and AIA+milan (starts from exemplars, then experiments). — [arXiv HTML](https://arxiv.org/html/2309.03886)
- Scoring:
  - Numeric code descriptions are scored by normalized MSE against the true function on [-128, 128]; success means NMSE < 0.1.
  - String code descriptions are scored by exact match on 10 test inputs.
  - Language descriptions are scored by a "unit test": a fine-tuned Vicuna-13b evaluator must pick the true input/output pair from distractors. — [arXiv HTML](https://arxiv.org/html/2309.03886)
- Table 2 success rates (extracted):
  - GPT-4 AIA: numeric code 0.33, string code 0.23, string language unit test 0.82.
  - GPT-3.5 AIA: 0.12, 0.13 and 0.66.
  - Llama-2-13b/70b chat AIA: about 0 on code. — [arXiv HTML](https://arxiv.org/html/2309.03886)
- One-shot versus interactive (extracted, reported as AIA / milan / AIA+milan):
  - Entities, GPT-4: 0.56 / 0.89 / 0.89.
  - Entities, GPT-3.5: 0.39 / 0.81 / 0.88.
  - Relations, GPT-4: 0.78 / 0.74 / 0.92.
  - Relations, GPT-3.5: 0.37 / 0.64 / 0.68.
  - So interactive-only lost to the exemplar-based one-shot on entities and won slightly on relations. The combination was best or tied. — [arXiv HTML](https://arxiv.org/html/2309.03886)
- Mean dialogue length was about 4.1 interactions for GPT-4, 2.1 for GPT-3.5 and 1.4 for the Llama-2 models. Llama-2 often hallucinated function outputs before any were returned. — [arXiv HTML](https://arxiv.org/html/2309.03886)
- Failure mode: agents can miss small corruptions confined to part of a function's input domain (noise or out-of-distribution patches). The authors note these could stand in for training-set bias in real models. — [arXiv HTML](https://arxiv.org/html/2309.03886)
- The unit-test score was much more forgiving than exact-match code scoring (string functions, GPT-4: 0.82 versus 0.23). — [arXiv HTML](https://arxiv.org/html/2309.03886)

### Inferences
- FIND's "localized domain corruption" failure is exactly the case a counterexample-returning checker targets. An agent that picks its own queries tends not to probe the corrupted region. This gives direct motivation for checker-supplied counterexamples over self-directed probing alone.
- FIND's design gives three reusable ablation arms:
  - exemplars only (one-shot)
  - self-directed queries only
  - exemplars plus queries
  
  For the repo, the analogues are:
  - net weights plus a sample table only (one-shot)
  - agent can query the net
  - agent gets checker counterexamples
- FIND's numeric metric (NMSE < 0.1 on a grid) is weaker than exact agreement. The repo's exact or decision-agreement checks are stricter and should be reported as such.

### Gaps
- No FIND experiment gives the agent checker-produced failing inputs or counts revision rounds. I found no follow-up FIND paper that does this.
- The headline "fraction of functions the best AIA fails on" was garbled in the extracted text and is not reported here.
- The interactive-versus-one-shot numbers are for language-description categories (entities, relations). The extracted table showed no milan baseline for numeric or string code.

## 2. MAIA, simulation-scored neuron explanations (Bills et al.), rigorous assessment (Huang et al.), and revising explanations with counterexamples

### Takeaway
MAIA (2024) generalizes FIND's AIA to vision models, with tools for synthesizing and editing inputs. Its tool ablation shows that exemplars alone and generated inputs alone each underperform the combination. Bills et al. (OpenAI 2023) state that asking GPT-4 to propose counterexamples and then revising explanations in light of their activations raises simulation scores. This is the earliest counterexample-driven explanation-revision result in interpretability, but the public summary does not quantify it. Huang et al. (2023) show that one-shot GPT-4 neuron explanations have high error rates under observational and interventional tests. More recent work (InterpAgent 2026, a 2025 LessWrong agent, LINE 2026) runs explicit hypothesize-test-revise loops for features, and reports that most gains come in the first one or two revision rounds.

### Cited Findings
- MAIA: Tamar Rott Shaham, Sarah Schwettmann, Franklin Wang, Achyuta Rajaram, Evan Hernandez, Jacob Andreas, Antonio Torralba. arXiv:2404.14394 (v2 Feb 2025). Project page: multimodal-interpretability.csail.mit.edu/maia. — [arXiv HTML](https://arxiv.org/html/2404.14394). The venue (ICML 2024) was not stated on the fetched page and is unverified here.
- MAIA design: a GPT-4V backbone with these tools:
  - dataset_exemplars
  - text2image (Stable Diffusion v1.5, or DALL-E 3 in an ablation)
  - edit_images (InstructPix2Pix)
  - describe_images and summarize_images, which run in fresh GPT-4V instances to limit confirmation bias
  - log_experiment

  — [arXiv HTML](https://arxiv.org/html/2404.14394)
- MAIA evaluation on 85 synthetic neurons with known ground truth:
  - In 2AFC human judgments, MAIA descriptions were preferred over MILAN (0.73) and roughly tied with human labels (0.53).
  - On real neurons (about 100 units each from ResNet-152, DINO and CLIP), MAIA descriptions predicted activations better than MILAN's. The values appear only in a figure. — [arXiv HTML](https://arxiv.org/html/2404.14394)
- MAIA tool ablation: dataset exemplars only, or generated inputs only, each underperformed the full system. The best result was both together with DALL-E 3. Exact values appear in a figure only. — [arXiv HTML](https://arxiv.org/html/2404.14394)
- MAIA downstream task, spurious-feature removal on ResNet-18/Spawrious: test accuracy 0.837 using 22 MAIA-selected neurons, against 0.731 for the original model, 0.786 for MILAN and 0.757 for ℓ1 top-22. — [arXiv HTML](https://arxiv.org/html/2404.14394)
- MAIA failure modes the authors acknowledge: confirmation bias, conclusions drawn from small samples, and image-tool failures. Its outputs "aren't formal verification." — [arXiv HTML](https://arxiv.org/html/2404.14394)
- Bills et al. 2023, "Language models can explain neurons in language models" (OpenAI, 9 May 2023): Steven Bills, Nick Cammarata, Dan Mossing, Henk Tillman, Leo Gao, Gabriel Goh, Ilya Sutskever, Jan Leike, Jeff Wu, William Saunders. The paper has a "Revisions" results section and an "ablation scoring" section; revision experiments were implemented by Henk Tillman. — [paper page](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)
- The OpenAI announcement says scores can be increased "by asking GPT-4 to come up with possible counterexamples, then revising explanations in light of their activations." The vast majority of explanations score poorly. — [OpenAI blog](https://openai.com/research//language-models-can-explain-neurons-in-language-models) (as summarized by search; the paper's Revisions section did not render in the fetch)
- A NeurIPS 2023 workshop paper, "The Importance of Prompt Tuning for Automated Neuron Explanations," reports that reformatting the explainer prompt improves explanation quality and cuts cost. — [NeurIPS virtual page](https://neurips.cc/virtual/2023/83391)
- Huang, Geiger, D'Oosterlinck, Wu, Potts, "Rigorously Assessing Natural Language Explanations of Neurons," BlackboxNLP 2023 (ACL Anthology 2023.blackboxnlp-1.24, pp. 317–331). arXiv:2309.10312. — [ACL Anthology](https://preview.aclanthology.org/fix_video/2023.blackboxnlp-1.24/)
  - It proposes an observational mode (does the neuron fire on all and only inputs matching the concept?) and an intervention mode (does the neuron causally mediate the concept?).
  - Applied to GPT-4 explanations of GPT-2 XL neurons, "even the most confident explanations have high error rates and little to no causal efficacy." It evaluates explanations; it does not revise them. — [arXiv](https://arxiv.org/abs/2309.10312)
- InterpAgent: "Automated Interpretability and Feature Discovery in Language Models with Agents," Marin-Llobet, Ferrando et al. (other co-authors may be missing from the truncated page). arXiv:2605.01555, May 2026. Code: github.com/arnaumarin/InterpAgent. — [arXiv HTML](https://arxiv.org/html/2605.01555v1)
  - Loop: for each hypothesis, generate 12 predicted-positive prompts and 12 matched negative controls, run them through the model, and score on seven metrics (Detection F1, Fuzzing F1, Surprisal AUROC, and others). The agent revises to target the weakest metrics. It stops at a budget of about 4–5 iterations, or early when no new counterexamples appear.
  - Refined hypotheses beat the baseline label in 63.5% of features with GPT-4o-mini (1,600 features), 79.2% with Gemini 2.5 Pro (300) and 83.3% with Claude 4 Sonnet (60).
  - Per-iteration win rates show that most gains come in iterations 1–2.
  - There is no formal ablation with versus without testing. — [arXiv HTML](https://arxiv.org/html/2605.01555v1)
- LessWrong "Automating Interpretability with Agents" (blog, not peer reviewed):
  - It reports that Delphi automated feature explanations fail 38% of the time.
  - It builds an agent that generates should-activate and should-not-activate examples (counterexample-style tests), runs causal experiments and refines.
  - It reports performance "on par with vanilla Delphi." — [LessWrong](https://www.lesswrong.com/posts/PH8dq6jGozGkpsCzq/automating-interpretability-with-agents-2) (via search summary)
- LINE (arXiv 2604.08039, 2026): an iterative LLM neuron-labeling method for vision models. The LLM proposes new concepts conditioned on previously evaluated ones. The LLM-generated label wins for 22% (ImageNet-1K) to 38% (Places365) of neurons on average. — [arXiv HTML](https://arxiv.org/html/2604.08039v1) (via search summary)
- EleutherAI's "Automatically Interpreting Millions of Features in LLMs" (arXiv 2410.13928) introduced detection and fuzzing scores, which later refinement loops use as feedback metrics. — [arXiv](https://arxiv.org/html/2410.13928) (not fetched in detail)

### Inferences
- The interpretability literature mostly scores explanations with statistical proxies: simulation correlation, detection/fuzzing F1 and 2AFC human preference. It does not use exact functional equivalence. The repo's checker, which returns exact counterexample inputs, is a stronger and more program-synthesis-like signal than anything in this line.
- Huang et al.'s observational mode (all and only matching inputs) is essentially a precision/recall check. The repo's counterexamples are the false positives and false negatives of that check, in input space.
- Bills et al.'s counterexample revision is the earliest direct precedent for "revise using counterexamples," but its effect size is not reported in an accessible form.

### Gaps
- The quantitative results of Bills et al.'s "Revisions" section (score deltas, number of rounds, whether the counterexamples were LLM-proposed only) could not be retrieved because the paper page did not render its body in the fetch. They need a manual read.
- I found no interpretability paper that runs a clean ablation of {no feedback, pass/fail only, counterexamples} for explanation revision.
- MAIA's tool-ablation numbers and its ICML 2024 venue are unverified here.

## 3. LLM program induction from input/output examples with execution feedback: rounds and feedback content

### Takeaway
In program induction, the strongest evidence is that execution feedback from an external interpreter is essential, while the model's own "interpretation" or self-critique helps little or hurts. Most of the gain from refinement arrives in the first one to three rounds. In repair-versus-resample comparisons at matched cost, more diverse initial samples usually beat more repair turns per sample. Feedback quality is the bottleneck: human or stronger-model feedback roughly 1.6x the repair success rate.

### Cited Findings
- Hypothesis Search: Ruocheng Wang, Eric Zelikman, Gabriel Poesia, Yewen Pu, Nick Haber, Noah D. Goodman. ICLR 2024. arXiv:2309.05660. Code: github.com/Relento/hypothesis_search. — [arXiv](https://arxiv.org/abs/2309.05660)
  - Method: the LLM proposes natural-language hypotheses, implements them as Python, and verifies them on the training examples.
  - ARC (random 100-task subset): 30% with LLM-summarized hypotheses, 33% with human-selected hypotheses, against 17% for direct prompting.
  - Ablations show that both the abstract hypotheses and the concrete program representation help. Detailed ablation numbers were not retrieved.
- Qiu et al., "Phenomenal Yet Puzzling: Testing Inductive Reasoning Capabilities of Language Models with Hypothesis Refinement." arXiv:2310.08559. Code: github.com/linlu-qiu/lm-inductive-reasoning. — [arXiv HTML](https://arxiv.org/html/2310.08559). The venue (ICLR 2024) was not shown on the fetched page and is unverified here.
  - Loop: N hypotheses per iteration, scored on the seen exemplars by a symbolic interpreter; the best one is kept. Feedback is a template listing each exemplar the hypothesis gets wrong as input, expected output and the hypothesis's actual output. That is literally counterexample feedback. The loop stops when all exemplars pass or at the maximum T. T ∈ {1, 3}, N ∈ {1, 5}, GPT-4-0613. — [arXiv HTML](https://arxiv.org/html/2310.08559)
  - Results, task accuracy for T=1,N=5 → T=3,N=5 (extracted):
    - ACRE 48.0 → 59.0
    - MiniSCAN 70.0 → 85.0
    - List Functions 52.4 → 61.2
    - MiniARC 9.2 → 14.6
    - T=3,N=1 reached 95.0 on MiniSCAN, beating T=3,N=5 (85.0); the authors attribute this to overfitting the exemplars.
    - IO prompting baselines: 28.0 / 0.0 / 39.6 / 13.8. — [arXiv HTML](https://arxiv.org/html/2310.08559)
  - Feedback source matters. Self-refinement with the LM as its own interpreter (SR, T=3,N=5) reached only 32.0 / 0.0 / 52.0 / 9.2. Self-consistency sampling (N=5, majority vote) did not beat plain IO. When the LM applies its own rules, MiniSCAN task accuracy collapses from over 80% to near zero. — [arXiv HTML](https://arxiv.org/html/2310.08559)
  - There is no ablation over feedback format (for example pass/fail versus listed counterexamples). — [arXiv HTML](https://arxiv.org/html/2310.08559)
- Self-Debugging: Xinyun Chen, Maxwell Lin, Nathanael Schärli, Denny Zhou. arXiv:2304.05128. — [arXiv](https://arxiv.org/abs/2304.05128). The venue (ICLR 2024) is not on the arXiv page and is unverified here.
  - This is the cleanest feedback-content ablation found. Table 2 (extracted) compares Baseline / Simple (a correct-or-wrong signal) / UT (unit-test results) / UT+Expl / UT+Trace.
    - TransCoder, GPT-4: 77.3 / 80.9 / 88.8 / 90.4 / 89.5.
    - TransCoder, Codex: 80.4 / 89.3 / 91.6 / 92.5 / 87.9.
    - MBPP, GPT-4: 72.8 / 78.8 / 80.6 / 80.4 / 80.2.
    - MBPP, GPT-3.5: 67.6 / 70.8 / 72.2 / 74.2 / 72.8. — [arXiv HTML](https://arxiv.org/html/2304.05128v2)
  - The maximum is 10 debugging turns. Successful debugging "mostly ends within 3 turns," and "the major improvement comes from the first debugging turn." — [arXiv HTML](https://arxiv.org/html/2304.05128v2)
  - Without execution, simple feedback can hurt: Codex on MBPP scored 57.6, against a 61.4 baseline. Reusing failed programs plus feedback can match baselines that sample more than 10x as many programs. — [arXiv HTML](https://arxiv.org/html/2304.05128v2); [arXiv abstract](https://arxiv.org/abs/2304.05128)
- Olausson, Inala, Wang, Gao, Solar-Lezama, "Is Self-Repair a Silver Bullet for Code Generation?" ICLR 2024. arXiv:2306.09896. — [arXiv](https://arxiv.org/abs/2306.09896)
  - They compare self-repair against i.i.d. sampling at matched token budgets (pass@t).
  - Gains are "often modest," vary across subsets, and are sometimes absent. GPT-4 on APPS, 10 initial samples with 1 repair each: 1.05x the pass@20 baseline. With 2 initial samples and 10 repairs each: 0.97x.
  - Increasing initial samples n_p consistently helps. Increasing repairs per sample n_fr is often not worth it, and n_fr=1 was the best overall choice. — [arXiv HTML](https://arxiv.org/html/2306.09896v5)
  - Feedback quality is the bottleneck. GPT-4 repair success was 33.3% with its own feedback and 52.6% with human feedback (1.58x). GPT-4's feedback was inaccurate in 32/80 cases, against 7/80 for humans. Stronger-model feedback made Code Llama and GPT-3.5 beat their i.i.d. baselines. — [arXiv HTML](https://arxiv.org/html/2306.09896v5)
  - Recommended design: measure cost in tokens rather than sample counts, compare against i.i.d. at equal budget, vary n_p and n_fr separately, and use bootstrapped estimates from a frozen repair tree. — [arXiv HTML](https://arxiv.org/html/2306.09896v5)
- Tang, Hu, Zhou, Zhong, Zheng, Si, Ellis, "Code Repair with LLMs gives an Exploration-Exploitation Tradeoff" (REx). NeurIPS 2024. arXiv:2405.17503. — [NeurIPS poster](https://neurips.cc/virtual/2024/poster/93642); [arXiv](https://arxiv.org/abs/2405.17503)
  - It frames choosing which program to refine (the one passing the most tests, or a less-explored one) as an arm-acquiring bandit solved with Thompson sampling.
  - It solves more problems with fewer LLM calls on loop invariants, ARC and APPS: typically 1.5x–5x fewer calls across domains.
  - ARC-specific numbers (about 1.1–1.6x fewer calls, 64-call budget, Hypothesis Search as the baseline) come from a third-party review only and are unverified. — [liner review](https://liner.com/review/code-repair-with-llms-gives-an-explorationexploitation-tradeoff)
- Greenblatt (Redwood Research, June 2024 blog, not peer reviewed): about 8,000 GPT-4o Python programs per ARC task. The 12 most promising are revised by showing GPT-4o what each program actually outputs on the examples.
  - It scored 50% on a 100-task sample of the ARC-AGI public eval.
  - "To reach 50% with just samples from V2, I would need 100,000 samples," so revision was doing "a huge amount of work." — [Redwood blog](https://blog.redwoodresearch.org/p/getting-50-sota-on-arc-agi-with-gpt) (quoted via search summary)

### Inferences
- Qiu et al.'s feedback template (input, expected, got) is nearly identical to the repo's checker output. Their T=1 → T=3 deltas (+9 to +15 task-accuracy points) are the closest published estimate of the value of counterexample rounds for rule induction.
- Self-Debugging's Simple → UT gap is the clearest evidence that content beyond pass/fail matters, at about +8 points on TransCoder with GPT-4. Its finding that pass/fail alone without execution can hurt supports including a "pass/fail only" arm.
- Olausson et al. imply that a fair claim about "the counterexample loop helps" needs a budget-matched i.i.d.-resampling baseline. Without one, gains may just reflect more samples.

### Gaps
- None of these papers adds "boundary points" (inputs near decision boundaries) to the counterexamples. I found no direct precedent for that arm. The closest are Qiu's failing-exemplar lists and FIND's localized-corruption analysis.
- Hypothesis Search's detailed ablation numbers, including any execution-feedback revision rounds on ARC, were not retrieved.

## 4. LLM-guided symbolic regression with feedback loops (LLM-SR, LaSR)

### Takeaway
LLM-SR (ICLR 2025 oral) is a generate-fit-score-revise loop over Python equation skeletons with an evolutionary experience buffer. Its ablations show that removing the skeleton-plus-optimizer program representation or the iterative loop raises NMSE by about six orders of magnitude. LaSR (NeurIPS 2024) adds an LLM-learned "concept library" that conditions evolutionary search. In both, feedback is a scalar score (MSE) plus selected prior programs, not counterexamples.

### Cited Findings
- LLM-SR: Parshin Shojaee, Kazem Meidani, Shashank Gupta, Amir Barati Farimani, Chandan K. Reddy. ICLR 2025 (described as an oral in the repo README). arXiv:2404.18400. Code: github.com/deep-symbolic-mathematics/LLM-SR. — [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/28df8e730c054c5331855fd4d5403ba9-Abstract-Conference.html); [arXiv HTML](https://arxiv.org/html/2404.18400)
  - Loop: each iteration samples 4 skeletons (Python functions with free parameters) at temperature 0.8. Parameters are fitted with BFGS or Adam and scored by negative MSE. Programs go into an islands-based buffer (10 islands, Boltzmann selection, shorter programs favored), and 2 examples feed the next prompt. It runs about 2.5K iterations, against more than 2M for the GP baselines. — [arXiv HTML](https://arxiv.org/html/2404.18400)
  - Ablations (Oscillation 2, GPT-3.5, extracted):
    - full LLM-SR: NMSE 2.12e-7 in-domain / 3.81e-5 out-of-domain
    - "w/o iterative refinement": 1.01e-1 / 1.81e-1
    - without skeleton plus optimizer: 3.78e-1 / 3.75e-1
    - removing the natural-language prior or the program form also degraded results (values not extracted)
    
    The page reader's label for the "w/o iterative refinement" arm is ambiguous, so check it against the paper's figure. — [arXiv HTML](https://arxiv.org/html/2404.18400)
- LLM-SRBench (arXiv 2504.10415) is a follow-up benchmark for LLM equation discovery. — [arXiv](https://arxiv.org/pdf/2504.10415) (not fetched in detail)
- LaSR, "Symbolic Regression with a Learned Concept Library": Arya Grayeli, Atharva Sehgal, Omar Costilla-Reyes, Miles Cranmer, Swarat Chaudhuri. NeurIPS 2024 (an ICML 2024 AI4Science workshop version also exists). arXiv:2409.09359.
  - It uses zero-shot LLM queries to abstract concepts from high-performing hypotheses and to guide evolutionary steps.
  - It outperforms state-of-the-art SR on the Feynman equations and finds a new LLM scaling law. — [mlanthology NeurIPS 2024](https://mlanthology.org/neurips/2024/grayeli2024neurips-symbolic); [NeurIPS poster](https://nips.cc/virtual/2024/poster/96212)

### Inferences
- LLM-SR's "skeleton + numeric fitter" split is analogous to an agent writing program structure while thresholds or constants are fitted from the net. The large ablation gap suggests that letting the agent leave numeric constants to a fitter or the checker could matter a lot.
- The SR literature measures the loop by iteration-count curves (NMSE versus iteration) against non-LLM baselines run with far more iterations. A plot of "fraction of seeds solved versus rounds" would be the comparable figure for the repo.

### Gaps
- LaSR's ablation numbers (for example, with versus without the concept library) were not retrieved.
- Neither paper ablates the content of the feedback (scalar MSE versus residual or counterexample points).

## 5. Recommended ablations for a counterexample-guided decompilation study, drawn from this literature

### Takeaway
The literature supports a factorial design: feedback content × rounds × budget. It should include a budget-matched resampling baseline and a strength-of-feedback arm. The specific arms below are each grounded in a cited precedent, except "counterexamples with boundary points," which appears novel.

### Cited Findings
- No-feedback / one-shot arm: FIND's non-interactive milan baseline and Qiu's IO prompting baseline. — [FIND](https://arxiv.org/html/2309.03886); [Qiu et al.](https://arxiv.org/html/2310.08559)
- Pass/fail-only arm: Self-Debugging's "Simple" feedback (a correct-or-wrong signal), which gave smaller gains than unit-test feedback and could hurt without execution. — [Self-Debugging](https://arxiv.org/html/2304.05128v2)
- Counterexample arm (input, expected, got): Qiu et al.'s feedback template, Self-Debugging's UT feedback, and Greenblatt's "show what the program actually outputs." — [Qiu et al.](https://arxiv.org/html/2310.08559); [Self-Debugging](https://arxiv.org/html/2304.05128v2); [Redwood blog](https://blog.redwoodresearch.org/p/getting-50-sota-on-arc-agi-with-gpt)
- Self-generated versus external feedback arm: Qiu et al.'s SR (LM as interpreter) versus a symbolic interpreter; Olausson's self versus stronger-model versus human feedback. — [Qiu et al.](https://arxiv.org/html/2310.08559); [Olausson et al.](https://arxiv.org/html/2306.09896v5)
- Agent-chosen probes versus checker-supplied counterexamples: FIND's AIA versus AIA+exemplars, and MAIA's exemplars-only versus synthesis-only versus both. — [FIND](https://arxiv.org/html/2309.03886); [MAIA](https://arxiv.org/html/2404.14394)
- Rounds: report the curve, not just the final value. Qiu used T ∈ {1, 3}. Self-Debugging capped at 10 turns, with most gains in turn 1 and success within 3. InterpAgent saw most gains in iterations 1–2. — [Qiu et al.](https://arxiv.org/html/2310.08559); [Self-Debugging](https://arxiv.org/html/2304.05128v2); [InterpAgent](https://arxiv.org/html/2605.01555v1)
- Budget matching: compare against i.i.d. resampling at equal token cost (pass@t), vary initial samples versus repairs separately, and report bootstrapped variance. Revision often loses at low budget. — [Olausson et al.](https://arxiv.org/html/2306.09896v5)
- Search policy over candidates (which failing program to revise): REx's Thompson sampling versus greedy, measured in LLM calls to solve. — [Tang et al. NeurIPS 2024](https://neurips.cc/virtual/2024/poster/93642)
- Evaluation strictness: FIND shows that a lenient unit-test metric inflates scores relative to exact code matching (0.82 versus 0.23). Huang et al. show that confident explanations fail rigorous observational and causal tests. — [FIND](https://arxiv.org/html/2309.03886); [Huang et al.](https://preview.aclanthology.org/fix_video/2023.blackboxnlp-1.24/)

### Inferences
- A suggested minimal grid for the repo (inference, not taken from a single source):
  - Feedback content:
    - F0: none, one-shot
    - F1: pass/fail plus mismatch count
    - F2: k counterexamples (x, net(x), program(x))
    - F3: F2 plus boundary or near-boundary points
    - optionally F4: agent may query the net freely, without counterexamples (FIND-style)
  - Rounds: R ∈ {0, 1, 2, 3, 5, until-pass}.
  - Budget control:
    - an i.i.d. resampling baseline at matched total tokens or calls
    - "many initial drafts × 1 repair" versus "few drafts × many repairs" (Olausson's n_p/n_fr)
  - Report: solved fraction versus rounds and versus tokens, plus held-out agreement (to detect overfitting to counterexamples, as Qiu saw on MiniSCAN). Use a strict metric (exact or decision agreement on a dense grid or verified region) alongside any lenient one.
- Include a "counterexample count k" sweep. More counterexamples per round is the program-repair analogue of more feedback information, and none of the cited works sweeps it.

### Gaps
- I found no published ablation that compares pass/fail-only, counterexamples, and counterexamples plus boundary points for LLM program induction against a neural network. That combination appears to be an open gap the repo could fill.
- I found no work that measures held-out generalization of a decompiled program as a function of counterexample rounds. Qiu's MiniSCAN overfitting note is the closest.
