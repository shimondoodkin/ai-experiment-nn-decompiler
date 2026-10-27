# CEGIS and LLM self-debugging / self-repair with test or verifier feedback

Context for the report writer: the repo's loop (a blind LLM agent proposes a Python program, a checker compares it with a ReLU MLP on ~200k inputs and prints PASS/FAIL plus counterexample inputs, then the agent repairs the program) is an instance of CEGIS. The learner is an LLM and the verifier is a sampling-based (incomplete) equivalence tester. The notes below give the prior work on that loop.

Verification status: items under "Cited Findings" were fetched or searched in this session (arXiv abstract/HTML pages, search results). Items I know from prior knowledge but did not fetch are listed under "Gaps" and marked UNVERIFIED. Treat them as leads, not citations.

## Q1. CEGIS: origin, definition, relation to CEGAR, and CEGIS with a neural network as the specification

### Takeaway
CEGIS (a learner proposes a candidate, a verifier either accepts it or returns a counterexample, and the counterexample is added to the learner's example set) comes from Solar-Lezama's sketching work (ASPLOS 2006, PhD thesis 2008). It is the synthesis-side counterpart of CEGAR (Clarke et al., CAV 2000), which refines abstractions instead of candidate programs. The closest prior work with a neural net as the oracle or specification is in verifiable RL: a program or decision tree is extracted from a neural policy, sometimes inside a counterexample-guided loop (Zhu et al., PLDI 2019; VIPER, NeurIPS 2018). I found no paper that does CEGIS for exact functional equivalence between a synthesized program and a trained ReLU MLP, which is the repo's setting.

### Cited Findings
- Jha and Seshia, "Are There Good Mistakes? A Theoretical Analysis of CEGIS", SYNT 2014 (EPTCS 157, pp. 84-99; arXiv:1407.5397). They formalize CEGIS as a learner/verifier loop and prove that it is guaranteed to terminate with a correct program when the candidate space is finite. For infinite spaces it may not terminate. — [arXiv:1407.5397](https://arxiv.org/abs/1407.5397)
- Zhu, Xiong, Magill, Jagannathan, "An Inductive Synthesis Framework for Verifiable Reinforcement Learning", PLDI 2019. The paper synthesizes a simple deterministic program that approximates a trained neural policy and is black-box with respect to the network. It searches jointly for the program and an inductive invariant using a "counterexample and syntax-guided inductive synthesis procedure". The program runs as a shield next to the neural policy. This is the clearest published CEGIS loop with a neural net as the reference behavior. Note that its goal is safety-property preservation, not exact equivalence with the net. — [arXiv:1907.07273](https://arxiv.org/abs/1907.07273)
- Bastani, Pu, Solar-Lezama, "Verifiable Reinforcement Learning via Policy Extraction" (VIPER), NeurIPS 2018. VIPER distills a DNN policy into a decision tree using a Q-DAGGER variant, producing trees about 10x smaller than DAGGER's. The tree's piecewise-linear partitions are then verified for correctness, robustness and stability. Verification found a counterexample for a toy Pong controller, fixed by lengthening the paddle. This is distillation plus verification, not a CEGIS loop against the net. — [arXiv:1805.08328](https://arxiv.org/pdf/1805.08328); [NeurIPS 2018 paper](https://proceedings.nips.cc/paper_files/paper/2018/file/e6d8545daa42d5ced125a4bf747b3688-Paper.pdf)
- Michaud et al., "Opening the AI black box: program synthesis via mechanistic interpretability" (MIPS), 2024. MIPS distills trained RNNs into Python by converting the RNN into a finite state machine with an integer autoencoder and then running symbolic regression. It solves 32 of 62 algorithmic tasks, including 13 that GPT-4 does not solve. It is the closest "net to readable program" prior work, but it has no counterexample loop. — [arXiv:2402.05110](https://arxiv.org/pdf/2402.05110)
- A targeted search for "counterexample-guided" combined with "equivalence" and "neural-network-to-program distillation" returned no matching paper. — [search result summary pointing to MIPS](https://arxiv.org/abs/2402.05110v1)

### Inferences
- The repo's loop is CEGIS with an incomplete verifier: random or sampled testing over 200k inputs instead of SMT or MILP equivalence. The Jha-Seshia termination guarantee for finite candidate spaces assumes a sound verifier. With sampling, a PASS is evidence of equivalence, not proof.
- A ReLU MLP is piecewise linear, so an exact verifier is possible in principle: an MILP or SMT query "exists x such that program(x) != net(x)". Neural-network verifiers such as Marabou, alpha-beta-CROWN and Reluplex-style tools would do this. That would turn the loop into textbook CEGIS. This is my inference; I did not find a paper doing it for program-vs-MLP equivalence.

### Gaps
- UNVERIFIED (prior knowledge, not fetched): Solar-Lezama, Tancau, Bodik, Seshia, Saraswat, "Combinatorial Sketching for Finite Programs", ASPLOS 2006 (DOI 10.1145/1168857.1168907) is the paper usually credited with the CEGIS loop. Solar-Lezama's UC Berkeley PhD thesis "Program Synthesis by Sketching" (2008, tech report likely UCB/EECS-2008-177) gives the canonical exposition and the name CEGIS.
- UNVERIFIED: Clarke, Grumberg, Jha, Lu, Veith, "Counterexample-Guided Abstraction Refinement", CAV 2000; journal version in JACM 2003. CEGAR refines an abstraction of the system when an abstract counterexample turns out to be spurious. CEGIS refines a candidate program. Both share the "check, get counterexample, refine" skeleton.
- UNVERIFIED: Jha, Gulwani, Seshia, Tiwari, "Oracle-Guided Component-Based Program Synthesis", ICSE 2010 (OGIS: an I/O oracle plus distinguishing inputs). Jha and Seshia, "A Theory of Formal Synthesis via Inductive Learning", Acta Informatica 2017 (arXiv:1505.03953), the journal extension of the SYNT 2014 results. Abate, David, Kesseli, Kroening, Polgreen, "Counterexample Guided Inductive Synthesis Modulo Theories" (CEGIS(T)), CAV 2018. Alur et al., "Syntax-Guided Synthesis", FMCAD 2013.
- UNVERIFIED: in the reverse direction (neural candidate, formal verifier), Chang, Roohi, Gao, "Neural Lyapunov Control", NeurIPS 2019, uses a learner/falsifier loop that adds SMT counterexamples to training. Abate et al.'s "Fossil" tool does neural CEGIS for Lyapunov and barrier certificates. Verma et al., "Programmatically Interpretable Reinforcement Learning" (PIRL), ICML 2018, uses a neural policy as an oracle to guide program search. Dreossi et al., "Counterexample-Guided Data Augmentation", IJCAI 2018. None of these were fetched.
- Not found: any CEGIS paper whose specification is "be functionally equal to this trained MLP on its whole input box".

## Q2. LLM self-debugging / self-repair with execution, test or verifier feedback: key papers (titles, years, venues)

### Takeaway
The core LLM papers are Self-Debug (Chen et al.), Self-Refine (Madaan et al.), Reflexion (Shinn et al.), Olausson et al.'s "Is Self-Repair a Silver Bullet" (ICLR 2024), Lemur (ICLR 2024), LLM-in-CEGIS for SyGuS (Li, Parsert, Polgreen, CAV 2024) and LLM plus SMT CEGIS for planning (Jha et al., MILCOM 2023). The consistent result is that external, executable feedback (tests, verifier counterexamples) helps. Pure self-critique helps much less.

### Cited Findings
- Chen, Lin, Scharli, Zhou, "Teaching Large Language Models to Self-Debug", arXiv:2304.05128 (v1 April 2023, v2 October 2023). The abstract reports up to +12% on TransCoder and MBPP with unit tests and +2-3% on Spider (+9% on the hardest Spider level). It also says that "Self-Debugging can match or outperform baseline models that generate more than 10x" candidate programs. — [arXiv:2304.05128](https://arxiv.org/abs/2304.05128)
- Olausson, Inala, Wang, Gao, Solar-Lezama, "Is Self-Repair a Silver Bullet for Code Generation?", ICLR 2024. Tested on HumanEval and APPS with Code Llama, GPT-3.5 and GPT-4. They find "performance gains are often modest" once the cost of repair is counted, that gains vary by subset, and that self-repair is bottlenecked by the model's ability to give feedback on its own code. Code: github.com/theoxo/self-repair. — [arXiv:2306.09896](https://arxiv.org/abs/2306.09896); [HTML v5](https://arxiv.org/html/2306.09896v5)
- Shinn, Cassano, Berman, Gopinath, Narasimhan, Yao, "Reflexion: Language Agents with Verbal Reinforcement Learning", arXiv:2303.11366. Reports 91% pass@1 on HumanEval vs 80% for GPT-4 alone. — [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)
- Madaan et al. (16 authors), "Self-Refine: Iterative Refinement with Self-Feedback", arXiv:2303.17651. A single LLM acts as generator, critic and refiner, with no external feedback. Reports about 20% absolute average improvement over one-step generation across 7 tasks with GPT-3.5, ChatGPT and GPT-4. Code: selfrefine.info. — [arXiv:2303.17651](https://arxiv.org/abs/2303.17651)
- Wu, Barrett, Narodytska, "Lemur: Integrating Large Language Models in Automated Program Verification", ICLR 2024. The LLM proposes loop invariants or sub-goals, a verifier (ESBMC or UAutomizer) checks them, and failures trigger repair. Code: github.com/wu-haoze/Lemur-program-verification. — [arXiv:2310.04870](https://arxiv.org/abs/2310.04870); [HTML](https://arxiv.org/html/2310.04870)
- Li, Parsert, Polgreen, "Guiding Enumerative Program Synthesis with Large Language Models", CAV 2024. The LLM is placed inside a CEGIS loop for SyGuS. In "iLLM-synth" the LLM receives the most recent failed candidate plus the verifier's counterexample. — [arXiv:2403.03997](https://arxiv.org/abs/2403.03997); [HTML v2](https://arxiv.org/html/2403.03997v2)
- Jha, Jha, Lincoln, et al., "Neuro Symbolic Reasoning for Planning: Counterexample Guided Inductive Synthesis using Large Language Models and Satisfiability Solving", IEEE MILCOM 2023 (arXiv:2309.16436). The LLM is the learner and Z3 is the verifier, and SMT counterexamples are fed back through dialogue. Evaluated on blocks-world planning with GPT-4, GPT-3.5-turbo and older GPT-3 models. I did not retrieve its numbers. — [arXiv:2309.16436](https://arxiv.org/pdf/2309.16436); [SRI page](https://nusci.csl.sri.com/publication/milcom23b)
- Adnan and Kuhn, "The Debugging Decay Index: Rethinking Debugging Strategies for Code LLMs", arXiv:2506.18403 (June 2025). The search summary says it was later published in Scientific Reports (December 2025); I did not confirm the venue. — [arXiv:2506.18403](https://arxiv.org/html/2506.18403v2)
- "ConvCodeWorld: Benchmarking Conversational Code Generation in Reproducible Feedback Environments", ICLR 2025. Compilation feedback is always present. Execution feedback is none, partial-test or full-test. Verbal feedback (from GPT-4o) is none, novice or expert. Together these give 9 feedback combinations. — [arXiv:2502.19852](https://arxiv.org/pdf/2502.19852); [ICLR 2025 poster](https://iclr.cc/virtual/2025/poster/28169)

### Inferences
- The repo's loop sits closest to Self-Debug's "UT" setting (failing inputs with expected vs actual outputs) and to Lemur and iLLM-synth (a formal or semi-formal checker returns a counterexample). It is not like Self-Refine, which uses no external signal.

### Gaps
- UNVERIFIED venues (abstract pages fetched did not list them): Self-Debug at ICLR 2024; Reflexion at NeurIPS 2023 (code believed at github.com/noahshinn/reflexion); Self-Refine at NeurIPS 2023 (code believed at github.com/madaan/self-refine).
- UNVERIFIED (prior knowledge, not fetched): Huang et al., "Large Language Models Cannot Self-Correct Reasoning Yet", ICLR 2024 (arXiv:2310.01798). It finds that intrinsic self-correction without external feedback does not reliably improve reasoning and can degrade it. Zhong et al., "Debug like a Human" (LDB, runtime-execution-trace feedback), Findings of ACL 2024 (arXiv:2402.16906).
- UNVERIFIED (prior knowledge, not fetched): FunSearch, Romera-Paredes et al., "Mathematical discoveries from program search with large language models", Nature 2024 (DOI 10.1038/s41586-023-06924-6; code github.com/google-deepmind/funsearch). AlphaEvolve, Novikov et al., Google DeepMind 2025 white paper (arXiv:2506.13131). Both are evaluator-scored evolutionary loops: a scalar score and no counterexamples are fed back, which differs from CEGIS. I did not fetch their numbers.
- Not fetched: LLM loop-invariant synthesis papers other than Lemur, for example "Finding Inductive Loop Invariants using LLMs" (Kamath et al., 2023), "Ranking LLM-Generated Loop Invariants" (Chakraborty et al., EMNLP Findings 2023) and AutoSpec (CAV 2024). Titles and venues are unverified.

## Q3. What was measured: one-shot vs feedback, repair rounds, amount of counterexample content, diminishing returns

### Takeaway
1. Test or counterexample feedback gives most of the gain. A bare correct/incorrect signal already helps a lot on code tasks. Adding failing unit tests gives the next biggest jump. Natural-language explanations or traces add little on top.
2. Most successful repairs happen in round 1, nearly all within about 3 rounds. Effectiveness decays exponentially after that.
3. Under a fixed sample budget, self-repair beats plain resampling only modestly. Feedback quality (a stronger model, or a human) is the bottleneck.
4. A formal verifier in the loop gives large gains on hard instances where baseline tools fail.

### Cited Findings
**Self-Debug (Chen et al.): feedback-format ablation, accuracy %**
Source for this block: [ar5iv HTML of arXiv:2304.05128](https://ar5iv.labs.arxiv.org/html/2304.05128)
- MBPP, Codex: baseline 61.4, Simple (correct/incorrect only) 68.2, UT (unit-test feedback) 69.4, UT+Explanation 69.8, UT+Trace 70.8.
- MBPP, GPT-4: baseline 72.8, Simple 78.8, UT 80.6, UT+Expl 80.4, UT+Trace 80.2.
- MBPP, GPT-3.5: baseline 67.6, Simple 70.8, UT 72.2, UT+Expl 74.2, UT+Trace 72.8.
- TransCoder, GPT-4: baseline 77.3, Simple 80.9, UT 88.8, UT+Expl 90.4, UT+Trace 89.5.
- TransCoder, Codex: baseline 80.4, Simple 89.3, UT 91.6, UT+Expl 92.5, UT+Trace 87.9.
- Spider (no unit tests), Codex: baseline 81.3, Simple 81.3, +Expl 84.1. For GPT-4 the gains are at most 0.4 points.
- Rounds: the maximum was 10 turns, but "successful debugging processes mostly end within 3 turns". On TransCoder the "major improvement comes from the first debugging turn".
- Most fixed cases were output mismatches, i.e. wrong values rather than crashes: 61.9% on TransCoder (GPT-3.5) and 69.2% on MBPP (GPT-4).
- Without execution, a bare "Simple" self-check on MBPP with Codex fell below baseline (57.6 vs 61.4), while a self-generated trace reached 66.2.

**Olausson et al. (ICLR 2024): self-repair vs i.i.d. resampling at equal budget**
Source for this block: [HTML v5 of arXiv:2306.09896](https://arxiv.org/html/2306.09896v5)
- GPT-4 on APPS: 10 initial samples with 1 repair each gives 1.05x pass@20. 2 initial samples with 10 repairs each gives 0.97x pass@22, i.e. worse than resampling.
- Best-case relative gains: up to 8% for GPT-4 on APPS, up to 34% for GPT-3.5 on APPS competition-level problems, up to 10% for Code Llama on HumanEval, and up to 3% for GPT-3.5 on HumanEval (near ceiling).
- Budget split: more initial samples help consistently, while extra repair samples per feedback give "marginal gains at best". One repair per feedback string was the best setting.
- Feedback quality: swapping in a stronger model's feedback beat both resampling and self-repair at all budgets. With human feedback, GPT-4's repair success rose from 33.3% to 52.6% (1.58x); on competition-level problems it went from 3.67% to 14.67%.
- Expert raters judged GPT-4's own feedback inaccurate in 32/80 cases vs 7/80 for human feedback.
- Repair success rates on APPS: GPT-4 10.8%, GPT-3.5 4.7%, Code Llama 1.1%. On HumanEval: GPT-4 49.6%.

**Lemur (ICLR 2024): LLM + verifier vs verifier alone**
Source for this block: [HTML of arXiv:2310.04870](https://arxiv.org/html/2310.04870)
- Code2Inv, 133 programs: Lemur(GPT-4) solved 107 with an average of 4.7 proposals, Lemur(GPT-3.5) solved 103 with 8.6, the Code2Inv tool solved 92, and ESBMC alone solved 68.
- SV-COMP hard set, 47 programs, 15-minute timeout: Lemur(GPT-4) solved 25 with an average of 7.2 proposals, Lemur(GPT-3.5) solved 14, and UAutomizer and ESBMC solved 1 each. With 12 hours, UAutomizer solved 2 and ESBMC 4.
- 12 of the solved SV-COMP instances needed at least 5 proposals.
- Discrepancy: the paper's text says 26 instances were solved while its table says 25.

**Li, Parsert, Polgreen (CAV 2024): LLM in a CEGIS loop, SyGuS, 609 benchmarks, 600 s timeout**
Source for this block: [HTML v2 of arXiv:2403.03997](https://arxiv.org/html/2403.03997v2)
- LLM alone (GPT-3.5, up to 6 re-prompts with no counterexample): 303 solved (49.8%).
- Plain enumerator: 188.7. A* enumerator: 309. cvc5: 415 (68.1%).
- iLLM-synth, where the LLM receives the failed candidate plus the counterexample inside CEGIS: 408 (67.0%) with A*.
- Union of LLM and A*-pCFG-synth: 488 (80.1%), which beats cvc5.
- Diminishing returns: re-prompting was capped at 6 because new solutions fell below 2% per extra attempt. Solved benchmarks took about 4 attempts on average; that figure is partly garbled in the HTML.

**Debugging Decay Index (Adnan and Kuhn, 2025)**
Source for this block: [arXiv:2506.18403](https://arxiv.org/html/2506.18403v2)
- Debugging effectiveness per attempt fits an exponential decay. Across 18 models, most "lose 60-80% of their debugging capability within just 2-3 attempts".
- Strategic "fresh start" restarts, which clear the conversation history, raised Llama3.1 from 72.6% to 82.8% and DeepSeek-Coder-V2 from 84.1% to 92.1%. These are author-reported; I found no independent replication.

**Self-Debug sample efficiency**
- Self-Debug matches or beats baselines that sample more than 10x as many programs. — [arXiv:2304.05128](https://arxiv.org/abs/2304.05128)

**ConvCodeWorld (ICLR 2025)**
- Performance "varies significantly" with the feedback combination: no tests vs partial tests vs full tests, and novice vs expert verbal feedback.
- Models trained on one feedback combination generalize poorly to unseen combinations.
- Its static proxy, ConvCodeBench, correlates with the live benchmark at Spearman 0.82-0.99.
— [arXiv:2502.19852](https://arxiv.org/pdf/2502.19852)

### Inferences
- For the repo, failing inputs together with net-vs-program outputs is the high-value part of the feedback. This mirrors Self-Debug's UT condition. Adding prose explanations is unlikely to add much. It would be cheap to test this with an ablation: PASS/FAIL only, vs failing inputs, vs failing inputs plus which neuron or region flipped.
- Expect most repairs to land in rounds 1-3. If an agent is still failing after about 3 rounds, the DDI result and Olausson's budget result both suggest restarting from scratch or resampling rather than continuing the same conversation.
- The Iris fixes (moving a threshold, splitting one region into two half-planes joined by `or`) are "output mismatch" repairs, the category that dominated Self-Debug's fixed cases.

### Gaps
- I did not get exact per-round curves (Self-Debug Fig. 8 is shown only as a figure).
- I did not get ConvCodeWorld's per-condition numbers, Reflexion's ablations by number of trials, or the MILCOM 2023 paper's results.
- I found no study that varies counterexample count (1 vs k failing inputs) for LLM repair.

## Q4. Does counterexample quality matter (random vs boundary / minimal counterexamples)?

### Takeaway
In theory, for CEGIS, using minimal counterexamples does not change what can be synthesized. Restricting counterexamples to be "history-bounded" changes the class of solvable spaces, but neither variant dominates the other (Jha and Seshia 2014). For LLM repair, the empirical evidence is about feedback content and accuracy, not counterexample selection. I found no LLM study comparing random vs boundary vs minimal counterexamples.

### Cited Findings
- MinCEGIS (CEGIS with minimal counterexamples) has the same synthesis power as CEGIS. HCEGIS (counterexamples bounded by examples from earlier iterations) has different power from CEGIS, "none dominates the other". — [Jha and Seshia, SYNT 2014, arXiv:1407.5397](https://arxiv.org/abs/1407.5397)
- Feedback accuracy matters empirically. Inaccurate self-feedback (32/80 for GPT-4) vs accurate human feedback (7/80 inaccurate) corresponds to a 1.58x difference in repair success. — [Olausson et al., arXiv:2306.09896](https://arxiv.org/html/2306.09896v5)
- Adding a counterexample to the LLM prompt inside CEGIS gives 408 solved vs 303 for re-prompting without one. The configurations also differ in other ways (the enumerator is involved), so this is not a clean ablation. — [Li et al., arXiv:2403.03997](https://arxiv.org/html/2403.03997v2)

### Inferences
- The theoretical "power" results concern which spaces are learnable at all, not how fast. Minimal or boundary counterexamples could still reduce iteration counts in practice, especially for an LLM learner that reasons about where a threshold sits. A counterexample close to a decision boundary pins down the threshold, while a random one far from it does not. This is my inference, not a cited result.
- For a ReLU MLP, natural "boundary" counterexamples are points near the net's or the program's decision boundary, or near ReLU activation-pattern changes. A checker could report the counterexample closest to the program's boundary.

### Gaps
- UNVERIFIED: the journal version (Jha and Seshia, Acta Informatica 2017, arXiv:1505.03953) reportedly extends this theory, including to infinite domains. Not fetched.
- Not found: any empirical study of counterexample selection strategy (random vs minimal vs boundary vs diverse sets) for LLM-based repair or for SyGuS solvers.

## Q5. Runnable open-source code

### Takeaway
Repositories exist for Olausson et al.'s self-repair, Lemur, Self-Refine and ConvCodeWorld (Hugging Face space). Reflexion, FunSearch and Sketch code are widely known but were not verified in this session.

### Cited Findings
- Self-repair (Olausson et al.): github.com/theoxo/self-repair — [arXiv HTML](https://arxiv.org/html/2306.09896v5)
- Lemur: github.com/wu-haoze/Lemur-program-verification — [arXiv HTML](https://arxiv.org/html/2310.04870)
- Self-Refine: "Code, data, and demo" at selfrefine.info — [arXiv:2303.17651](https://arxiv.org/abs/2303.17651)
- ConvCodeWorld: project page and Hugging Face space — [project page](https://convcodeworld-convcodeworld.static.hf.space/)
- Li, Parsert, Polgreen (CAV 2024): the fetched HTML gave no code link. — [HTML v2](https://arxiv.org/html/2403.03997v2)

### Inferences
- The cheapest reusable piece for the repo is likely Lemur's propose/verify/repair driver structure, or the self-repair harness for budget accounting. The repo's own checker already plays the verifier role.

### Gaps
- UNVERIFIED repos: Reflexion (github.com/noahshinn/reflexion), FunSearch (github.com/google-deepmind/funsearch, which ships the evaluator and skeleton but not the full distributed system), the Sketch synthesizer (Solar-Lezama, MIT), CVC5's SyGuS mode, Self-Debug (no official repo that I know of), LDB (github.com/FloridSleeves/LLMDebugger) and the Zhu et al. PLDI 2019 artifact. None of these were checked in this session.
