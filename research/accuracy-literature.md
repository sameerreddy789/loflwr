# Why accuracy can be low, and how to improve it

Research review: **3 November 2026**. This is an explanation and experiment plan
for the solo builder, not a claim that the proposed fixes have already improved
our model. The active specification remains SmolLM2-135M-Instruct, twenty
BANKING77 intents, shared frozen A, B-only updates, and the matched comparisons
in [evaluation.md](evaluation.md).

The strongest initial hypotheses are insufficient **unique minority examples**,
extreme label segregation across clients, dataset-size weighting that favours
the already data-rich domain, and a generative scoring rule that can prefer
particular label strings. A small general-purpose decoder and a narrow adaptation
configuration can add limitations. Papers justify investigating these mechanisms;
they do not establish which one causes a particular error in our running study.

## What “100%” would mean

The benchmark asks for the correct intent on previously held-out queries.
Training accuracy, a rehearsed query list, full validation accuracy, and final
test accuracy are different quantities. A 100% score on a finite test is possible
in principle, but no reviewed source guarantees it for this setup or for future
queries. The original BANKING77 paper explicitly describes partially overlapping,
fine-grained categories [1].

Keep 100% as an aspiration. Do not relabel ambiguous test examples, train on test
errors, remove inconvenient classes, or select only easy queries to manufacture
it. A clarification/abstention feature can make a banking demonstration more
useful, but must report coverage and accuracy among answered queries separately;
it does not improve the original twenty-way forced-choice accuracy by itself.

## The most important design facts

| Fact in our experiment | Likely consequence | What would discriminate the cause |
| --- | --- | --- |
| The imbalanced manifest gives client 1 ninety examples per intent and other clients only two or three per intent | Repeated training cannot introduce paraphrases absent from those tiny minority samples | Compare the already specified balanced manifest; inspect validation confusions and sample coverage without changing the test |
| Each client owns just four of the twenty intents | A local model can improve its own domain while making decisions between domains worse | Evaluate domain scores before/after local updates on a diagnostic copy; compare matched centralized B-only and federated training |
| Sample-weighted FedAvg uses coefficients 0.9 and four times 0.025 when all clients survive | Minority clients have little direct coefficient weight despite owning 80% of the evaluated intents | Compare sample weights with equal weights using identical input states, work and data |
| The fastest client also owns the most data | Dataset size and local work both favour the same domain; their effects are confounded across scenarios | Inspect processed work and delta norms; use within-condition strategy comparisons |
| Our scorer sums completion-token log probabilities, including the fixed EOS convention | Different label lengths and language-model priors can alter rankings independently of the intended class concept | Record all twenty validation scores, token lengths and prediction frequencies; compare prespecified scoring variants on validation only |
| A 135M general instruction model was chosen to fit weak devices | Good efficiency does not imply a strong pre-existing banking decision boundary | Compare frozen model, centralized B-only, and a small encoder reference on the same manifests |
| Only rank-8 B factors on q/v projections are trained | Adaptation is restricted; common A solves an aggregation issue but is not an accuracy guarantee | First verify training fit and gradients; test more capacity only after simpler diagnoses |
| Median clipping limits large updates | It can suppress useful work as well as excessive influence | Compare equal averaging with and without clipping; inspect factors, norms and domain effects |

These are hypotheses, except for arithmetic/configuration facts. In particular,
do not conclude that a low current validation score proves SmolLM2 is unusable,
that frozen A is defective, or that clipping must help.

### Repetition is not the same as more data

With twenty healthy attempts, accumulation four and the planned heterogeneous
optimizer-step budgets, nominal example visits are:

| Domain/client | Unique training queries in imbalance | Nominal visits over twenty attempts | Visits divided by unique count |
| --- | ---: | ---: | ---: |
| Card lifecycle / 0 | 10 | 160 | 16.0 |
| Card payments / 1 | 360 | 1,280 | 3.56 |
| Transfers / 2 | 10 | 640 | 64.0 |
| Cash withdrawals / 3 | 10 | 320 | 32.0 |
| Account verification / 4 | 10 | 160 | 16.0 |

This is arithmetic, not measured coverage. Actual sampling, incomplete jobs and
dropout change work, and the ratio does not guarantee every record was seen that
often. It nevertheless explains why “just train longer” may overfit the minority
queries instead of solving generalization. The balanced manifest gives every
intent forty distinct queries and is essential context for the extreme stress
test.

### Why shared A does not solve every federated problem

For shared A and scaling, averaging B preserves the average effective adapter
update. That removes the mismatch between separately averaging independently
trained A and B factors. FFA-LoRA addresses this mechanism [3]. The remaining
network is nonlinear, local objectives differ, clients take different numbers of
steps, and aggregation can still combine competing directions.

For one adapted layer, the rows of `delta_B @ A` remain in the row span of the
fixed A. That is a statement about this parameterization, not proof that rank
eight is inadequate for our task. Increasing rank, changing A initialization, or
training additional projections creates a new adapter identity and a new
experiment. It must not be spliced into the frozen comparison.

FedNova motivates examining unequal local work and objective mismatch [6]. Its
theory is not automatically a theorem for our AdamW, B-only, median-clipped
implementation. Likewise, a FedProx penalty can reduce local drift but does not
change a 90% server coefficient into a fairer one.

## What the literature actually supports

| Source/method | Relevant finding | Transfer limit for this project |
| --- | --- | --- |
| Original BANKING77 / dual sentence encoders [1] | Studies 77 fine-grained intents, including balanced ten- and thirty-example-per-intent conditions. Fixed conversational embeddings with a small classifier outperform its BERT-based alternatives; the authors report CPU training in minutes | Our twenty-intent subset, two-to-three-example minority condition, manifests and federated budgets differ. Do not copy their scores into our results or promise the same CPU time |
| SetFit [2] | Contrastive fine-tuning on labelled sentence pairs followed by a classifier avoids prompt/verbalizer dependence; supplies an efficient few-shot classification alternative | The reviewed main paper is not a verified score for our BANKING77 subset. Pair construction reuses source examples; it does not create independent linguistic evidence. A centralized encoder reference is not the promised federated B-only system |
| FFA-LoRA [3] | Freezes the nonzero-initialized factor and trains the zero-initialized factor to improve federated factor aggregation; studies heterogeneity, hyperparameters and DP settings | Reviewed experiments use RoBERTa-Large/GLUE and LLaMA/GSM8K on A100 hardware, not this small SmolLM model, banking task or CPU budget. Our system does not implement DP |
| LoRA-FA [4] | Freezing the down-projection can reduce activation memory; the current reviewed version also describes gradient corrections toward full-gradient updates | Plain frozen-A AdamW is not automatically the complete corrected LoRA-FA optimizer. Do not infer that swapping optimizers guarantees better accuracy |
| Contextual calibration [5] | LM answer preferences, prompt choices and token priors can materially affect few-shot classification; content-free inputs estimate bias | Original experiments concern GPT-2/GPT-3 prompting, not our fine-tuned adapter. Calibration is a testable candidate, not an established correction for every error |
| Surface-form competition [7] | A label concept and one particular string probability are different; compares raw, length-normalized and domain-conditional PMI scores | The authors also discuss settings where alternatives do better. Scoring changes are model-selection choices requiring validation and a declared final rule |
| SmolLM2 model card [8] | Confirms compact on-device intent, general instruction training and English focus, while documenting limitations | Publisher general benchmarks do not estimate banking accuracy. Model size alone neither establishes nor rules out task adequacy |
| all-MiniLM-L6-v2 model card [9] | Provides 384-dimensional sentence embeddings and documents correct pooling for a Transformers implementation | It is a candidate frozen-embedding reference, not an already measured FedLoRA-Edge result or a federated implementation |

ConveRT is historical evidence, not the recommended new dependency: its original
repository says the models were withdrawn from public distribution [10]. A
maintained sentence-transformer reference is more practical, after pinning and
checking its model license and local runtime.

No numeric external benchmark is treated as directly comparable to our custom
subset. The existing local TF-IDF/logistic-regression reference is more directly
informative: it recorded **41.5% validation accuracy**, macro F1 **0.36003** and
worst-domain accuracy **20%** on the same 400-example imbalanced scenario.
Its run is `tfidf-imbalanced-validation`; configuration and manifest hashes are
preserved in [the exported evidence](results/tfidf-validation.json). This is
centralized, class-weighted task context, not evidence about federated LoRA. It
also shows that minority-domain errors are not unique to generative decoding.

A newly measured reference, `tfidf-balanced-validation-20261103`, keeps the
classifier hyperparameters and the same 200-query validation set but uses the
balanced 800-example training manifest. It achieves **87% validation accuracy**,
macro F1 **0.86829**, and worst-domain accuracy **77.5%**; see
[its exported evidence](results/tfidf-balanced-validation.json). This is a
45.5-percentage-point accuracy difference within this classifier comparison.
The change simultaneously alters total training quantity and per-intent
coverage, so it does not isolate imbalance alone. It strengthens the case for
testing coverage before replacing the model, but does not establish the effect
on FFA-LoRA or on the final test.

## A bounded improvement sequence

### 1. Establish correctness before hyperparameter search

Check the complete input-to-score path with real tokenizer/model outputs:
completion boundaries, PAD/EOS aliasing, causal shift, label mappings, truncation,
and agreement between a simple scorer and its optimized version. Verify that B
has nonzero finite gradients and changes, base/A stay unchanged, the optimizer
owns only B, and every attempt starts from the correct incoming state. Read
actual work and rejection logs rather than assuming every configured step ran.

Use a small, balanced **training-only** sanity set to test whether the model can
fit all twenty labels under a declared diagnostic budget. Successful training
fit is not held-out accuracy. Failure to fit is a reason to investigate the
objective, updates or capacity before blaming generalization. Existing smoke
tests establish that optimization occurs, not that this learning task is solved.

### 2. Complete the frozen comparison and inspect validation errors

Keep the current study's configurations and identities intact. For each recorded
checkpoint, inspect prediction counts, per-intent confusion, worst-domain score,
loss, and pairwise top-score margins. Separate within-domain mistakes from
predicting the wrong domain. A model predicting mostly the rich domain suggests
a different next step from one distinguishing domains but confusing close
intents inside each domain.

The matched centralized B-only reference separates some federation effects from
the limitations of the model/training recipe. The plain-equal control separates
the weighting change from clipping. Neither reference is an automatic upper
bound. Compare the healthy balanced scenario before treating the deliberately
extreme imbalanced scenario as the model's general capability.

### 3. Run cheap validation-only scoring diagnostics

On a copied checkpoint, record the twenty score vectors and evaluate a small,
declared set of candidate rules:

1. The frozen sum-log-probability rule.
2. Mean completion-token log probability, with an explicit EOS convention.
3. A domain-prior correction, such as subtracting a fixed content-free banking
   prompt's label log probability, inspired by [5,7].

Keep raw scores and identify which errors change. Inspect label token lengths
and a content-free prediction histogram. A scalar temperature applied uniformly
to all twenty scores cannot change their argmax, so temperature scaling alone
cannot improve forced-choice accuracy. Per-label corrections can change it, but
may overfit a small validation set. Do not test hundreds of prompts or offsets
against the same 200 validation examples and describe the winner as unbiased.

If a candidate is adopted, name the new scoring configuration and apply it to
all relevant baselines under a new comparison identity. Preserve the originally
specified results. A new final-test analysis must disclose any earlier test
access; repeated selection on that test destroys its holdout role.

### 4. Improve data coverage or optimization based on the diagnosis

If tiny-set fit is good but validation generalization is poor, unique examples
and task representation are stronger candidates than extra repeats. Keep the
400-example imbalance stress test fixed; run the already specified balanced
800-example scenario separately. Any further labelled training expansion must
be a new manifest using eligible training data only, with its own hashes.

If both training and validation are still improving, a small, prespecified
learning-rate/work-budget ablation can be justified. If training improves while
validation deteriorates, test a declared early-stopping rule or less local work
in a new experiment. Hyperparameters are selected on validation, never final
test. Do not simultaneously change rank, optimizer, scoring and sample counts
and attribute the result to one of them.

### 5. Add an efficient reference before paying for a larger decoder

First test frozen sentence embeddings plus a small balanced classifier on the
same public manifests. If that is useful, consider SetFit as an explicitly
separate task-efficiency comparison. This provides a strong answer to whether
the chosen banking classification problem needs a generative model at all.
It does not silently replace the federated method or satisfy adapter-only
communication claims.

Only after diagnosing a capacity limitation should a rank/module ablation or
SmolLM2-360M alternative become a priority. Recheck RAM, inference latency,
training throughput, payload identity and offline deployment. A larger decoder
may help but is not a justified first response to two labelled examples per
minority intent.

## Evidence and review register

All links below were accessed on **3 November 2026** through Agent Reach's Jina
Reader route, with GitHub CLI metadata used for the SetFit repository. The
`agent-reach` and `mcporter` executables were unavailable; no Exa search, backend
doctor or update check ran. This was targeted primary-source retrieval, not a
systematic literature review. Reader cache timestamps are not publication dates.

| ID | Primary source | Status and actual review depth |
| --- | --- | --- |
| 1 | [Casanueva et al., Efficient Intent Detection with Dual Sentence Encoders](https://aclanthology.org/2020.nlp4convai-1.5/); [full text](https://arxiv.org/html/2003.04807v1) | NLP4ConvAI 2020 workshop publication verified in ACL Anthology. Dataset/method/experimental-setup/discussion text read. Numerical tables did not extract fully; no table score quoted |
| 2 | [Tunstall et al., Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055); [full text v1](https://arxiv.org/html/2209.11055v1); [SetFit repository](https://github.com/huggingface/setfit) | 2022 arXiv v1 and repository citation verified; no venue independently verified. Method, experimental setup, model-efficiency sections read; no BANKING77 result inferred from this paper |
| 3 | [Sun et al., Improving LoRA in Privacy-preserving Federated Learning](https://arxiv.org/abs/2403.12313); [full text](https://arxiv.org/html/2403.12313v1) | arXiv metadata states ICLR 2024. Method, experiments, rank and initialization discussions reviewed. No reproduction or independent proof audit |
| 4 | [LoRA-FA](https://arxiv.org/abs/2308.03303) | Abstract and version history read; v3 is dated 9 May 2026. The current abstract describes gradient corrections. Exact optimizer code, compatibility and venue not audited |
| 5 | [Zhao et al., Calibrate Before Use](https://arxiv.org/abs/2102.09690) | Abstract and metadata read; ICML 2021 stated. Full HTML extraction was incomplete. No reported gain transferred to our fine-tuned task |
| 6 | [Wang et al., Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization](https://arxiv.org/abs/2007.07481) | Abstract and metadata read; theoretical claims not rederived or verified for our optimizer |
| 7 | [Holtzman et al., Surface Form Competition](https://arxiv.org/abs/2104.08315); [full text v9](https://arxiv.org/html/2104.08315v9) | Abstract/version history and scoring/analysis sections read. General LM scoring evidence; no local reproduction |
| 8 | [SmolLM2-135M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | Model description, training, benchmarks, limitations and Apache 2.0 section read. Publisher observations, not project metrics |
| 9 | [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Embedding dimension and Transformers/pooling usage read. Exact revision/license adoption and local benchmark remain separate checks |
| 10 | [Original ConveRT model repository](https://github.com/PolyAI-LDN/polyai-models) | Current README withdrawal statement read; no model downloaded |

The arithmetic about data, work and weights is derived from our declared
configuration. The cited TF-IDF values come from the named local evidence file.
All other proposed improvements remain experiments, and none establishes a
100% accuracy guarantee.
