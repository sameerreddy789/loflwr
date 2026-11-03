# Can a public dataset provide a fresh banking accuracy test?

Review date: **3 November 2026**. Decision: use CLINC150 for a **narrow external
transfer and unsupported-request test**. None of the candidates reviewed supplies
a verified, directly compatible independent test of all twenty current intents.
Changing the benchmark cannot turn the existing 94% result into 100% accuracy.

## Candidate assessment

| Dataset | Verified collection and task | Fit for this project |
| --- | --- | --- |
| [CLINC150](https://github.com/clinc/oos-eval) | Crowdsourced English single-intent queries; 150 intents across 10 domains; 30 official test queries per intent | `card_declined` is a reasonable proxy for `declined_card_payment`. Other broad banking labels cannot safely become our fine-grained labels. Selected for one-class transfer and two OOS stress groups. |
| [Bitext retail banking](https://huggingface.co/datasets/bitext/Bitext-retail-banking-llm-chatbot-training-dataset) | Publisher explicitly describes hybrid synthetic generation; 25,545 question-answer pairs, 26 intents | `activate_card` and `cancel_transfer` have clean metadata-level mappings. Could support a separately labeled synthetic robustness/development study; not independent human confirmation. Not scored here. |
| [NLU++](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/nlupp) | 2,071 banking examples with 48 intent modules and 13 slots; compositional multi-label task | Useful future banking benchmark with a different output contract. Modules merge actions such as make/open/apply/setup/get/activate and objects such as transfer/payment/deposit; they cannot reliably yield our twenty labels without separate annotation. Not scored here. |
| [NPCI TauIndianBankBench](https://huggingface.co/datasets/NPCI/tau-indian-banking) | 1,000 synthetic tasks across 50 scenario families, with a synthetic customer database; tool-using conversational agents | Evaluates required actions and final database state. This classifier has neither tools nor a bank database; agent success is not intent accuracy. Not scored here. |
| [HWU64 mirror](https://huggingface.co/datasets/DeepPavlov/hwu64) | 64 assistant intents, 8,954 training and 1,076 test examples; mirror of an earlier dataset | Candidate for unrelated-topic stress only. Original licensing/lineage review incomplete; no need to add it when CLINC provides a verified source. Not downloaded/scored. |
| [MASSIVE](https://github.com/alexa/massive) | Official repository existence and purpose verified | Not adopted. Full ontology, license and collection-lineage review were not completed in this pass; no independent-banking claim is made. |

The NLU++ paper's section 3.4 says four expert annotators started from fully
anonymized real-user examples, reannotated them and created additional examples
for less frequent intents/slots. Its ethics section identifies PolyAI employees
as collectors and annotators. It is not described as a BANKING77 test split, but
this review does **not** establish absence of underlying example overlap. More
fundamentally, its multi-label ontology does not provide the required target
labels. Consult the [paper](https://arxiv.org/abs/2204.13021) and
[full text](https://arxiv.org/html/2204.13021v3), not the repository README's
unfinished citation block. The paper identifies Findings of NAACL 2022.

## Implemented evaluation

The [protocol](external-clinc-protocol.md) declares source labels, deduplication,
metrics and unchanged inference before scoring. Its [pre-inference lock](results/external-clinc-v1/plan.json)
was committed and pushed in `23dba1f`. It fixes the original Flower head500,
MPNet assets, twenty-way argmax and all three original margin thresholds.

Preparation selected 270 official CLINC test rows. One declined-card row exactly
matched an original BANKING77 query and was excluded by the declared rule. The
retained set contains **29 mapped-intent queries and 240 unsupported queries**.
No retained row exactly matches CLINC train/validation. Four mapped queries have
token-set Jaccard similarity >= 0.8 to original BANKING77 examples; these remain
included and are disclosed. This is not complete semantic decontamination, and
MPNet pretraining exposure has not been ruled out.

The [measured result report](results/external-clinc-v1/README.md) keeps these
groups separate. The model still selects among all twenty intents; it is never
given a one-label answer set. OOS false accepts measure unsupported requests
that pass the existing margin rule. Declining a request is not proof of correct
OOS recognition or of successful subsequent clarification.

No source reviewed here provides a justified 20-intent replacement for the
locked 300-query BANKING77 confirmation. The all-intent score remains **94%
(282/300)**. A perfect result on one class would not revise that figure, and
the small sample would not establish 99% population accuracy.

## Next development decision

Preserve this external evaluation as evidence, not a source of revised labels,
favorable exclusions or retuned thresholds. For a general support demo, collect
separate in-scope and unsupported **development** examples and compare an explicit
scope detector with the current margin-only rule. Freeze the resulting system
before a new independent evaluation. Report false accepts, in-scope coverage,
wrong answers and clarification outcomes separately.

For all-twenty-intent accuracy, independently annotated queries with the exact
ontology remain the most direct route; see the [collection guide](../docs/fresh-evaluation.md).
Public data can support development, but synthetic examples, reprocessed
BANKING77 mirrors and a changed label set must not be presented as independent
confirmation of the unchanged task.

## Source register and review depth

All sources below were accessed on 3 November 2026. Research used Agent Reach's
GitHub CLI and Jina Reader routes plus primary Hugging Face metadata. The
`agent-reach`/`mcporter` executables were unavailable; no Exa search, social
survey, backend doctor or update check is claimed. Published dataset descriptions
are distinguished from the project's actual measurements.

| Source / revision | Status and license | Material actually reviewed |
| --- | --- | --- |
| CLINC `828f8093932c8fe6ca7936c3d2e52903b1c523de`; [paper](https://aclanthology.org/D19-1131/) | EMNLP-IJCNLP 2019 paper; **CC BY 3.0 Unported**, actual LICENSE read | README, full domain ontology, paper abstract/metadata, source bytes and selection/deduplication audit. Git blob and SHA-256 values are in the evaluation lock. Original authors: Stefan Larson et al. |
| Bitext `3e3621092fc6baaf7f53ceb6f091c60ae99acb67` | Dataset card; **CDLA-Sharing 1.0** declared | README, generation description and 26-intent list. Query files not evaluated; no publisher performance claim adopted. |
| NLU++ `57ec275d8078af65b7731c2a98be812d844a6d6b`; [paper](https://arxiv.org/abs/2204.13021) | Paper identifies Findings of NAACL 2022; **CC BY 4.0**, paper ethics section and repository license | README, complete intent/slot ontology, paper abstract, collection section and ethics statement. Fold files not imported or scored. No benchmark reproduction. |
| NPCI `23e77c23ed777ea007916aaf2ca2a9bafbb73e78`; [benchmark code](https://github.com/npci/TauIndianBankBench) | **MIT** declared on dataset card; card links [2026 preprint](https://arxiv.org/abs/2610.30161) | Dataset README and task description only; the linked paper/code and evaluation implementation were not separately audited. |
| HWU64 mirror `0dd289ccdeb185ec065d1ebcf5de1c443cd1620f` | No license declared in inspected mirror README; original terms unresolved | Metadata, counts and README lineage only. No use of raw examples. |
| MASSIVE official repository | License/publication not reviewed in this pass | Repository metadata only. Not accepted as a suitable evaluation source. |

CLINC source data stays under ignored `assets/external-clinc-v1/source/` with
the original LICENSE and README. Public results contain only counts, IDs,
hashes and metrics; no raw questions or model weights are redistributed.
