# Retrieval experiments

Run commands from the repository root. All experiments use only the supplied
training TSVs. The previous submission and its writer are unchanged.

## Environment

This checkout has Python 3.11.9 at the path below and usable packages in
`venv/Lib/site-packages`, but its virtual-environment Python launcher is missing.
For this machine, use:

```powershell
$env:PYTHONPATH=(Resolve-Path 'venv\Lib\site-packages').Path
$pythonExe='C:\Users\ashis\AppData\Local\Programs\Python\Python311\python.exe'
& $pythonExe -m unittest discover -s business_entity_resolution/tests -v
```

In a working environment, ordinary `python -m ...` commands work instead.
The experiment code adds no dependencies; it uses the existing pandas 3.0.6
environment for the repository's evaluation module.

## Reproduce the pilot

```powershell
& $pythonExe -m business_entity_resolution.src.retrieval_pilot --size 2000 --restore-split
& $pythonExe -m business_entity_resolution.src.retrieval_experiment --rerank-baseline --output reports/retrieval_dedup_rerank
& $pythonExe -m business_entity_resolution.src.retrieval_experiment --output reports/retrieval_v3_pilot --workers 6
& $pythonExe -m business_entity_resolution.src.retrieval_ablation
```

`--restore-split` only creates a missing split. It uses the original seed-2026
algorithm: 220,684 validation S1 entities, including the 50,000-entity dev sample.
An existing split is never regenerated. The 2,000 pilot entities are selected
proportionally by country, match-count bucket, name script, and a common-name
proxy (core-name frequency within dev). Tiny strata have a minimum allocation
when possible. Therefore pilot estimates are exploratory; they are not an
untouched confirmation score. All true S2/S3 siblings of selected S1 entities
are diagnosed. Singleton entities remain in oracle macro F0.5.

The original frozen split was absent from this checkout and the supplied ZIP;
the restored split is reproducible but has not been hash-compared with the
historical run's missing artifact.
Its 50,000 selected dev entities contain 173,070 true links, matching the
historical report's aggregate count.

The pilot index scans **every training S2/S3 row**. Only keys emitted by pilot
queries are retained, and an over-cap posting is discarded as soon as its
document frequency exceeds its cap. Counts continue after discarding. This is
an exact projection of a full index for these queries, not a sample of candidate
records. Truth records are retained separately for diagnosis; truth does not
choose index keys, insert candidates, or contribute to retrieval/pair scores.

The same scan counts both occurrences (legacy) and documents (corrected).
Legacy scoring preserves repeated posting and query contributions for a faithful
comparison. Corrected scoring counts a document and query key once. Count and
posting construction are deliberately fused into one streaming phase; query,
cache, and setup phases are reported separately. Windows memory measurements
are process RSS/working set and lifetime peak RSS, not tracemalloc allocations.
The split-building peak can exceed the subsequent index's working set.
Enhanced indexing supports a bounded local process queue (`--workers`, default
4). Only key extraction runs in workers; ordered frequency/posting updates run
in the parent. Tests compare sequential and parallel results exactly. Reported
parent RSS and separately sampled worker peak sums are distinguished; the sum
of individual peaks is not a simultaneous process-tree peak.

Outputs under `reports/retrieval_pilot`:

- `pilot_ids.tsv`: fixed entity membership, country and match count.
- `losses_legacy.tsv`, `losses_dedup.tsv`: one row per true link; raw/usable key
  families, mutually exclusive reason at K=100, exact uncapped rank, score,
  script mismatch, missing address, and sibling count.
- `summary.json`: reason counts, recall and oracle at 20/50/100/300/1000 and
  the complete usable pool, country/source breakdowns, coverage, key frequency
  distributions, timings and memory.
- `projected_index.sqlite3`: reusable bounded index. Its signature includes
  input file stamps, selected IDs, keys, relevant normalization code, and byte
  order. A mismatched cache is rejected; select a new output directory.

Reports, raw ID manifests, local caches and candidate records are ignored by
Git. Keep concise aggregate results in this documentation; do not commit raw
rows, full datasets, or caches.

## Additional retrieval and reranking

`retrieve_v3.py` is an experimental retriever. Original keys remain available;
additional views include accent-folded core names and address words, capped
address word-pairs, address word/long-number combinations, unordered number
pairs, and capped four-character consonant-skeleton fragments. Keys remain
country-partitioned. There is no country one-hot feature or France-specific rule.
No external business data or transliteration package is used.

The reranking experiment reserves half of its pool for global IDF and divides
the remaining capacity between name, name-shape, and address routes, with
deduplication and deterministic filling. These initial budgets and weights are
experimental, not tuned defaults. It reports the pool's measured oracle and
compares IDF order, name-only similarity, and name-plus-address similarity.
Only pool candidate records are materialized through a streaming join into a
local SQLite cache. Features include folded core-name equality, token/character
and skeleton overlap, address word similarity, normalized number agreements and
conflicts, and missing address. IDs and true match counts are not features.

`--pool-size 1000` is the initial experiment setting, not a claim that its oracle
is sufficient. Use another output directory for a different size because pool
record caches are signature-checked. The K=1000 and full-pool measurements make
this limitation explicit. Reranking cannot recover a record outside its pool.

## Baseline pilot measurements

2,000 S1 entities (1,200 US / 800 India), 6,924 true links, 111 singletons.
The selected queries all have Latin names; 549 India true links cross name
scripts. Script mismatch in the loss TSV is measured on both sides of each
link. The rerank report's `native_name` grouping describes the S1 query only;
it is not a cross-script-pair accuracy estimate.

| Variant | Recall@20 | Oracle@20 | Recall@100 | Oracle@100 | Full-pool oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original v2 occurrence counting | 0.68905 | 0.82603 | 0.78683 | 0.89286 | 0.97304 |
| Corrected v2 document counting | 0.70494 | 0.83537 | 0.78784 | 0.89296 | 0.97312 |

Corrected loss reasons at K=100: 5,455 in top K; 1,054 below K; 404 with every
shared key pruned; 11 with no shared raw key. India full-pool oracle is 0.95406;
US is 0.98583. Deduplication is a correctness repair and improves K=20 on this
pilot, but is not by itself a large recall improvement. It slightly reduces
K=1000 recall (0.89168 to 0.89081), so the effect is not uniformly positive.

First execution: 390.56 seconds scanning/counting/bounded postings,
8.61 seconds legacy querying, 9.14 seconds corrected querying. Indexing working
set was about 191 MB; process lifetime peak was 698 MB during split creation.
These figures describe a 2,000-query projected index, not full-test performance.

Baseline reranking uses the same 2,000 entities and a route-reserved pool of up
to 1,000 candidates. Pool oracle is 0.95905, compared with 0.95051 for the plain
IDF top 1,000. Its materialization and reranking pass took 319.18 seconds with
268 MB process peak RSS.

| Baseline-pool ranking | Recall@20 | Oracle@20 | Recall@100 | Oracle@100 | Oracle@300 |
| --- | ---: | ---: | ---: | ---: | ---: |
| IDF | 0.70494 | 0.83537 | 0.78784 | 0.89296 | 0.93148 |
| Name only | 0.64240 | 0.76183 | 0.76459 | 0.86921 | 0.92389 |
| Name + address | 0.84532 | 0.92639 | 0.88879 | 0.95127 | 0.95602 |

Address-aware reranking improves oracle@100 by 0.05831 over corrected v2 on this
pilot. Name-only reranking regresses. The 0.95905 pool ceiling still prevents
0.98; an expanded retrieval experiment is necessary before model training.
Address-aware oracle@100 is 0.96886 for US and 0.92488 for India, so the gain
is present in both countries.

## Enhanced pilot results

| Additional routes (each compared with corrected baseline) | Oracle@100 | Oracle@300 | Full-pool oracle |
| --- | ---: | ---: | ---: |
| None | 0.89296 | 0.93148 | 0.97312 |
| Folded representations | 0.91388 | 0.94723 | 0.97859 |
| Address combinations | 0.97079 | 0.98066 | 0.99088 |
| Skeleton fragments | 0.90390 | 0.94046 | 0.97848 |
| All additional routes | 0.97389 | 0.98359 | 0.99365 |

The ablation baseline reproduces the corrected pilot exactly using the enhanced
index restricted to original key families. Address combinations produce most of
the improvement; folded and skeleton routes provide smaller additional gains.
All enhanced queries have usable keys. At K=100, 6,484 true links are present,
307 rank below K, 129 have all shared keys pruned, and four share no raw key.

Enhanced route-reserved K=1000 pool oracle is 0.98889. Address-aware reranking
of that pool yields oracle 0.97867 at K=100 and 0.98630 at K=300. At K=20, its
0.94691 oracle is worse than enhanced IDF's 0.95817; the reranker should not be
adopted unconditionally at every K. Name-only reranking again regresses.

The enhanced six-worker count/index scan took 389.25 seconds and the 2k query
pass took 21.67 seconds. Pool materialization and reranking took 391.09 seconds.
These are exploratory pilot measurements, not model or public scores.

## Larger confirmation and pair decisions

```powershell
& $pythonExe -m business_entity_resolution.src.retrieval_pilot --size 50000 --prepare-only --output reports/retrieval_confirmation50k
& $pythonExe -m business_entity_resolution.src.retrieval_experiment --pilot-dir reports/retrieval_confirmation50k --output reports/retrieval_v3_confirmation50k --workers 6 --skip-rerank
& $pythonExe -m business_entity_resolution.src.pair_model prepare
& $pythonExe -m business_entity_resolution.src.retrieval_experiment --pilot-dir reports/pair_model/train --output reports/pair_model/train_index --workers 4 --skip-rerank
& $pythonExe -m business_entity_resolution.src.pair_model fit
& $pythonExe -m business_entity_resolution.src.pair_audit
```

The pair harness refuses to train if the 50k candidate oracle at the selected K
is below 0.98. It samples 2,000 training-assigned S1 entities, uses every actual
top-300 candidate as a positive/hard negative, and gives each S1 equal total
training weight. A compact histogram gradient-boosted decision model learns 19
name/address/retrieval features. No ID, country, or true match count is a feature;
the candidate's source S2/S3 is allowed. Every pair above the threshold is
accepted, with no one- or three-match output cap.

The original 2k development pilot selects the threshold for entity macro F0.5.
A separate 2k dev subset disjoint from the pilot is used once for pair-model
confirmation. Model-training entities are separate from both, and overlapping
S2/S3 truth IDs between training and the full 50k validation set cause an error.
The 50k retrieval confirmation includes the pilot; it is a larger retrieval
check, not an entirely untouched sample. Pair-confirmation entities were not
used to fit the classifier or select its threshold.

`reports/pair_model/summary.json` records entity F0.5, pair precision/recall,
singleton accuracy, all-match completeness, match-count diagnostics, country
scores, oracle gap, versions, signatures, and model hash. Local confirmation
matching/candidate TSVs and the trained model are saved beside it. These TSVs
contain validation S1 entities and are **not test submissions**.

Optional modeling dependencies are pinned in `requirements-model.txt` to the
already-installed versions. The learner library is scikit-learn 1.9.1 under
BSD-3-Clause; no external pretrained model/weights are used. This records library
provenance and does not label that library MIT/Apache or settle final package
model-licensing requirements.

### 50k retrieval confirmation

Frozen dev: 50,000 S1 entities, 173,070 true links, 2,792 singletons. No settings
were changed between the enhanced pilot and this larger check.

| K | Pair recall | Oracle macro F0.5 |
| ---: | ---: | ---: |
| 20 | 0.89797 | 0.95753 |
| 50 | 0.92162 | 0.96788 |
| 100 | 0.93649 | 0.97476 |
| 300 | 0.95939 | 0.98480 |
| 1000 | 0.97190 | 0.98967 |
| Full usable pool | 0.98409 | 0.99408 |

At K=300, India oracle is 0.97406 and US oracle is 0.99197. At K=1000 they are
0.98257 and 0.99441 respectively. The global gate is met at K=300, but 0.98480
leaves only 0.00480 for classifier errors before falling below the 0.98 target.
India remains harder, and classifier quality must be measured directly.

Only two queries have zero usable keys. K=100 losses: 162,078 in the list,
8,239 ranked lower, 2,699 with every shared key pruned, 54 sharing no raw key.

The first six-worker index construction took 852.93 seconds; saving its cache
took 16.26 seconds, with parent peak RSS about 1,212 MB. The oracle-only query
path was then optimized to skip unused route sets and rerun from that cache:
loading took 5.71 seconds and all 50k queries took 300.20 seconds, with parent
peak RSS about 1,061 MB. Rank/score equivalence is covered by a regression test.
The query-only summary records the cache reuse; the original construction
timing is recorded here from the first execution.

Training, tuning, and confirmation feature preparation produced 1.8 million
candidate-pair rows in 491.29 seconds including index loading, pool selection,
three streaming record joins, and feature extraction. Parent peak RSS was about
1,596 MB. This mixed-phase timing should not be multiplied blindly to predict
full-test inference time.

### First trained decision model

The training sample contains 596,160 scored pairs, including 6,663 positives.
All negatives come from actual top-300 retrieval lists. Training and prediction
took 6.39 seconds after cached features were loaded. Threshold 0.60 was selected
on the original 2k pilot; it was applied unchanged to the separate confirmation
entities.

| Metric | Threshold-selection pilot | Separate 2k confirmation |
| --- | ---: | ---: |
| Entity macro F0.5 | 0.84371 | 0.84009 |
| Candidate oracle | 0.98359 | 0.98695 |
| Pair precision | 0.90979 | 0.89862 |
| Pair recall | 0.78365 | 0.79786 |
| Singleton accuracy | 0.76577 | 0.60714 |
| All true matches recovered (linked entities) | 0.50926 | 0.52278 |
| US entity F0.5 | 0.88628 | 0.88853 |
| India entity F0.5 | 0.77985 | 0.76743 |

Confirmation predicts 6,145 links, of which 5,522 are correct, against 6,921 true
links. Average predicted matches is 3.0725 versus 3.4605 true matches. The output
does not have a fixed per-entity match cap. There were **zero** shared true S2/S3
IDs between the sampled training entities and the full 50k validation set.

An independent audit reads the written confirmation TSVs and original labels
and reproduces F0.5 **0.8400873644551493** exactly. It also checks prediction
containment in candidate lists. Entities with a cross-script true link score
0.71078 (249 entities), versus 0.85848 without one (1,751 entities). Entities
with a missing address on at least one true link score 0.78545 (289 entities).
Private error rows are saved in `reports/pair_model/confirmation_errors.tsv`.
This is diagnostic evaluation; the confirmation results were not used to
change thresholds or refit the model.

Model SHA-256:
`54ff746fcc9f7c39143b411ed7b4e060a47b9e32127973ef251cbaf93ce27097`.

The historical local baseline of 0.4389 was evaluated on 50k entities, whereas
this model result is on a separate 2k confirmation sample. They are not a paired
same-entity comparison. Neither the model's test score nor a France score has
been measured. The **0.98 prediction target has not been reached**: the current
0.14686 gap to candidate oracle is primarily a pair-decision problem. Further
work should focus on stronger cross-script pair features, more representative
training examples, and conservative false-positive/singleton decisions, using
development data and a fresh confirmation set for any subsequent iteration.

Verification: 35 unit/regression tests pass, covering scoring, singleton and
multi-sibling handling, duplicate keys, caps, Unicode/Indic routes, projection
equivalence, sequential/parallel equality, caches, feature ID exclusion, and
written output behavior. Full test inference and uploads were not run.

The existing submission was left unchanged. Its SHA-256 is
`fccc34b0fc41726621823e3b64410d79960b85c69b2b6919df2976265d8d8015`
(`submission_log/Submission1_Day2/matching_results/matching_results.tsv`).

## Decision gate

0.98 is a target, not a measured prediction score. Candidate oracle is an upper
bound using truth and must not be reported as model accuracy. The corrected v2
full-pool oracle of 0.97312 cannot support 0.98 on this pilot at any K. Key recovery
is required, alongside ranking improvements. Before any full-test inference,
require strong pilot retrieval, a 50k confirmation, trained pair decisions on
training-assigned entities, separate held-out threshold/confirmation evaluation,
and the official validator. The current changes do not produce or upload a
replacement submission.
