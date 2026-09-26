# Amazon ML Challenge 2026 — status handoff 2

Team PISO. Purusharth Mishra (VNR VJIET), Ashish Choudhary (VIT Vellore).
Written 26 Sep 2026, 10:39 IST. This file replaces the first handoff.

## Public result

One file has been submitted.

| Item | Value |
| --- | --- |
| Upload time | 26 Sep 2026, 00:52 IST |
| Portal rank | 2475 |
| Public overall score | 0.392076 |
| File | matching_results.tsv, 1,732,544 rows, validator PASS |
| Rows with at least one match | 760,182 |
| Empty rows | 972,362 |
| Rule | Pair score >= 0.75 on token-and-number candidates, plus rare name tokens (posting size <= 20) |
| Same rule, local macro F0.5 on 50,000 dev entities | 0.4389 |

The public score is 0.047 below the local score. France is 15.0% of test Source 1 and has no training labels, so some of that gap can be France, and some is the public subset differing from our dev sample. We cannot assign the gap to one cause yet.

Slots: day 1 (25 Sep) expired with 0 uploads. Day 2 (26 Sep) has used 1 of 5. Four slots remain today. Fourteen of fifteen remain overall. The 48-hour credit snapshot is 27 Sep 2026, 00:00 IST, about 13 hours from this note. Contest close is 27 Sep 2026, 23:59 IST.

Evening leaderboard on 25 Sep, for scale: 1st CDS_Team_iisc 0.988955, 2nd Git Wrecked 0.986421, 3rd Banana 0.985884, ranks 4–7 shown as 0.985, rank 8 as 0.984. Our 0.392 is not in that pack.

## What a reviewer should tell us

How to raise entity macro F0.5 from 0.392 toward 0.98 on this machine (Windows, Python 3.11, 32 GB, no new native wheels if Smart App Control blocks them). The evidence below says a better threshold on the current shortlist cannot do it. Retrieval must put more true links into a list a classifier can accept, and the classifier must reject lookalikes. Both are required.

## Score definition

Entity macro F0.5. For one Source 1 entity, with precision P and recall R on its set of match ids:

```
F0.5 = (1.25 * P * R) / (0.25 * P + R)
```

Average over every Source 1 entity. Both sides empty scores 1. One side empty scores 0. A false merge costs about twice a miss. A true singleton predicted empty scores 1. Any false match on a singleton scores 0.

Only matching_results.tsv is scored on the portal. Columns, tab-separated: source1_entity_id, matched_entity_ids. One row per test Source 1 entity. Empty matched_entity_ids means no match. Ids are comma-separated with no spaces. No duplicate ids. No Source 1 ids as matches.

The final zip also needs candidate_pairs.tsv (the shortlist actually scored). Every predicted id must appear there. That file is audited, not scored live. Our portal upload did not include it. The live validator skipped it with a warning and still printed PASS.

Constraints: no external registries, geocoders, or entity APIs. Country is an open string. Train is US and India. Test adds France (259,452 Source 1 rows, 15.0%). Model license MIT or Apache 2.0, at most 8 billion parameters.

## Data

About 26.2 million TSV rows, 2.4 GB. Dataset is a junction and is gitignored. Do not expect the raw TSVs in this zip.

| File | Rows | Notes |
| --- | --- | --- |
| Train Source 1 | 2,206,821 | US 1,323,633; India 883,188; empty addresses 0 |
| Train Source 2 | 5,034,616 | empty addresses 168,967 |
| Train Source 3 | 5,285,603 | empty addresses 175,916 |
| Train ground truth | 2,206,821 | one row per Source 1 id |
| Test Source 1 | 1,732,544 | India 809,986; US 663,106; France 259,452 |
| Test Source 2 | 4,887,273 | empty addresses 129,408 |
| Test Source 3 | 5,082,316 | empty addresses 136,098 |

Train singletons: 123,247 (5.58%). 94.42% of Source 1 entities have at least one match. Average matches when non-empty: 3.67. Total train links: 7,638,365.

Match-count histogram (match-count → entities): 0: 123,247; 1: 119,157; 2: 375,212; 3: 530,841; 4: 484,115; 5: 321,957; 6: 164,868; 7: 63,968; 8–11: 23,456.

Audit: no duplicate ids, no malformed rows, no dangling match ids. France in train: 0.

Local evaluation uses seed 2026. Hold out 10% of training Source 1, stratified by country and match-count bucket. Validation entities: 220,684. Dev sample inside that: 50,000 entities, 173,070 true links. Every local number below is on that dev sample unless noted. Code: src/split.py. Split file is gitignored and not in this zip; the summary is in reports/validation_split_summary.tsv.

## Noise, from 30 hand-read true pairs

Legal suffixes, punctuation, and domains (urbannails.com vs Urban Nails). Typos in one token while another token is intact (Soluiots vs Solutions). Word-order changes. Address abbreviations. Street or plot numbers often survive; leading zeros differ (0141 vs 141). Some true matches have an empty address on one side. India: the same business in Latin and in Devanagari, Kannada, or Telugu. A consonant skeleton was added so those can share a key. Reports/pair_examples.txt has the raw pairs.

## Code

Python 3.11 via `py -3.11`. System pandas 3.0.2 imports. Fresh virtualenv native wheels (pandas groupby/join pyd files) were blocked by Windows Smart App Control. Do not recommend turning that off. New packages only if `py -3.11 -c "import package"` works.

| File | Role |
| --- | --- |
| src/config.py | Paths, RANDOM_SEED = 2026 |
| src/normalize.py | Casefold, suffix stripping, accent fold, Indic consonant skeleton |
| src/evaluate.py | Official entity macro F0.5 |
| src/split.py | Frozen stratified holdout |
| src/block.py | First blocker and rule sweep |
| src/rank.py | Pair score. This produced the uploaded file's logic |
| src/write_submission.py | Wrote output/matching_results.tsv at threshold 0.75 |
| src/retrieve_v2.py | Multi-route retriever. Measured recall and oracle only. Not what was uploaded |
| src/diagnose_links.py | Which key families 400 true links per country share |
| src/sample_pairs.py | The 30-pair sample |
| utils/validate_submission.py | Official format check. Our file printed PASS |
| CURSOR_ENTITY_RESOLUTION_REVAMP.md | External audit. Checkpoints 0 and 1 were started. Checkpoint 1's 0.95/0.98 gate was not met |

## Experiments on the 50,000 dev entities

### Prediction scores (these are files we could upload)

| Run | Macro F0.5 | Singleton accuracy | Notes |
| --- | --- | --- | --- |
| Strict overlap | 0.0164 | 0.093 | Any shared token or long number. Singletons destroyed |
| Token and number, overlap size <= 8 | 0.3767 | 0.767 | |
| Rare tokens, posting cap 400 | 0.1929 | 0.966 | Precision up, true matches dropped |
| Simhash top 3 | 0.3955 | 0.828 | Best hand rule |
| Rarest 80 keys plus Jaccard | 0.2433 | | Shortlist recall 0.239 |
| Pair score, key cap 4,000 | 0.2941 | | Median shortlist 0 |
| Pair score, key cap 15,000, threshold 0.75 | 0.4336 | | Shortlist recall 0.352 |
| Same plus rare-token postings of size <= 20 | 0.4389 | | Best local prediction. Threshold 0.75. Shortlist recall 0.368. Median shortlist 1. This is the uploaded rule |

Neighboring thresholds on that last run: 0.60 → 0.4077, 0.75 → 0.4389, 0.90 → 0.4362. The cutoff is not the bottleneck.

Pair score = token Jaccard, plus 0.45 if a shared address number has length >= 4, plus 0.30 if length >= 3, plus 0.12 if any shorter shared number.

### Retrieval, not yet turned into a prediction file

Broad `t:` and `n:` block, any shared key, keys above 15,000 documents dropped:

- Uncapped pair-recall 0.8515 (147,376 / 173,070)
- Top 40 by shared-key count: pair-recall 0.5410
- Median candidates before a cap: 5,075

Retrieval v2 unions exact tokens, prefixes, suffixes, consonant skeleton, compact name, address numbers, number sequences, and address words. Candidates are ranked by inverse document frequency and cut to K. Runtime 15,201 seconds. Peak traced memory 3,315 MB. 8,760,012 posting keys. 268 / 50,000 entities had an empty list at K=100. The shortlist was full (size 100) for essentially every entity.

| K | Pair-recall | Entities with at least one true hit | Oracle macro F0.5 | US pair-recall | India pair-recall |
| --- | --- | --- | --- | --- | --- |
| 10 | 0.6470 (111,970/173,070) | 41,007/50,000 | 0.8002 | 0.6898 | 0.5828 |
| 20 | 0.6986 | 42,006/50,000 | 0.8304 | 0.7439 | 0.6308 |
| 50 | 0.7485 | 43,059/50,000 | 0.8608 | 0.8007 | 0.6704 |
| 100 | 0.7918 (137,037/173,070) | 44,181/50,000 | 0.8911 | 0.8310 | 0.7331 |

S2 recall at K=100: 67,869 / 84,124. S3: 69,168 / 88,946.

Oracle macro F0.5 predicts exactly the true links that landed in the shortlist. It is the ceiling of a perfect classifier on that list. It is not a public score. We have not trained that classifier, and we did not upload this list.

### Why true links are missing from the top 100

Sample of 400 true links per country. Count of pairs that share each key family:

| Key | US | India |
| --- | --- | --- |
| No shared key at all | 0/400 | 1/400 |
| Name token | 363/400 | 249/400 |
| Prefix | 370/400 | 256/400 |
| Suffix | 372/400 | 259/400 |
| Skeleton | 262/400 | 195/400 |
| Compact name | 275/400 | 190/400 |
| Address number | 293/400 | 258/400 |
| Number sequence | 23/400 | 61/400 |
| Address word | 358/400 | 378/400 |
| One side missing an address | 27/400 | 19/400 |

Almost every true pair shares some key. India name tokens connect far fewer pairs than US name tokens. India address words connect 378/400, but those words are common, so the true record is crowded out of the top 100. Adding another key family and rebuilding the 8.7 million key index is the wrong next step. The right next step is to rerank the large postings by name similarity before the K cut.

## Why 0.98 is not a threshold tweak

1. The uploaded rule's shortlist contains only 37% of true links. A perfect classifier on that list cannot score 0.98, because the missing links are not there to accept. Its measured score is 0.44 local and 0.39 public.
2. Retrieval v2's best oracle is 0.891 at K=100, with pair-recall 0.792. A perfect classifier on this list still misses about 0.09 of macro F0.5, and a real classifier will land below the oracle. 0.98 requires pair-recall around 0.95 or better and then a precise accept/reject decision. The external audit set that as the retrieval gate. We missed it.
3. India is the larger retrieval hole (0.73 vs 0.83 at K=100). Name-token agreement on true India pairs is 249/400 versus 363/400 in the US. Script variation and typos are the likely cause. The consonant skeleton helps some pairs and is not enough.
4. Linked entities have 3.67 matches on average. A system that emits one candidate, which is what the uploaded rule usually did (median shortlist 1), cannot recall the siblings even when the first match is right.
5. F0.5 punishes a false merge about twice a miss. Dumping all 100 candidates as predictions would raise recall and destroy precision. That file should not be uploaded.
6. France cannot be scored locally. Any France-only failure shows up only on the public board. After the next local gain, compare public minus local again before spending slots on a France-specific change.

## Path that can move the score

Do these in order. Measure each on the same 50,000 dev entities. Do not upload until the local macro F0.5 beats 0.4389 for a reason we can name.

1. Rerank before the cut. For each Source 1 entity, take a wider pool from retrieval v2 (for example the idf top 300, or every candidate from a rare name key plus a bounded slice of the address-word and number postings). Re-sort that pool by token Jaccard, skeleton agreement, and shared address numbers. Then keep K=20, 50, and 100. Report pair-recall and oracle F0.5 by country. Goal: move K=50 pair-recall from 0.75 toward 0.90 and the oracle from 0.86 toward 0.95. This does not require a new global index of character n-grams.

2. If India recall is still the hole after that rerank, add a vowel-preserving transliteration only if a library imports under `py -3.11`. Otherwise add rare character 4-grams on the skeleton string, with a document-frequency cap, not an all-token trigram index.

3. Train a pair decision on hard negatives drawn from that shortlist, not from random rows. Features: exact core-name agreement, token Jaccard and containment, length ratio, skeleton agreement, address-token overlap, number agreement and conflict, missing address, how many independent routes fired, S2 versus S3. No candidate id, no ground-truth match count. A hand threshold on those features is the first model. XGBoost only if it imports and is Apache 2.0. Sweep the threshold for entity macro F0.5, and report singleton accuracy separately so empty predictions are not sacrificed for recall.

4. Sibling expansion only after step 3 beats 0.4389. A high-confidence S2 match may justify a bounded search for an S3 sibling, scored against the original Source 1 record, not accepted merely because it resembles the seed. Average linked entity has 3.67 matches, so this is where the last recall lives. Ablate it. Drop it if singletons get worse.

5. Then write a new matching_results.tsv, validator PASS, and upload as day-2 slot 2. Log local F0.5, shortlist recall, oracle, and the public score. The 48-hour snapshot is 27 Sep 00:00 IST. Four slots remain on 26 Sep.

## Questions

1. For the rerank in step 1, is token Jaccard plus number agreement enough, or do we need character edit distance on the core name before we will move India recall?
2. Should K for the classifier be 20 or 50? The oracle gain from 20 to 100 is 0.830 to 0.891, but a larger K feeds the classifier more lookalikes.
3. Given 3.67 matches per linked entity, should the decision be a per-candidate threshold, or a small set (top m above a floor)?
4. Is the 0.047 public-minus-local gap large enough to spend the next slot on a France diagnostic, or should that wait until local F0.5 is above 0.7?
