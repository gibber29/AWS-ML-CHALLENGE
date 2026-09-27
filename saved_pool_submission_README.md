# Score an already completed test candidate pool

These scripts reuse `reports/test_pools.sqlite3` from the running
`write_model_submission.py` job. They do not rebuild its retrieval index.

## Before the pool finishes

From a separate PowerShell window in the repository root, preserve the old
leaderboard file:

```powershell
Copy-Item output/matching_results.tsv output/baseline_0392076.tsv
```

Wait for `Pooled 1,732,544 entities, ... unique candidates` in the original
job's log. That line is printed after the SQLite transaction commits. Then
interrupt the original writer while it scans candidate records, before it starts
`scored ...`. Do not stop during the `pooled ...` progress lines.

## Run from the repository root

Copy `quick_k_sweep.py` and `score_saved_pools.py` to the repository root.
Use the existing Python 3.11 installation and environment.

```powershell
py -3.11 quick_k_sweep.py
```

This prints confirmation macro F0.5, oracle, and country scores for K=20, 50,
100, 150, 300 with the trained model's saved threshold. Select a K based on
the measured score and remaining runtime. The 2,000-entity confirmation set is
small, so closely spaced scores are uncertain. Do not assume K=20 equals K=300.

```powershell
py -3.11 score_saved_pools.py --k 20
```

Replace 20 with the selected K (1 to 300). The script refuses to proceed until
all 1,732,544 pools exist, verifies the model checksum and feature order, and
does not overwrite existing output. It writes both TSVs under
`output/model_k20/` (or the corresponding K). The matching file is first
written as a `.tmp` file and renamed after all rows are written. A complete
pool still requires scanning test Source 2 and 3 once and scoring up to K times
1,732,544 pairs; this is not an instant conversion.

```powershell
py -3.11 utils/validate_submission.py --matching output/model_k20/matching_results.tsv --candidate output/model_k20/candidate_pairs.tsv --test-dir dataset/test
```

Upload only the validated `matching_results.tsv` to the live portal. Preserve
both TSVs for the final package. Do not upload a `.tmp` file or raw candidates
as matches. This system measured about 0.84 macro F0.5 on a separate 2,000
entity confirmation set at K=300; neither a higher public score nor 0.98 is
established by this scoring run.
