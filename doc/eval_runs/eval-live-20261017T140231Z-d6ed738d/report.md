# Evaluation report — `eval-live-20261017T140231Z-d6ed738d`

> **Caveat:** Cases are the 6-case SYNTHETIC smoke set (`consilium-smoke-v1`): synthetic CAD, synthetic notes. Not evidence about real patients or images.
> **Caveat:** Run produced from a working tree with uncommitted changes.

## Provenance

| Field | Value |
|---|---|
| Mode / LLM backend | live / openai |
| Dataset | `consilium-smoke-v1` sha256 `e07a0aab2561` (6 cases, split all) |
| Stages x samples | S1, S2, S3, S4 x 3 (seed 13) |
| Git | `5be7f313cb6b` (dirty) |
| Models | critic=gpt-4o-mini-2024-07-18@T=0.0, director=gpt-4o-mini-2024-07-18@T=0.2, single_agent=gpt-4o-mini-2024-07-18@T=0.2, specialist=gpt-4o-mini-2024-07-18@T=0.2 |
| Prompts | critic=v1@c19ed3103142, director=v1@057c2a1568e4, ophthalmologist=v1@57c5a8d65f1e, optometrist=v1@d24e7c377cf6, pharmacist=v1@7ca0d3b3ed68, single_agent=v1@3b907d921167 |
| Routing | `3351ba47201f` (S2 shared `33164568fb75`) |
| Critic max rounds | 3 |
| Instrument | deterministic verifier (verify_draft) on every stage's final draft; embedding `hashed-ngram-tf@v1` |

## Per-stage metrics (mean [95% bootstrap CI] (n cases))

| Metric | S1 | S2 | S3 | S4 |
|---|---|---|---|---|
| Unsupported-claim rate | 0.111 [0.019, 0.204] (n=6) | 0.088 [0.032, 0.148] (n=6) | 0.071 [0.011, 0.147] (n=6) | 0.030 [0.000, 0.070] (n=6) |
| Omission rate | 0.329 [0.278, 0.375] (n=6) | 0.358 [0.333, 0.386] (n=6) | 0.331 [0.292, 0.367] (n=6) | 0.074 [0.000, 0.185] (n=6) |
| Unsafe-language drafts | 0.000 [0.000, 0.000] (n=6) | 0.000 [0.000, 0.000] (n=6) | 0.000 [0.000, 0.000] (n=6) | 0.000 [0.000, 0.000] (n=6) |
| Out-of-scope attribution rate | 0.000 [0.000, 0.000] (n=6) | 0.000 [0.000, 0.000] (n=6) | 0.000 [0.000, 0.000] (n=6) | 0.011 [0.000, 0.033] (n=6) |
| Minority-opinion preservation | n/a | 0.667 [0.667, 0.667] (n=2) | 0.167 [0.000, 0.333] (n=2) | 0.000 [0.000, 0.000] (n=2) |
| S_inter (cosine similarity) | n/a | 0.268 [0.213, 0.301] (n=6) | 0.218 [0.166, 0.260] (n=6) | 0.239 [0.174, 0.292] (n=6) |
| Unique-fact coverage | n/a | 0.380 [0.213, 0.481] (n=6) | 0.287 [0.111, 0.444] (n=6) | 0.278 [0.111, 0.444] (n=6) |
| Rounds to approval (S4) | n/a | n/a | n/a | n/a |
| UNVERIFIED rate (S4) | n/a | n/a | n/a | 1.000 [1.000, 1.000] (n=6) |
| Tokens per case | 1945.6 [1912.8, 1975.7] (n=6) | 7057.8 [6755.2, 7257.7] (n=6) | 6799.3 [6576.0, 6994.9] (n=6) | 17143.2 [16446.0, 17989.5] (n=6) |
| Cost per case (USD, est.) | 0.00042 [0.00041, 0.00043] (n=6) | 0.00159 [0.00149, 0.00166] (n=6) | 0.00154 [0.00145, 0.00161] (n=6) | 0.00380 [0.00354, 0.00407] (n=6) |
| Failure rate | 0.000 | 0.000 | 0.000 | 0.000 |
| wall_ms p50 / p95 | 2658 / 3515 | 8307 / 9655 | 8005 / 8731 | 23652 / 26841 |
| llm_latency_ms p50 / p95 | 2658 / 3514 | 13060 / 14868 | 12090 / 13844 | 27968 / 32249 |

## Paired per-case differences (later - earlier)

| Metric | S2-S1 | S3-S2 | S4-S3 | S4-S1 |
|---|---|---|---|---|
| Unsupported-claim rate | -0.023 [-0.130, 0.093] | -0.017 [-0.077, 0.051] | -0.042 [-0.134, 0.037] | -0.081 [-0.174, -0.009] |
| Omission rate | 0.030 [0.000, 0.067] | -0.028 [-0.083, 0.000] | -0.256 [-0.342, -0.148] | -0.255 [-0.356, -0.144] |
| Unsafe-language drafts | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| Out-of-scope attribution rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.011 [0.000, 0.033] | 0.011 [0.000, 0.033] |
| Minority-opinion preservation | n/a | -0.500 [-0.667, -0.333] | -0.167 [-0.333, 0.000] | n/a |
| S_inter (cosine similarity) | n/a | -0.050 [-0.083, -0.024] | 0.021 [0.005, 0.038] | n/a |
| Unique-fact coverage | n/a | -0.093 [-0.259, 0.000] | -0.009 [-0.028, 0.000] | n/a |
| Rounds to approval (S4) | n/a | n/a | n/a | n/a |
| UNVERIFIED rate (S4) | n/a | n/a | n/a | n/a |
| Tokens per case | 5112.3 [4854.8, 5288.4] | -258.5 [-337.8, -164.9] | 10343.9 [9756.2, 11038.8] | 15197.7 [14516.4, 16027.1] |
| Cost per case (USD, est.) | 0.00118 [0.00108, 0.00124] | -0.00005 [-0.00010, -0.00001] | 0.00226 [0.00206, 0.00246] | 0.00338 [0.00313, 0.00365] |

## Hypotheses (negative and null results are reported)

| Claim | Test | Status |
|---|---|---|
| routing reduces reasoning convergence | S3-S2 s_inter lower | supported |
| routing surfaces more role-specific facts | S3-S2 unique_fact_coverage higher | not shown (CI includes 0) |
| verification removes unsupported claims | S4-S3 unsupported_claim_rate lower | not shown (CI includes 0) |
| verification removes omissions | S4-S3 omission_rate lower | supported |
| full CONSILIUM beats a single agent | S4-S1 unsupported_claim_rate lower | supported |

## Critic fault injection (recall [Wilson 95%]; misses by name)

Critic: live Critic gpt-4o-mini-2024-07-18@T=0.0. Corpus: 7 known-good drafts.

| Fault class | deterministic | llm_critic | combined |
|---|---|---|---|
| numeric_gross | 7/7 [0.65, 1.00] | 7/7 [0.65, 1.00] | 7/7 [0.65, 1.00] |
| numeric_subtle | 7/7 [0.65, 1.00] | 7/7 [0.65, 1.00] | 7/7 [0.65, 1.00] |
| fabricated_with_ref | 0/7 [0.00, 0.35]; missed clean_v1, smoke-001, smoke-002, smoke-003, smoke-004, smoke-005, smoke-006 | 7/7 [0.65, 1.00] | 7/7 [0.65, 1.00] |
| fabricated_no_ref | 7/7 [0.65, 1.00] | 7/7 [0.65, 1.00]; 5 critic error(s), fail-closed | 7/7 [0.65, 1.00] |
| deleted_mandatory | 7/7 [0.65, 1.00] | 7/7 [0.65, 1.00] | 7/7 [0.65, 1.00] |
| dose_insertion | 7/7 [0.65, 1.00] | 7/7 [0.65, 1.00] | 7/7 [0.65, 1.00] |
| laterality_flip | 6/7 [0.49, 0.97]; missed smoke-003 | 7/7 [0.65, 1.00] | 7/7 [0.65, 1.00] |
| suppressed_disagreement | 3/3 [0.44, 1.00] | 3/3 [0.44, 1.00] | 3/3 [0.44, 1.00] |
| clean false-rejection rate | 0.000 | 1.000 | 1.000 |

## Clinician rubric

No clinician scores attached to this run. Pipeline dry run only: SYNTHETIC rater scores + canned judge: pipeline dry run, NOT clinical data (judge gpt-4o-mini-2024-07-18@T=0.0).

## Threats to validity

- Same base model for every agent: convergence may persist regardless of routing (AGENTS.md §24.6) — the ablation, not the design, decides.
- `S_inter` uses a lexical eval-only embedding; paraphrases count as diverse. Values are embedding-dependent and not comparable across embedders.
- Small N (6 cases): wide intervals; read paired per-case differences, not just means.
- The deterministic verifier is both S4's first audit layer and the claim-level instrument, so S4 is measured by a ruler it was optimised against.
- Cost uses configured list prices (configs/eval.yaml), an estimate.

_Generated by `consilium.eval.report` from stored run `eval-live-20261017T140231Z-d6ed738d` only._
