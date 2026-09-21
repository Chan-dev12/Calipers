# Ablation results

| Experiment                 | Recall@5       | MRR            | NDCG@10        | P@5            | NDCG@10 95% CI | Chunks | Mean chars | Time  |
|----------------------------|----------------|----------------|----------------|----------------|----------------|--------|------------|-------|
| E0_baseline                | 0.684          | 0.532          | 0.558          | 0.186          | [0.414, 0.704] | 2263   | 987        | 2s    |
| E1_recursive               | 0.589 (-0.095) | 0.500 (-0.032) | 0.525 (-0.033) | 0.171 (-0.014) | [0.395, 0.647] | 2647   | 860        | 2s    |
| E1b_recursive_java         | 0.452 (-0.232) | 0.463 (-0.070) | 0.442 (-0.116) | 0.150 (-0.036) | [0.319, 0.569] | 2953   | 780        | 2s    |
| E1c_recursive_java_matched | 0.542 (-0.143) | 0.407 (-0.125) | 0.436 (-0.123) | 0.150 (-0.036) | [0.324, 0.563] | 2640   | 857        | 636s  |
| E2_semantic                | 0.696 (+0.012) | 0.452 (-0.080) | 0.534 (-0.024) | 0.164 (-0.021) | [0.409, 0.665] | 2205   | 836        | 1993s |
| E3_bm25                    | 0.643 (-0.042) | 0.509 (-0.023) | 0.553 (-0.006) | 0.150 (-0.036) | [0.404, 0.692] | 2263   | 987        | 1s    |
| E4_hybrid                  | 0.732 (+0.048) | 0.654 (+0.122) | 0.695 (+0.137) | 0.186 (+0.000) | [0.558, 0.822] | 2263   | 987        | 7s    |
| E5_rerank                  | 0.732 (+0.048) | 0.603 (+0.071) | 0.663 (+0.105) | 0.186 (+0.000) | [0.531, 0.785] | 2263   | 987        | 3216s |
| E6_query_rewrite           | 0.750 (+0.066) | 0.626 (+0.094) | 0.689 (+0.131) | 0.193 (+0.007) | [0.574, 0.797] | 2263   | 987        | 72s   |

n = 28 questions. Deltas are vs E0_baseline. CI by bootstrap resampling, 1000 iterations.

**Caveats**
- E1_recursive: NDCG@10 interval overlaps the baseline's. With n=28 this difference is not statistically distinguishable.
- E1b_recursive_java: NDCG@10 interval overlaps the baseline's. With n=28 this difference is not statistically distinguishable.
- E1c_recursive_java_matched: NDCG@10 interval overlaps the baseline's. With n=28 this difference is not statistically distinguishable.
- E2_semantic: NDCG@10 interval overlaps the baseline's. With n=28 this difference is not statistically distinguishable.
- E3_bm25: NDCG@10 interval overlaps the baseline's. With n=28 this difference is not statistically distinguishable.
- E4_hybrid: NDCG@10 interval overlaps the baseline's. With n=28 this difference is not statistically distinguishable.
- E5_rerank: NDCG@10 interval overlaps the baseline's. With n=28 this difference is not statistically distinguishable.
- E6_query_rewrite: NDCG@10 interval overlaps the baseline's. With n=28 this difference is not statistically distinguishable.
