# Ablation results

| Experiment   | Recall@5       | MRR            | NDCG@10        | P@5            | NDCG@10 95% CI | Time |
|--------------|----------------|----------------|----------------|----------------|----------------|------|
| E0_baseline  | 0.511          | 0.680          | 0.620          | 0.347          | [0.445, 0.784] | 2s   |
| E1_recursive | 0.436 (-0.075) | 0.572 (-0.108) | 0.520 (-0.100) | 0.305 (-0.042) | [0.361, 0.677] | 1s   |
| E3_bm25      | 0.392 (-0.120) | 0.582 (-0.098) | 0.469 (-0.151) | 0.242 (-0.105) | [0.325, 0.628] | 0s   |
| E4_hybrid    | 0.450 (-0.062) | 0.675 (-0.004) | 0.557 (-0.064) | 0.274 (-0.074) | [0.394, 0.715] | 1s   |

n = 19 questions. Deltas are vs E0_baseline. CI by bootstrap resampling, 1000 iterations.

**Caveats**
- E1_recursive: NDCG@10 interval overlaps the baseline's. With n=19 this difference is not statistically distinguishable.
- E3_bm25: NDCG@10 interval overlaps the baseline's. With n=19 this difference is not statistically distinguishable.
- E4_hybrid: NDCG@10 interval overlaps the baseline's. With n=19 this difference is not statistically distinguishable.
