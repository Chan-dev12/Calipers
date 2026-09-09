# Ablation results

| Experiment   | Recall@5       | MRR            | NDCG@10        | P@5            | NDCG@10 95% CI | Time |
|--------------|----------------|----------------|----------------|----------------|----------------|------|
| E0_baseline  | 0.572          | 0.618          | 0.589          | 0.294          | [0.466, 0.708] | 1s   |
| E1_recursive | 0.480 (-0.092) | 0.531 (-0.087) | 0.519 (-0.070) | 0.250 (-0.044) | [0.403, 0.631] | 1s   |
| E3_bm25      | 0.560 (-0.012) | 0.639 (+0.022) | 0.579 (-0.010) | 0.244 (-0.050) | [0.447, 0.693] | 1s   |
| E4_hybrid    | 0.579 (+0.006) | 0.659 (+0.041) | 0.606 (+0.018) | 0.256 (-0.039) | [0.501, 0.699] | 2s   |

n = 36 questions. Deltas are vs E0_baseline. CI by bootstrap resampling, 1000 iterations.

**Caveats**
- E1_recursive: NDCG@10 interval overlaps the baseline's. With n=36 this difference is not statistically distinguishable.
- E3_bm25: NDCG@10 interval overlaps the baseline's. With n=36 this difference is not statistically distinguishable.
- E4_hybrid: NDCG@10 interval overlaps the baseline's. With n=36 this difference is not statistically distinguishable.
