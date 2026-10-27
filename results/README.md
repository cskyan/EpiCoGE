# Reference results

`reference_per_seed.csv` contains the published main-model AsEP Test macro metrics
for the three training seeds, separately for Group and Ratio. Its thresholds are
the values selected on the corresponding original Val runs, supplied as reference
records rather than thresholds to force onto a newly trained model.

`reference_metrics.csv` gives the mean and sample standard deviation (`ddof=1`)
of those three seeds. Each metric is first calculated for each of the 170 Test
complexes and then averaged with equal complex weight. In particular, macro F1 is
the average of complex-level F1, not F1 calculated from the reported mean precision
and recall. AUROC uses continuous scores; the other metrics use score ≥ threshold.

| Column | Meaning |
|---|---|
| `split` | Official AsEP Group or Ratio protocol |
| `seed` | Training random seed |
| `threshold` | Corresponding original Validation-selected threshold |
| `auroc`, `mcc`, `precision`, `recall`, `f1` | Complex-macro metric |
| `<metric>_sd` | Sample SD across three training seeds |

Train your own EpiCoGE weights and select their thresholds on Val. Results can
vary with hardware, software versions and regenerated feature-cache arithmetic;
the reference files are not an assertion of bitwise reproducibility on other systems.
