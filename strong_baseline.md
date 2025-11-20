# Strong Baseline

This file documents the **strong baseline** used in our experiments.  
For all the TDC binary classification tasks, we tested the Intern-S1 model on those classification tasks' test set.

Because the original TDC tasks are only (molecule, bbinary label) pairs, the Intern-S1 model cannot directly predict the labels.

So we take the templates from Tx-Gemma and fit all the tasks into natural language questions. 

To run the simplre baseline, execute the following command:
```bash
python strong_baseline.py
```

The resulting **task-wise AUROC** scores are:

| Task | AUROC  |
|------|--------|
| hERG | 68.55% |
| AMES | 65.53% |
| DILI | 70.83% |
| Skin Reaction | 45.02% |
| ToxCast | 57.67% |
| Tox21 | 63.81% |
| ClinTox | 62.05% |
| herg_central | 56.73% |
| PAMPA_NCATS | 74.86% |
| HIA_Hou | 96.39% |
| Bioavailability_Ma | 50.55% |
| BBB_Martins | 79.84% |
| Pgp_Broccatelli | 57.64% |
| CYP1A2_Veith | 77.60% |
| CYP2C19_Veith | 59.13% |
| CYP2C9_Veith | 65.36% |
| CYP2D6_Veith | 60.30% |
| CYP3A4_Veith | 66.08% |
| CYP2C9_Substrate_CarbonMangels | 49.82% |
| CYP2D6_Substrate_CarbonMangels | 58.47% |
| CYP3A4_Substrate_CarbonMangels | 59.72% |
| HIV | 62.76% |
| SARSCoV2_3CLPro_Diamond | 68.60% |
| SARSCoV2_Vitro_Touret | 47.37% |
| butkiewicz | 54.01% |
| SAbDab_Chen | 60.54% |
| HuRI | 49.25% |
| Weber | 51.52% |
| MHC1 IEDB | 48.4%  |
| MHC2 IEDB | 50.83% |

This strong baseline serves as a practical model, whose AUROC scores are higher than 0.5.
