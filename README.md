# Lab II — Survival analysis of pediatric bone-marrow transplants

Baseline-feature clustering, Kaplan–Meier curves, and Cox regression on the UCI *Bone marrow transplant: children* cohort (187 patients).

## Design choices

- Predictors are restricted to information available at transplant. Recovery times, relapse, and GvHD fields are excluded (several of those columns also use the sentinel value `1000000`).
- Association with survival is assessed with univariate Cox models and log-rank tests. Pearson correlation with `survival_time` is saved only as a contrast, because censoring makes it misleading.
- Patients are clustered with K-means in a PCA space that keeps 85% of the variance of the standardized baseline features. `K` maximizes silhouette subject to a minimum cluster size of 15. DBSCAN on the same space is a secondary check.
- Cox model A uses eight covariates (about 10 events per variable). Model B adds cluster indicators. Model C is a ridge fit on the full clustering design, used to check coefficient signs.
