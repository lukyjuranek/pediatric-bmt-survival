# Explainer script — Lab II, pediatric bone-marrow survival

Source of truth: `notebooks/lab2_survival.ipynb`, package `bmt`, and `tables/results.json`.
Target length: about four minutes. Eight scenes, each roughly 25–40 seconds of speech.

Style on screen: dark background, one amber accent, monospaced numbers, no emojis, no clip art.
Real names to keep visible: `run_analysis`, `survival_time`, `survival_status`, `load_arff`, `clean`, `build_cox_frame`, `fit_pca`, `select_k`, `fit_km`, `fit_cox`.

---

## Scene 1 — The question

**Narration** (~30 s)

This project asks one clinical question. After a child receives a bone-marrow transplant, which facts known on the day of transplant are tied to how long they survive? The notebook is Lab Two. It answers with three tools: grouping similar patients, Kaplan–Meier survival curves, and Cox regression. One function, `run_analysis`, runs the whole study on the UCI bone-marrow file. The goal is association at baseline. It is a lab analysis of one public cohort, not a score you would use to choose a transplant.

**On screen**

- Title: “Survival after pediatric bone-marrow transplant”
- Subtitle: “Lab II · UCI Bone marrow transplant: children”
- Three labeled cards fade in, left to right: Clustering → Kaplan–Meier → Cox regression
- A thin pipeline arrow underneath: `bone-marrow.arff` → `run_analysis` → `figures/` and `tables/results.json`

---

## Scene 2 — The cohort

**Narration** (~35 s)

The file is `bone-marrow.arff`: one hundred eighty-seven children, thirty-seven columns. The clock is `survival_time`, days from transplant, from six days out to three thousand three hundred sixty-four. The event is `survival_status`. One means the child died. Zero means follow-up ended while they were still alive. That zero is censoring: we know they lived at least that long, and we do not know what happened after. Eighty-five patients died. One hundred two were censored. A reverse Kaplan–Meier, which treats censoring as the event, puts median follow-up at one thousand five hundred twenty-two days. Ordinary median survival is never reached, because more than half the cohort is still alive at the last follow-up.

**On screen**

- A table row animating into shape `(187, 37)`, labeled `validate`: expected 187 rows, 37 columns, 85 deaths
- Two counters: `survival_status == 1` → 85 deaths; `== 0` → 102 censored
- A timeline from day 6 to day 3364, with a marker at median follow-up 1522
- A small overall Kaplan–Meier sketch: the curve stays above one half, annotation “median survival: not reached”

---

## Scene 3 — What is allowed in the model

**Narration** (~40 s)

Many columns look informative and are banned. Relapse, acute and chronic graft-versus-host disease, and the times `ANCrecovery`, `PLTrecovery`, and `time_to_aGvHD_III_IV` are observed only after transplant. Putting them in the model would leak the future into the predictors. Those three time columns also store the sentinel value one million when the time was not seen: five rows, seventeen rows, and one hundred forty-five rows. `load_arff` reads the file as latin-1, because the header is broken, and turns question marks into missing values. `clean` casts the codes to numbers and adds `log_CD34` and `log_CD3`. Forty-five rows have some missing field. Twenty-three are missing a true baseline field. The sentinel values stay out of every model matrix, because those columns are post-transplant.

**On screen**

- Two columns of variable chips
- Left, struck through, labeled `POST_TRANSPLANT`: Relapse, IIIV, aGvHDIIIIV, extcGvHD, ANCrecovery, PLTrecovery, time_to_aGvHD_III_IV
- Sentinel callout: `1000000` on those three time columns, counts 5, 17, 145
- Right, kept, labeled baseline: Recipientage, Donorage, CD34kgx10d6, Riskgroup, Disease, HLAmatch, Stemcellsource
- Shape change: raw `(187, 37)` → `clean` adds `log_CD34`, `log_CD3`

---

## Scene 4 — Two matrices

**Narration** (~35 s)

From the cleaned table the pipeline builds two matrices, and both use only pre-transplant columns. Clustering starts from thirteen inputs: recipient and donor age, the two log cell doses, six binary codes, two ordinal codes, and `Disease`. Disease is one-hot encoded into five levels: ALL, AML, chronic, lymphoma, nonmalignant. Median imputation for the continuous columns, most-frequent imputation for the codes, then a `StandardScaler`. The matrix that reaches PCA is one hundred eighty-seven by seventeen. Cox model A is thinner on purpose. `build_cox_frame` keeps eight covariates, about ten deaths per variable: recipient age per five years, donor age per ten years, log CD34 dose, and five yes-or-no codes. The largest variance inflation factor is two point one three, so those eight are not copies of one another.

**On screen**

- Data-flow diagram, shapes changing in place:
  - `df` `(187, 39)` after `clean`
  - `build_cluster_frame` → `(187, 13)`
  - one-hot `Disease` → `(187, 17)`
  - `StandardScaler` → same shape, columns now mean 0, scale 1
  - `build_cox_frame` → `(187, 8)`, column names `Recipientage_per5`, `Donorage_per10`, `log_CD34`, `Diseasegroup`, `Riskgroup`, `Txpostrelapse`, `HLAmismatch`, `Stemcellsource`
- Footer: “85 events / 8 covariates ≈ 10 events per variable” and “max VIF = 2.13”

---

## Scene 5 — PCA, then K-means

**Narration** (~40 s)

Clustering is principal components, then K-means. `fit_pca` keeps components until eighty-five percent of the variance is kept. That takes eleven components, and they actually hold eighty-nine percent. K-means then tries every K from two to eight, with fifty random starts and seed forty-two. `select_k` wants the highest silhouette, and it also demands that every cluster have at least fifteen patients. K equals six, seven, and eight score higher — silhouettes around zero point two two — but their smallest group has only nine patients, so they are thrown out. K equals five wins, with silhouette zero point one eight. The five groups, named from the data, are: ALL and low risk, forty-six patients; chronic disease with a poorer HLA match, thirty-six; mostly marrow and a lower CD34 dose, thirty-eight; AML and often a second transplant, thirty-six; and nonmalignant disease in younger children, thirty-one. A DBSCAN check on the same space, epsilon about five point six and twenty-two minimum samples, finds one blob and zero noise. It does not recover the five groups. Adjusted Rand index versus K-means is zero.

**On screen**

- Left: a PCA scatter of 187 dots in PC1–PC2 (16% and 12% of variance), colored as five clusters once K is chosen
- Right: silhouette bars for K = 2…8. K = 6, 7, 8 grayed out with a tag “min size 9 < 15”. K = 5 highlighted, silhouette 0.184
- Then a profile strip, one card per cluster, using the real names and sizes:
  - C0 “ALL, low-risk” n=46, events=17
  - C1 “chronic, poorer HLA match” n=36, events=17
  - C2 “mostly marrow, lower CD34” n=38, events=22, median 421 days
  - C3 “AML, often post-relapse” n=36, events=18, median 1243 days
  - C4 “nonmalignant, younger recipients” n=31, events=11
- Small footnote: “DBSCAN: 1 cluster, 0 noise, ARI = 0.00”

---

## Scene 6 — Kaplan–Meier and the log-rank test

**Narration** (~30 s)

Kaplan–Meier estimates the chance of still being alive as the days pass, and it uses the censored patients correctly: they contribute up to the day they leave, and then they drop out of the risk set. The notebook fits one curve for the whole cohort, then one curve inside each K-means group. The log-rank test asks whether those five curves are the same. The test statistic is five point four one, and the p-value is zero point two five. The groups do look different on paper. Cluster two, mostly marrow and a low CD34 dose, has a median of four hundred twenty-one days, while clusters zero, one, and four never reach a median. The curves are still compatible with a single survival distribution. Adding the cluster indicators in model B does not change that: the likelihood-ratio statistic is six point seven six on four degrees of freedom, p equals zero point one five.

**On screen**

- Five Kaplan–Meier step curves, days 0 to 3364, survival from 1.0 down toward 0
- Colors match scene 5. A dashed overlay is the mean Cox model-A curve inside each cluster, labeled “Cox mean”
- Annotation on C2: median 421 days. Annotation on C0, C1, C4: “median not reached”
- Result plate: log-rank statistic 5.41, p = 0.25. Beside it: model B likelihood-ratio 6.76 on 4 df, p = 0.15

---

## Scene 7 — The Cox model

**Narration** (~40 s)

The model to sit with is Cox regression. It says the hazard, the instantaneous risk of death, equals a baseline hazard times e to the beta times x. The baseline can be any shape. The covariates multiply it. A hazard ratio above one means higher risk; below one means lower risk. In model A the one coefficient whose interval stays off one is recipient age: per five years, the hazard ratio is one point three three, with p equal to zero point zero two four. The other seven confidence intervals cross one. Peripheral-blood stem cells lean protective, hazard ratio zero point seven zero, and the p-value is zero point two seven. A rank Schoenfeld test flags `HLAmismatch` at p equals zero point zero three four, so the reported fit stratifies on that covariate. Training concordance is zero point six three. Five-fold cross-validation, with the imputer refit inside each fold, drops it to zero point five six, plus or minus zero point zero four. Model C is a ridge fit, penalizer zero point one, on the full standardized clustering matrix. It is there to check that coefficient signs agree, not to compete on accuracy.

**On screen**

- Formula, built in two beats: `h(t | x) = h0(t) · exp(β · x)`, then “hazard ratio = exp(β)”
- A forest plot of model A, log scale, vertical line at 1, using the unstratified coefficients from `cox_a_summary.csv`:
  - Recipient age (per 5 years): HR 1.33 (1.04–1.70), p = 0.024 — the only interval clear of 1
  - Donor age (per 10 years): 1.13
  - High risk: 1.33
  - Second transplant after relapse: 1.06
  - log CD34 dose: 0.94
  - HLA mismatch: 0.93
  - Malignant disease: 0.90
  - Peripheral-blood stem cells: 0.70
- Footer metrics: train C-index 0.63 → 5-fold CV 0.56 ± 0.04. Stratum: `HLAmismatch`. Model C penalizer 0.1

---

## Scene 8 — What to take away

**Narration** (~30 s)

Here is the honest summary. Baseline features carry a weak signal, and recipient age is the association that survives a multivariable Cox model. The five clusters are a readable map of who is in this cohort. They are not a survival classifier: the log-rank p-value is zero point two five, and DBSCAN never finds them. The limits are the size and the design. One hundred eighty-seven patients and eighty-five deaths cannot support a rich model, and a cross-validated C-index of zero point five six is only a little above a coin flip. The clusters were chosen on the full sample and then tested on that same sample. Missingness itself is tied to survival — the baseline-missing log-rank p-value is zero point zero zero six — which is a warning about the data. This is one public cohort, analyzed once, with every predictor frozen on the day of transplant.

**On screen**

- Three takeaway lines, appearing one at a time:
  1. Age: HR 1.33 per 5 years (p = 0.024)
  2. Five clusters describe the cohort; they do not separate survival (log-rank p = 0.25)
  3. CV C-index 0.56 on 85 events
- A short limitations row: n = 187, clusters fit on all rows, DBSCAN ARI = 0, baseline-missing log-rank p = 0.006
- Closing line: “Predictors frozen at transplant. One public cohort.”
