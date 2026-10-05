# AR Collections Forecasting & Risk Scoring

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-gradient_boosting-green)
![SHAP](https://img.shields.io/badge/SHAP-explainability-orange)
![SQLite](https://img.shields.io/badge/SQL-SQLite-lightgrey)

> A collections team doesn't have a collections problem — it has a collections **timing**
> problem. This project predicts *which* invoices will pay late, *when* the cash arrives,
> and *who* to chase first, then proves the predictions can be trusted in rupees, not just statistics.

---

## 1. The business problem

Every B2B company sits on receivables — money customers owe. Two questions decide how well
finance manages it:

1. **Of the cash outstanding today, how much will arrive late?** (forecasting)
2. **Which customers should collectors call first?** (prioritization)

Answering with gut feel or segment averages leaves money on the table: collectors chase the
wrong accounts, cash forecasts miss, and working-capital planning suffers. This project builds
the full stack — invoice-level risk models, calibrated probabilities, a money-denominated
risk score, and the SQL reporting layer finance already uses — on a 10,833-invoice ledger
with a built-in regime shock to prove the validation is real.

---

## 2. The dataset

A synthetic B2B receivables ledger, Jan 2024 – Dec 2025. Synthetic, but built with the
three pathologies that make real AR data hard:

| Property | Value | Why it matters |
|---|---|---|
| Invoices | 10,833 across 250 customers | invoice-grain modeling |
| Segments | enterprise / mid-market / SMB × 4 regions | heterogeneity |
| Paid (outcome known) | 9,912 | training signal |
| **Censored** (unpaid at cutoff) | **921 (8.5%)** | outcome *unknown* — the trap |
| Late rate (paid) | 62.3% | the base rate |
| Outstanding at cutoff | **₹32,123.7 lakh** | the money at stake |
| **Regime break, Apr 2025** | late rate 56.2% → 75.4%; DSO ~55 → ~75 days | the stress test |

**The censoring trap.** 66% of Oct–Dec 2025 invoices are still unpaid at the cutoff.
Drop them and the observed late rate is 58.7% — the book looks healthier than it is,
because the slowest payers are exactly the ones still censored. Every step of this project
treats censored invoices as *unknown*, never as *on time*. Step 10 shows what happens when
you forget.

---

## 3. Methodology — twelve steps, each one earning its keep

### Step 1 · Data quality audit
Before modeling: duplicate keys (0), due-before-issue dates (0), paid-before-issue (0),
negative amounts (0), terms-vs-due-date agreement (100%). The ledger is internally
consistent — stated, then verified, not assumed.

### Step 2 · Exploratory analysis, basics → advanced
Basics: shapes, the 921 missing payment dates (missingness *with meaning*), late-rate
62.3%, aging buckets, amount/terms distributions. Advanced: **mature-invoice collection
curves** by segment (enterprise: 1.4% paid within 30 days vs SMB 36.5% — the curves a
treasury team actually uses), DSO trend, **vintage analysis** pinning the April 2025 break,
seasonality, customer-level Pareto (top 10 customers = 23% of outstanding), and the
censoring-bias demonstration.

### Step 3 · Feature engineering with a leakage firewall
25 features, every one knowable **at invoice issue time**:
- *Invoice-level:* log amount, terms, calendar encodings, segment/region one-hots
- *Customer history (priors only):* prior late rate, avg days late, avg days-to-pay,
  total billed, days since last invoice — computed strictly from earlier invoices
- *Exposure at issue:* outstanding amount on the day the invoice was raised (vectorized
  mask, no row loops)

Censored priors are excluded from late-rate denominators but counted toward exposure.
A runtime assertion guarantees no outcome column leaks into the feature set.
Signal check: worst prior-late-rate quartile goes late **73%** vs **51%** for the best.

### Step 4 · Target definitions
- **Risk band** (classification → collections action): 0 on-time (37.7%) → no action,
  1 mild 1–30d (48.2%) → reminder, 2 serious 31–60d (12.4%) → escalation,
  3 severe 61+d (1.7%) → collections call
- **Days-to-pay** (regression): when the cash arrives

The 921 censored invoices get **no target** (`usable_for_training = 0`) — kept for
cash-at-risk scoring, never for training, never mislabelled.

### Step 5 · Baselines — what the business does today
Majority-class guessing and **segment rules** (each segment's historical mode band, mean
days-to-pay, P(late)) — roughly tribal knowledge formalized. Bars to beat on the
post-downturn test window: band accuracy 0.532 / macro F1 **0.174**, binary ROC-AUC
**0.524**, days-to-pay MAE **16.05**.

### Step 6 · ML models
Three LightGBM models on a strict time split (train ≤ Apr 2025, validate May–Jun,
test Jul–Dec):

| Task | Baseline | Model | Result |
|---|---|---|---|
| Risk band | acc 0.532 / macro F1 0.174 | LGBM multiclass, `class_weight='balanced'` | **macro F1 0.29**, serious-band recall **0.63** |
| Late / on-time | ROC-AUC 0.524 | LGBM binary | **ROC-AUC 0.66**, PR-AUC **0.83** |
| Days-to-pay | MAE 16.05 | LGBM regression | **MAE 15.43** |

The honest headline: multiclass *accuracy* falls (0.36 vs 0.53) because the model stops
always guessing the majority band — macro F1 nearly doubles. Severe-band recall stays
weak (0.09, n=43): documented, not hidden.

**Model comparison, kept honest:** HistGradientBoosting (0.63) and a regularized
Logistic Regression (**0.69**) were also tried — the linear model ranks best, because the
signal is mostly monotone (worse history → worse outcome). The pipeline continues with
LightGBM for the multiclass bands, nonlinear interactions, and exact SHAP values — but
the comparison stays in the record.

### Step 7 · Threshold tuning — where do we draw the collections line?
Precision-recall vs threshold: best F1 at **0.38**. At 0.5: precision 0.77, recall 0.90;
at 0.7: precision 0.82, recall 0.55. The threshold is a business choice about how many
collectors you have, not a statistical default.

### Step 8 · Calibration — making 70% mean 70%
Step 9 turns P(late) into money, so probabilities must be *true*, not just well-ranked.
**Platt scaling** fit on validation, evaluated on untouched test data: ECE **0.066 → 0.033**
(halved), Brier 0.191 → 0.187. Isotonic regression was tried first and **rejected** — the
934-invoice validation window made it overconfident at the edges (worse log-loss).
Calibration holds within each segment, not just overall.

### Step 9 · Cash at risk — the money step
- **Money-calibration check:** Σ P(late)×amount = **₹37,653.6L** vs actual late cash
  **₹37,653.7L** — ratio ≈ 1.00. The probabilities are trustworthy *in rupees*.
- **The live book:** all 921 outstanding invoices scored → **₹26,587L of ₹32,124L
  (82.8%) expected late**. Enterprise carries ₹19,110L. Top 10 customers hold **23.8%**
  of at-risk cash — C0052 alone: ₹910L across 17 invoices.
- **Capture curve:** chasing the riskiest 20% of invoices captures **45%** of late cash.

`data/ar_cash_at_risk.csv` is the collections priority list.

### Step 10 · Walk-forward retro testing
Simulated production: each month Apr–Dec 2025, train only on earlier invoices, score the
month. Mean ROC-AUC **0.67** — the headline wasn't a lucky split. The April fold is the
stress test: trained on pre-downturn data only, the model still discriminated (AUC 0.67)
but **underestimated late cash** (money ratio 0.73) — the failure mode a monitoring
dashboard must catch, and the reason for monthly retraining. The November fold (ratio 2.99)
is *evaluation bias, not model failure*: only fast payers had resolved by the cutoff.

### Step 11 · SHAP explainability
Global (`n_prior` dominates — long-history customers are the most predictable), directional
(beeswarm), per-invoice waterfall, and dependence plots. The riskiest test invoice —
INV001682, enterprise, ₹57.7L, P(late) = 1.000, actually late — decomposes to: 65 prior
invoices, 26.8 avg days late historically, 86.6 avg days-to-pay; the one factor pulling
risk *down* is 58-day terms. That's the explanation a collections analyst acts on.

### Step 12 · SQL layer
`data/ar_collections.db` (SQLite) + `sql/ar_aging_dso.sql`: 8 business queries — aging
buckets at cutoff, **DSO trend** (receivables/sales × 30 per month-end), outstanding by
segment, customer concentration, late rate by segment, dunning analysis,
collection-effectiveness index, region × terms. The DSO query alone tells the downturn
story (~55 → **75 days**) with no ML. Documented honesty note: `dunning_count`
separates late/on-time *perfectly* in this synthetic data — a post-outcome variable,
excluded from ML features, reporting-only.

---

## 4. Key findings

1. **The downturn is the story.** April 2025: late rates 56% → 75%, DSO 55 → 75 days.
   Every validation step is built around surviving it.
2. **Censoring is the trap.** Naive analysis understates risk by 4+ points and once
   produced a 3× money-calibration error in retro testing.
3. **Probabilities you can put rupees on.** Expected vs actual late cash matches at
   ratio ≈ 1.00 — then ₹26,587L of outstanding is flagged expected-late with a
   customer-level priority list.
4. **History predicts.** Prior late-rate quartiles separate future late rates 51% → 73%.
5. **Concentration is the action.** Top 10 customers = 23.8% of at-risk cash.
   Collections doesn't need to boil the ocean.
6. **Simple can win.** A regularized linear model outranked the gradient boosters —
   reported, not buried.

## 5. Recommendations

1. Work the `ar_cash_at_risk` priority list top-down — start with C0052 (₹910L, 17 invoices).
2. Retrain monthly; monitor the money-calibration ratio (April showed 0.73 pre-retrain).
3. Review terms policy: longer terms correlate with lower late rates in the north region
   (71% late at net-30 vs 56% at net-46+).
4. Tighten dunning for customers crossing 30 days with high prior late rates.

## 6. Limitations — read before citing

- **Synthetic data.** Patterns are realistic but the DGP is known; `dunning_count` is
  unrealistically deterministic. Real AR data is messier.
- **Severe band is rare** (1.7%, n=172 train). Recall 0.09 — the model won't reliably
  catch the worst cases.
- **Single regime break.** One synthetic shock; real regime changes vary.
- **Censored invoices excluded from training** biases the training set toward faster
  payers — acknowledged, partially offset by walk-forward evaluation.
- **No partial payments** — invoices are all-or-nothing; real collections aren't.


```

## 8. Project structure

```
ar-collections-forecasting/
├── README.md
├── requirements.txt
├── ar_collections_analysis.ipynb   # whole pipeline, executed (start here)
├── ar_step1_dataset.py … ar_step11_sql.py
├── data/       # datasets + ar_collections.db
├── visuals/    # 11 charts
└── sql/        # ar_aging_dso.sql — 8 standalone business queries
```

## 9. Tech stack

Python · pandas · LightGBM · scikit-learn · SHAP · matplotlib · SQLite

## 10. Resume block — October 2026

- Built invoice-level late-payment forecasting on 10.8k-invoice AR ledger: LightGBM classifier (ROC-AUC 0.66, PR-AUC 0.83) and days-to-pay regressor (MAE 15.4 days), beating segment-rule baselines; kept an honest model comparison where regularized LogReg ranked best (0.69)
- Calibrated probabilities with Platt scaling (ECE halved to 0.033); money-calibration check matched expected vs actual late cash at ratio 1.00, then flagged ₹26,587L of ₹32,124L outstanding as expected-late with a customer priority list
- Validated with walk-forward retro testing across the April 2025 downturn (mean AUC 0.67); documented censoring bias that made naive late-rate estimates understate risk
- Added SQL layer (SQLite): AR aging buckets, DSO trend, collection-effectiveness queries; explained predictions with SHAP global + per-invoice waterfall
