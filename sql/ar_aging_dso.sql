-- ================= q1_aging_buckets =================

-- Q1. AR aging at cutoff: how overdue is the outstanding book?
SELECT
  CASE WHEN dpd <= 0 THEN 'not yet due'
       WHEN dpd <= 30 THEN '1-30 days'
       WHEN dpd <= 60 THEN '31-60 days'
       WHEN dpd <= 90 THEN '61-90 days'
       ELSE '90+ days' END AS bucket,
  COUNT(*) AS n_invoices,
  ROUND(SUM(amount_lakh), 1) AS outstanding_lakh,
  ROUND(100.0 * SUM(amount_lakh) / SUM(SUM(amount_lakh)) OVER (), 1) AS pct_of_book
FROM (SELECT amount_lakh,
             CAST(julianday('2025-12-31') - julianday(due_date) AS INT) AS dpd
      FROM invoices WHERE paid_date IS NULL)
GROUP BY bucket
ORDER BY MIN(dpd);

-- ================= q2_dso_trend =================

-- Q2. DSO trend: receivables / credit sales x 30, per month-end
WITH months(m) AS (
  SELECT DISTINCT substr(invoice_date, 1, 7) FROM invoices
)
SELECT m AS month,
  ROUND(SUM(CASE WHEN invoice_date <= date(m||'-01','+1 month','-1 day')
                 AND (paid_date IS NULL OR paid_date > date(m||'-01','+1 month','-1 day'))
                THEN amount_lakh ELSE 0 END), 1) AS receivables_lakh,
  ROUND(SUM(CASE WHEN substr(invoice_date,1,7)=m THEN amount_lakh ELSE 0 END), 1) AS sales_lakh,
  ROUND(SUM(CASE WHEN invoice_date <= date(m||'-01','+1 month','-1 day')
                 AND (paid_date IS NULL OR paid_date > date(m||'-01','+1 month','-1 day'))
                THEN amount_lakh ELSE 0 END)
        / NULLIF(SUM(CASE WHEN substr(invoice_date,1,7)=m THEN amount_lakh ELSE 0 END),0) * 30, 1) AS dso_days
FROM months, invoices GROUP BY m ORDER BY m;

-- ================= q3_outstanding_by_segment =================

-- Q3. Outstanding by segment at cutoff
SELECT segment, COUNT(*) AS n_invoices,
       ROUND(SUM(amount_lakh), 1) AS outstanding_lakh,
       ROUND(AVG(julianday('2025-12-31') - julianday(due_date)), 0) AS avg_days_past_due
FROM invoices WHERE paid_date IS NULL
GROUP BY segment ORDER BY outstanding_lakh DESC;

-- ================= q4_top_customers_outstanding =================

-- Q4. Top 10 customers by outstanding (concentration risk)
SELECT customer_id, COUNT(*) AS n_open,
       ROUND(SUM(amount_lakh), 1) AS outstanding_lakh
FROM invoices WHERE paid_date IS NULL
GROUP BY customer_id ORDER BY outstanding_lakh DESC LIMIT 10;

-- ================= q5_late_rate_by_segment =================

-- Q5. Late-payment rate by segment (resolved invoices only)
SELECT segment,
       COUNT(*) AS n_resolved,
       ROUND(100.0 * AVG(days_late > 0), 1) AS late_pct,
       ROUND(AVG(days_to_pay), 1) AS avg_days_to_pay,
       ROUND(AVG(CASE WHEN days_late > 0 THEN days_late END), 1) AS avg_days_late_when_late
FROM invoices WHERE paid_date IS NOT NULL
GROUP BY segment;

-- ================= q6_dunning_effectiveness =================

-- Q6. Does dunning intensity track lateness? (resolved invoices)
SELECT CASE WHEN dunning_count = 0 THEN '0 touches'
            WHEN dunning_count <= 2 THEN '1-2 touches'
            ELSE '3+ touches' END AS dunning_band,
       COUNT(*) AS n,
       ROUND(100.0 * AVG(days_late > 0), 1) AS late_pct,
       ROUND(AVG(days_late), 1) AS avg_days_late
FROM invoices WHERE paid_date IS NOT NULL
GROUP BY dunning_band ORDER BY dunning_count;

-- ================= q7_collection_effectiveness_index =================

-- Q7. Collection Effectiveness Index proxy: % of resolved cash collected
-- within terms, by quarter
SELECT substr(invoice_date, 1, 7) AS ym,
       ROUND(100.0 * SUM(CASE WHEN days_late <= 0 THEN amount_lakh ELSE 0 END)
             / SUM(amount_lakh), 1) AS pct_cash_on_time,
       ROUND(SUM(amount_lakh), 1) AS resolved_cash_lakh
FROM invoices WHERE paid_date IS NOT NULL
GROUP BY ym ORDER BY ym;

-- ================= q8_worst_regions_terms =================

-- Q8. Where do long terms meet slow payment? region x terms band
SELECT region,
       CASE WHEN terms_days <= 30 THEN 'net<=30'
            WHEN terms_days <= 45 THEN 'net 31-45'
            ELSE 'net 46+' END AS terms_band,
       COUNT(*) AS n,
       ROUND(100.0 * AVG(days_late > 0), 1) AS late_pct
FROM invoices WHERE paid_date IS NOT NULL
GROUP BY region, terms_band ORDER BY region, terms_days;

