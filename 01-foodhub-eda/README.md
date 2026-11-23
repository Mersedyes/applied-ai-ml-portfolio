# 01 · FoodHub: Exploratory Analysis of Food-Delivery Orders

**Question.** What drives order volume, revenue, and delivery performance for a food-delivery aggregator?
**Data.** 1,898 orders from a New York City food-delivery platform: restaurant, cuisine, cost, day type, rating, preparation time, and delivery time. Course-provided; not redistributed.
**Tools.** Python, pandas, NumPy, matplotlib, seaborn.

## Approach
1. Checked structure, types, missing values, and summary statistics.
2. Ran univariate analysis (cost, ratings, preparation and delivery time) and bivariate analysis (cuisine × cost, day type × delivery time, rating × time).
3. Computed platform revenue under a tiered commission (25% on orders above $20, 15% on orders above $5).
4. Derived recommendations on promotions, partnerships, and delivery operations.

## Results
Recomputed from the raw data in November 2026; consistent with the notebook outputs.

| Measure | Value |
|---|---|
| Mean food preparation time | 27.4 min |
| Mean delivery time | 24.2 min |
| Mean delivery time, weekday vs. weekend | **28.3 min vs. 22.5 min** |
| Share of orders placed on weekends | 71.2% |
| Top cuisines by orders | American (584), Japanese (470), Italian (298) |
| Orders with no rating | 38.8% |
| Orders above $20 | 29.2% |
| Net platform revenue under the tiered commission | $6,166.30 |

## 2026 re-audit
- **Two slide-level findings were wrong and are corrected here.** My submitted presentation said weekend deliveries take longer and listed Mexican cuisine in the top three. The data shows the opposite on delivery time (weekday deliveries are about 6 minutes slower) and Italian third; Mexican is fifth (77 orders).
- **Lesson:** reported numbers should be produced from the same executed analysis, not transcribed by hand into slides. See [`docs/evaluation-audit.md`](../docs/evaluation-audit.md).
- **Analytical limit:** weekday vs. weekend is confounded with order volume and staffing, which this dataset cannot separate. The comparison is descriptive, not causal.

## Presentation
[`presentation/FoodHub_EDA_Presentation.pdf`](presentation/FoodHub_EDA_Presentation.pdf) is my submitted deck (32 slides), unedited. Its weekday/weekend delivery finding and top-three cuisine list are corrected above.

## Files
- [`foodhub_eda.ipynb`](foodhub_eda.ipynb): cleaned notebook (course instructions removed; code and outputs unchanged). This is the last saved `.ipynb` revision; minor final edits existed only in an exported PDF.
- [`presentation/FoodHub_EDA_Presentation.pdf`](presentation/FoodHub_EDA_Presentation.pdf): original 2025 presentation.
