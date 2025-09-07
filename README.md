# Forecasting India→US Export Loss & Recovery Under U.S. Tariffs

## What this project delivers

* A **baseline forecast** of FY2025 exports (no tariffs) by category using **XGBoost (XGBRegressor)**
* A **tariff shock** applied via **price elasticity of demand**
* A **lever stack** (rebates, cost cuts, demand uplift) that *absorbs* part of the shock and shows **how much loss is avoided**
* A **worked example for Textiles** with explicit \$ and % impacts

---

## 1) Data acquisition & preparation

**Data source (5 years):**

* India’s Ministry of Commerce / DGCIS (annual exports by broad category)

**Schema used**

| Field                        | Example                                     | Notes               |
| ---------------------------- | ------------------------------------------- | ------------------- |
| `Year`                       | 2020…2024                                   | Fiscal years        |
| `Export Category`            | Textile Products, Pharma, Engineering, etc. | 9 aggregate buckets |
| `Export Value (USD Million)` | 10,959                                      | Annual value        |

**Pre‑processing**

* Sort by `Export Category`, `Year`
* Create `year_index = Year - Year.min()`
* One‑hot encode `Export Category` for modeling
* (Optional, if monthly data) create **lags** and **rolling means** per category

---

## 2) Model selection & why XGBoost

**Why XGBoost / XGBRegressor**

* Handles **tabular** data with **non‑linearities**
* Works well with **sparse one‑hot** category features
* Fast to train; robust to moderate feature scaling issues
* Allows **explainability** (feature importance, SHAP)

**Model object**

```python
from xgboost import XGBRegressor
model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)
```

**Training setup**

* **Features:** `year_index` + one‑hot(`Export Category`)
* **Target:** `Export Value (USD Million)`
* **Train window:** 2020–2024
* **Predict:** 2025 baseline (**no tariff**)

---

## 3) The dynamics of XGBoost & XGBRegressor (short)

* XGBoost builds **additive trees** optimizing a **regularized objective** (squared error for regression)
* `learning_rate` shrinks each tree’s contribution; `max_depth` controls interaction order; `subsample` and `colsample_bytree` add randomness to reduce overfit
* **Feature importance**: gain/weight/cover or **SHAP** values to see which features drive forecast (typically `year_index` + category dummies)

---

## 4) Elasticity concept (what, why, how)

**Definition**

$`E = \frac{\%\Delta \text{Quantity}}{\%\Delta \text{Price}}`$

A **price increase** from tariffs **reduces quantity** (E is typically **negative**).

**How assigned in this project (assumptions guided by literature & market structure)**

| Export Category      | Elasticity (E) | Rationale (short)                       |
| -------------------- | -------------: | --------------------------------------- |
| Textile Products     |           −1.1 | Highly substitutable, price‑competitive |
| Pharma Products      |           −0.3 | Essential demand, regulated             |
| Gems & Jewellery     |           −1.0 | Discretionary, deferrable               |
| Engineering Products |           −1.2 | Cost‑driven B2B procurement             |
| Chemicals & Allied   |           −0.9 | Input market with alternatives          |
| Electronics / E\&SW  |           −0.8 | Competitive, partial brand pull         |
| Marine               |           −0.7 | Moderate substitutability               |
| Leather              |           −1.0 | Fashion/discretionary                   |
| Agricultural         |           −0.4 | Semi‑essential, partial substitutes     |

**How elasticity is measured in the real world (if you extend the study)**

* **Panel regression** on (log) quantity vs (log) price with fixed effects for product & market, controlling for income, FX, and policy dummies
* **Event studies** around policy/tariff changes in comparable markets
* Meta‑analyses from trade orgs (WITS/IMF/UNCTAD) to benchmark category ranges

---

## 5) From forecast to tariff shock

**Step 1:  Baseline (no tariff):**
Use XGBoost to predict 2025 **without** tariff.

**Step 2: Tariff‑only scenario:**
Tariff rate `T` (e.g., 50%) → if fully passed to price, **price change** = `+T`.
Quantity effect:

$$
\%\Delta Q = E \times \%\Delta P
$$

Value proxy ≈ `Baseline × (1 + E × T)` (for small changes) or combine with ASP assumptions if available.

<img width="1390" height="690" alt="comparison" src="https://github.com/user-attachments/assets/d0042619-206a-407c-90cd-58ba75cbb0e3" />


---

## 6) Absorption framework & levers

**Two different knobs**

* **`absorb`**: share of tariff **not passed** to buyer (margin squeeze).
  `net_tariff_to_price = T × (1 − absorb)`
* **`pass_to_price`**: share of **internal savings** passed **down** to the buyer (price relief).
  `price_offset_from_savings = pass_to_price × cost_savings_total`

**Effective price change (after levers)**

$`\Delta P_{\text{eff}} = \underbrace{T(1- \text{absorb})}_{\text{net tariff}} \;-\; \underbrace{\text{rebate\_rate}}_{\text{RoSCTL/RoDTEP}} \;-\; \underbrace{\text{pass\_to\_price}\times \text{cost\_savings\_total}}_{\text{shared savings}}`$

**Quantity change**

$`\Delta Q\% = E \times \Delta P_{\text{eff}}`$

**Demand‑side uplift** (loyalty, bundling, marketing, customization, service) applied multiplicatively to volume:
`(1 + demand_uplift)`.

**Quantified levers & benchmarks used**

* **Rebates:**

  * **RoSCTL (textiles)**: **6.05%–8.2%** (used **\~7%** in base)
  * **RoDTEP (many lines in chemicals/pharma/iron & steel)**: **\~0.5%–2%** (commonly **0.8%**)
* **Cost savings (COGS/logistics/procurement):**

  * **Vendor rationalization**: **3–5%**
  * **Logistics optimization / FTA routing**: **8–15%**
  * **Better commodity forecasting**: **5–10%**
  * In the model we **sum** these (e.g., **\~21%**) and **pass 40–60%** to price (e.g., **50%** → **10.5%** price relief)
* **Demand uplift:**

  * **Bundling + loyalty + targeted marketing + customization + service**: typically **10–20%** volume uplift in aggregate (we used **\~12%** in base)
  * Benchmarks: loyalty/retention studies (5% higher retention meaningfully improves profits), bundling elevates perceived value & repeat purchase

<img width="1189" height="590" alt="corrective measure" src="https://github.com/user-attachments/assets/f8c31485-510b-4eee-a5ae-3dea228269f4" />



---

## 7) Worked example:  **Textile Products**

**From the chart used in this project:**

* **Baseline FY2025 (no tariff):** **\$10,812M**
* **Tariff‑only projection:** **\$4,900M**

  * **Loss without levers:** **\$5,912M** (**−54.7%** vs baseline)

**Levers applied (quantified in model):**

* **Rebate (RoSCTL ≈ 7%)**: **+\$757M**
* **Cost‑side savings passed to price (≈5.25% of baseline)**: **+\$568M**
* **Demand uplift (bundling + loyalty + marketing + customization ≈ 8.9% of baseline)**: **+\$958M**

**Final projection after levers:** **\$7,183M**

* **Loss avoided (“business saved”) vs tariff‑only:** **\$2,283M**
* **% of tariff loss recovered:** **38.6%**
* **Final vs baseline:** **−33.6%** (i.e., at **66.4%** of baseline)

---


## 8) How to reproduce (quick code sketch)

```python
# A) Prepare features
df['year_index'] = df['Year'] - df['Year'].min()
X = pd.get_dummies(df[['year_index','Export Category']], drop_first=True)
y = df['Export Value (USD Million)']

# B) Train XGBRegressor
from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                     subsample=0.9, colsample_bytree=0.9, random_state=42)
model.fit(X, y)

# C) Predict 2025 baseline per category
# ...build X_2025 with year_index+1 and category dummies, predict -> Baseline

# D) Apply tariff shock & levers (Textiles)
T = 0.50; absorb = 0.20; rebate_rate = 0.07
cost_savings_total = 0.21; pass_to_price = 0.50
demand_uplift = 0.12; E = -1.1

net_tariff_to_price = T*(1-absorb)
price_relief = rebate_rate + pass_to_price*cost_savings_total
deltaP_eff = net_tariff_to_price - price_relief
qty_change = E * deltaP_eff

final_value = baseline_textiles * (1 + qty_change) * (1 + demand_uplift)
```

---

## 9) Sources & benchmarks (for lever quantification)

* **RoSCTL** rebate rates (apparel/made‑ups): **6.05–8.2%**
* **RoDTEP**: many lines in chemicals/pharma/iron & steel \~**0.5–2%**, with several items at **\~0.8%**
* **Logistics optimization** potential: **8–15%** cost reduction via routing, carrier mix, consolidation
* **Vendor rationalization**: **3–5%** COGS reduction via consolidation & renegotiations
* **Commodity forecasting**: **5–10%** COGS protection via timing and hedging improvements
* **Demand levers** (bundling/loyalty/marketing/customization/after‑sales): **\~10–20%** uplift in repeat purchase/volume; retention literature notes strong profit leverage when retention improves

*(Adjust ranges per sector.)*

---

## 10) Conclusion (Textile category)

* **Tariff‑only** would have cut FY2025 textile exports by **\$5.9B (−54.7%)** from the baseline.
* Applying the **quantified levers**, the model **recovers \$2.3B** of that loss, absorbing **\~39%** of the shock and landing at **\$7.18B** (i.e., **66.4%** of the baseline).
* With **stronger absorption (30%)**, **higher pass‑through of savings**, or **greater demand uplift** (e.g., targeted bundling/loyalty), the **recovery band** improves further—use the sensitivity page to present upside cases to stakeholders.

---
