# Exploratory Data Analysis (EDA) Report: Telco Customer Churn

## 1. Executive Summary & Dataset Dimensions
- **Total Records**: 7,043 customer profiles
- **Total Attributes**: 21 columns (1 target variable, 20 customer features)
- **Duplicate Records**: 0
- **Missing / Blank Values**: 11 records in `TotalCharges` contained whitespace strings `' '`. These represent newly onboarded customers with `tenure = 0` who have not yet reached their first billing cycle. In the cleaned dataset, these have been imputed with `0.0` to preserve mathematical integrity and avoid row deletion.
- **Overall Churn Rate**: **26.54%** (1,869 churned vs. 5,174 retained customers).
- **Class Distribution**: Moderate class imbalance with approximately a 73:27 ratio between retained and churned cohorts.

---

## 2. Key Findings & Behavioral Drivers

### A. Contract Duration is the Single Strongest Churn Driver
- **Month-to-month contracts**: Churn rate is **42.7%**.
- **One-year contracts**: Churn rate drops drastically to **11.3%**.
- **Two-year contracts**: Churn rate drops to an ultra-low **2.8%**.
- *Insight*: Month-to-month customers exhibit nearly 15 times the churn propensity of two-year contract holders. Long-term contractual commitments establish strong retention.

### B. Customer Tenure & The "Onboarding Window"
- **Average Tenure (Retained Customers)**: **37.6 months**
- **Average Tenure (Churned Customers)**: **18.0 months**
- *Insight*: Churn risk is heavily concentrated within the first 1 to 6 months of customer tenure. Customers who survive the initial 12-month period show high stability and long-term loyalty.

### C. Monthly Charges & Cost Sensitivity
- **Average Monthly Charge (Retained)**: **$61.27**
- **Average Monthly Charge (Churned)**: **$74.44**
- *Insight*: Churned customers pay on average nearly $13 more per month. Customers experiencing higher billing burdens without high perceived value are far more susceptible to competitor offerings.

### D. Internet Service Technology Impact
- **Fiber optic customers**: Churn rate is **41.9%**.
- **DSL customers**: Churn rate is **19.0%**.
- **No Internet service**: Churn rate is only **7.4%**.
- *Insight*: Fiber optic customers pay top-tier rates but show the highest churn rate across tech types, pointing to either service reliability friction or steep price pressure.

### E. Payment Method Patterns
- **Electronic Check**: Highest churn rate at **45.3%**.
- **Mailed Check**: **19.1%**.
- **Bank Transfer (Automatic)**: **16.7%**.
- **Credit Card (Automatic)**: **15.2%**.
- *Insight*: Automated recurring payments (Credit Card and Bank Transfer) cut churn in half relative to manual electronic checks.

---

## 3. Generated Visualizations
All charts have been exported to `reports/figures/`:
1. `eda_churn_distribution.png`: Target class distribution.
2. `eda_churn_by_contract.png`: Churn percentages stratified by contract duration.
3. `eda_tenure_distribution.png`: Kernel density estimation (KDE) comparing tenure distributions.
4. `eda_charges_distribution.png`: Boxplot comparison of monthly charges across churn groups.
5. `eda_internet_service.png`: Churn rates across internet service infrastructure.
6. `eda_payment_method.png`: Churn rates sorted across payment methods.
7. `eda_correlation_matrix.png`: Pearson correlation matrix among numerical features.
