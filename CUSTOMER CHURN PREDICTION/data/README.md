# IBM Telco Customer Churn Dataset

## Source & Attribution
- **Repository**: [IBM Telco Customer Churn on Cloud Pak for Data](https://github.com/IBM/telco-customer-churn-on-icp4d)
- **Direct CSV URL**: `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
- **Original Source**: IBM Watson Analytics Sample Datasets / IBM Developer
- **License**: Apache License 2.0 (as licensed in the IBM repository)

## Dataset Description
This dataset contains 7,043 customer records for a telecommunications company. Each record represents a customer and includes demographic details, account tenure, subscribed services, contract arrangements, payment preferences, monthly/total charges, and whether the customer churned (`Yes` or `No`).

### Features Summary (21 Columns)
- **customerID**: Unique identifier for each customer.
- **gender**: Customer gender (`Male`, `Female`).
- **SeniorCitizen**: Indicator if the customer is a senior citizen (`1`, `0`).
- **Partner**: Whether the customer has a partner (`Yes`, `No`).
- **Dependents**: Whether the customer has dependents (`Yes`, `No`).
- **tenure**: Number of months the customer has stayed with the company.
- **PhoneService**: Whether the customer has a phone service (`Yes`, `No`).
- **MultipleLines**: Whether the customer has multiple lines (`Yes`, `No`, `No phone service`).
- **InternetService**: Internet service provider type (`DSL`, `Fiber optic`, `No`).
- **OnlineSecurity**: Whether the customer has online security (`Yes`, `No`, `No internet service`).
- **OnlineBackup**: Whether the customer has online backup (`Yes`, `No`, `No internet service`).
- **DeviceProtection**: Whether the customer has device tech protection (`Yes`, `No`, `No internet service`).
- **TechSupport**: Whether the customer has tech support (`Yes`, `No`, `No internet service`).
- **StreamingTV**: Whether the customer has streaming TV service (`Yes`, `No`, `No internet service`).
- **StreamingMovies**: Whether the customer has streaming movies service (`Yes`, `No`, `No internet service`).
- **Contract**: Contract duration (`Month-to-month`, `One year`, `Two year`).
- **PaperlessBilling**: Whether paperless billing is active (`Yes`, `No`).
- **PaymentMethod**: Payment mechanism (`Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)`).
- **MonthlyCharges**: Current monthly charge in USD (`float`).
- **TotalCharges**: Cumulative total charges in USD (`float`, contains blank strings for 11 customers with 0 tenure).
- **Churn**: Target variable indicating whether customer left within the last month (`Yes`, `No`).

## Re-downloading Instructions
To download this dataset locally into the `data/` folder, execute:
```bash
curl -s -L -o data/Telco-Customer-Churn.csv https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
```
