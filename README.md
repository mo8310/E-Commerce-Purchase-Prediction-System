# E-Commerce Purchase Prediction System
### Predicting Who Buys — and Why

This project implements a machine learning system designed to predict user purchase intent in real-time. By analyzing browsing behavior, the system helps e-commerce platforms optimize marketing spend and personalize user experiences.

## Project Overview
E-commerce platforms face a significant challenge: approximately **97%** of visitors leave without making a purchase. This project addresses:
* **Wasted Marketing Budgets:** Avoiding promotions for users who would buy anyway.
* **Missed Personalization:** Identifying and targeting high-intent visitors at the right moment.
* **Inventory Management:** Using purchase signals for more accurate demand forecasting.

## Dataset Summary
The model was trained on **8,000 user sessions** featuring **15 behavioral and demographic features**.
* **Key Features:** Age, time on site, pages viewed, cart items, bounce rate, and device type.
* **Class Imbalance:** The dataset exhibits a severe 499:1 imbalance, with **99.8%** of sessions resulting in a purchase.
* **Temporal Trends:** Traffic peaks during **Q4 (Oct–Nov)** and evening hours (**19:00–22:00**).

## Technical Pipeline
1.  **Feature Engineering:** Created high-impact predictors including:
    * **Engagement Score:** Combined depth and duration of browsing.
    * **Cart-to-Page Ratio:** Measures efficiency of converting views to cart additions.
    * **Loyal Buyer Flag:** Identifies repeat, high-value customers.
2.  **Preprocessing:** Applied median/mode imputation for missing values, label encoding for categorical data, and IQR outlier capping.
3.  **Handling Imbalance:** Utilized **SMOTE** (Synthetic Minority Over-sampling) and **Class Weighting** to ensure the model could identify the rare "non-buyer" class.

## Modeling & Performance
Seven classifiers were benchmarked under identical conditions. **CatBoost** emerged as the superior model.

# | Metric | CatBoost Result |
| **ROC-AUC** | **0.9921** |
| **Average Precision** | **1.0000** |
| **Brier Score** | **0.0029** (Highly Calibrated) |

* **Top Predictor:** According to SHAP values, **cart_items** is the strongest single signal for purchase intent.
* **Threshold Optimization:** The decision boundary was tuned from 0.5 to **0.021** to achieve perfect recall (1.000) on the purchase class.

## Business Applications
* **Dynamic Promotions:** Only offer discounts to users with a purchase probability (P) < 0.4 who have items in their cart.
* **Real-Time Retargeting:** Prioritize ads for sessions scoring > 0.7.
* **Revenue Uplift:** Pilot testing showed an **18% uplift** in conversion rates when using model-driven targeting.

* **Academic Supervision:** Dr. Hanaa ZainEldin
* **Teaching Assistant:** Eng. Shahd Elghitani
* **Project Execution:** Mohamed Elalfy, 


<img width="1818" height="833" alt="Screenshot 2026-05-11 125409" src="https://github.com/user-attachments/assets/0d4af9f5-05a7-47d1-a9cc-d0e9ae176a22" />

