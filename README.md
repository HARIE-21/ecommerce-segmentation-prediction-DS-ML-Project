# 🛒 Retail ML: Customer Segmentation & High-Value Buyer Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Gradio](https://img.shields.io/badge/Gradio-Web%20UI-ff69b4)
![Data Science](https://img.shields.io/badge/Data%20Science-Retail%20Analytics-brightgreen)

## 📌 Project Overview
This project is an end-to-end Machine Learning pipeline designed for e-commerce and retail businesses. It processes raw transactional data to understand customer purchasing behaviors, segments them into actionable business personas, and uses predictive modeling to identify high-value "Whale" customers. 

The project concludes with an interactive Web Dashboard built in Gradio (or Streamlit) that allows marketing teams to input customer metrics and instantly receive AI-driven marketing strategies.

## ✨ Key Features & Workflow
1. **Data Cleaning & Preprocessing:** Handled missing Customer IDs, removed cancelled/returned orders, and formatted datetime columns from a dataset of 500,000+ transactions.
2. **RFM Feature Engineering:** Transformed raw transactional rows into behavioral customer profiles based on **R**ecency (days since last order), **F**requency (total orders), and **M**onetary value (total spend).
3. **Unsupervised Learning (K-Means Clustering):** Segmented the customer base into 3 distinct business personas:
   * 🏆 **Champions / VIPs** (High Spend, High Frequency)
   * 🛍️ **Promising / Casuals** (Average Spend, Moderate Recency)
   * ⚠️ **At-Risk / Churning** (Low Spend, High Recency)
4. **Supervised Learning (Random Forest):** Trained an ensemble classifier to predict whether a customer will be a top-tier spender based strictly on their Recency and Frequency. 
5. **Business Intelligence & Strategy:** Applied the **Pareto Principle (80/20 Rule)** to prove that the top 20% of customers generate the vast majority of revenue, leading to data-backed marketing recommendations.
6. **Interactive Web UI:** A deployed dashboard where stakeholders can simulate customer behaviors and get instant ML predictions and strategic actions.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (K-Means, Random Forest, StandardScaler)
* **Data Visualization:** Matplotlib, Seaborn, Plotly
* **Web Framework:** Gradio / Streamlit

## 📊 Key Business Findings
* **Frequency Outweighs Recency:** Feature Importance extraction from the Random Forest model revealed that *how often* a customer buys is a significantly stronger indicator of lifetime value than *how recently* they bought.
* **The "Whale" Dependency:** The Pareto analysis confirmed that protecting the "Champions" segment via loyalty programs yields a much higher ROI than aggressive new customer acquisition.
* **Actionable Churn Prevention:** The clusters successfully isolated customers who haven't shopped in 60+ days, allowing for automated, targeted win-back discount campaigns.

## 🚀 How to Run the Project Locally

### 1. Clone the repository
```bash
git clone [https://github.com/yourusername/retail-ml-segmentation.git](https://github.com/yourusername/retail-ml-segmentation.git)
cd retail-ml-segmentation
