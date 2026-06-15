# 🛒 E-Commerce Customer Analytics & Prediction Dashboard

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![UI](https://img.shields.io/badge/UI-Gradio-green)

## 📌 Project Overview
This project is an end-to-end Machine Learning pipeline designed to help retail businesses maximize revenue through data-driven customer retention. 

It takes raw, messy e-commerce transaction data and transforms it into actionable marketing strategies by utilizing **Unsupervised Learning (Customer Segmentation)** and **Supervised Learning (Lifetime Value Prediction)**. The project includes an interactive web dashboard where stakeholders can input customer behavior metrics and instantly receive automated marketing actions.

## 🚀 Key Features
* **Data Engineering (RFM Analysis):** Aggregates raw transaction logs into behavioral profiles based on Recency (days since last order), Frequency (total orders), and Monetary value (total spend).
* **Customer Segmentation (K-Means Clustering):** Automatically groups customers into three distinct business personas: VIP Champions, Promising/Casual Buyers, and At-Risk Customers.
* **High-Value Prediction (Random Forest):** Predicts whether a customer will become a top-tier spender. Feature importance analysis revealed that *Purchase Frequency* is a significantly stronger driver of lifetime value than *Recency*.
* **Interactive UI:** A real-time web application built with Gradio/Streamlit that allows marketing teams to generate custom strategies without writing code.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (K-Means, RandomForestClassifier)
* **Visualizations:** Matplotlib, Seaborn, Plotly
* **Web Interface:** Gradio (or Streamlit)

## 📊 Business Insights Derived
1. **The 80/20 Rule is Real:** A Pareto analysis of the dataset revealed that the top 20% of customers generate the vast majority of total revenue.
2. **Frequency > Recency:** The Random Forest model proved that getting a customer to buy *multiple times* is a much stronger indicator of a "Whale" (high-value client) than simply having bought recently.
3. **Strategic Pivot:** Marketing budgets should shift from standard acquisition to aggressive retention of the VIP cluster via exclusive loyalty programs.

