import gradio as gr
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. DATA LOADING & MODEL TRAINING ON STARTUP
# ==========================================
print("Loading data and training models... Please wait.")

# Load and clean data
df = pd.read_csv('data.csv', encoding='unicode_escape')
df.dropna(subset=['CustomerID'], inplace=True)
df = df[df['Quantity'] > 0]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Generate RFM
snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()
rfm.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'}, inplace=True)
rfm = rfm[(rfm['Monetary'] > 0) & (rfm['Monetary'] < 50000)]

# Train K-Means (Segmentation)
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
kmeans = KMeans(n_clusters=3, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Map clusters to names dynamically
cluster_means = rfm.groupby('Cluster')['Monetary'].mean().sort_values()
segment_names = {
    cluster_means.index[0]: "⚠️ At-Risk / Churning",
    cluster_means.index[1]: "🛍️ Promising / Casual",
    cluster_means.index[2]: "🏆 Champions / VIPs"
}

# Train Random Forest (High Value Prediction)
threshold = rfm['Monetary'].quantile(0.75)
X = rfm[['Recency', 'Frequency']]
y = (rfm['Monetary'] >= threshold).astype(int)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

print("Models trained successfully! Launching UI...")

# ==========================================
# 2. GRADIO PREDICTION FUNCTION
# ==========================================
def analyze_customer(recency, frequency, monetary):
    """Takes user inputs from UI and returns ML predictions."""
    
    # K-Means Prediction (Scale inputs first)
    scaled_input = scaler.transform([[recency, frequency, monetary]])
    cluster_id = kmeans.predict(scaled_input)[0]
    segment = segment_names[cluster_id]
    
    # Random Forest Prediction (Uses only Recency & Frequency)
    rf_input = pd.DataFrame([[recency, frequency]], columns=['Recency', 'Frequency'])
    is_high_val = rf_model.predict(rf_input)[0]
    prob_high_val = rf_model.predict_proba(rf_input)[0][1] * 100
    
    value_status = "🌟 High-Value" if is_high_val == 1 else "👤 Standard"
    value_str = f"{value_status} ({prob_high_val:.1f}% Probability)"
    
    # Generate Business Recommendation
    if "VIPs" in segment:
        action = "Protect: Enroll in premium loyalty program. Do not discount; offer early access to new products."
    elif "Casual" in segment:
        action = "Grow: Send targeted cross-sell emails. Suggest items frequently bought together to increase basket size."
    else:
        action = "Win-Back: Send aggressive discount (e.g., 20% off) immediately to prevent permanent churn."
        
    return segment, value_str, action

# ==========================================
# 3. GRADIO USER INTERFACE (gr.Blocks)
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🛒 Customer Analytics Engine
        Enter a customer's behavioral metrics below. The system combines **K-Means Clustering** (Segmentation) 
        and **Random Forest** (Lifetime Value Prediction) to generate an instant marketing strategy.
        """
    )
    
    with gr.Row():
        # Left Column: Inputs
        with gr.Column():
            gr.Markdown("### 📥 Input Customer Data")
            recency_in = gr.Number(label="Recency (Days since last purchase)", value=15, precision=0)
            frequency_in = gr.Number(label="Frequency (Total number of purchases)", value=8, precision=0)
            monetary_in = gr.Number(label="Monetary (Total amount spent in £)", value=450)
            
            analyze_btn = gr.Button("🧠 Run ML Analysis", variant="primary")
            
        # Right Column: Outputs
        with gr.Column():
            gr.Markdown("### 📤 Machine Learning Insights")
            segment_out = gr.Textbox(label="1. K-Means Segment Profile")
            value_out = gr.Textbox(label="2. Random Forest Value Prediction")
            action_out = gr.Textbox(label="3. Recommended Marketing Action", lines=3)

    # Link the button to the function
    analyze_btn.click(
        fn=analyze_customer,
        inputs=[recency_in, frequency_in, monetary_in],
        outputs=[segment_out, value_out, action_out]
    )
    
    # Add a footer with example profiles
    gr.Examples(
        examples=[
            [5, 50, 3000],  # Example 1: Clear VIP
            [45, 3, 120],   # Example 2: Casual
            [250, 1, 15]    # Example 3: At-Risk
        ],
        inputs=[recency_in, frequency_in, monetary_in],
        label="Try these Sample Customer Profiles:"
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()