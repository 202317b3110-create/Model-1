import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Page Configuration
st.set_page_config(page_title="Real Estate 'What-If' Analyzer", page_icon="🏠", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    h1, h2, h3, h4 {
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    .stMetric {
        background: rgba(30, 41, 59, 0.7);
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
    }
    .stMetric label {
        color: #94a3b8 !important;
        font-size: 1.1rem;
    }
    .stSidebar {
        background-color: #111827 !important;
    }
    div[data-testid="stSidebarNav"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# City to State Mapping Dictionary
CITY_TO_STATE = {
    "Hyderabad": "Telangana", "Vijayawada": "Andhra Pradesh", "Madurai": "Tamil Nadu",
    "Chennai": "Tamil Nadu", "Port Blair": "Andaman & Nicobar", "Gangtok": "Sikkim",
    "Aizawl": "Mizoram", "Itanagar": "Arunachal Pradesh", "Shillong": "Meghalaya",
    "Puducherry": "Puducherry (UT)", "DNH & DD": "DNH & DD (UT)", "Ladakh": "Ladakh (UT)",
    "Lakshadweep": "Lakshadweep (UT)", "Dehradun": "Uttarakhand", "Srinagar": "Jammu & Kashmir",
    "Panaji": "Goa", "Agartala": "Tripura", "Kohima": "Nagaland", "Imphal": "Manipur",
    "Patna": "Bihar", "Bhubaneswar": "Odisha", "Ranchi": "Jharkhand", "Raipur": "Chhattisgarh",
    "Dispur": "Assam", "Shimla": "Himachal Pradesh", "Mumbai": "Maharashtra", "New Delhi": "Delhi (NCR)",
    "Bengaluru": "Karnataka", "Gurugram": "Haryana", "Lucknow": "Uttar Pradesh", "Chandigarh": "Chandigarh (UT)",
    "Jaipur": "Rajasthan", "Amritsar": "Punjab", "Kolkata": "West Bengal", "Kochi": "Kerala",
    "Ahmedabad": "Gujarat", "Indore": "Madhya Pradesh"
}

# -----------------------------------------------------------------------------
# Data Loading and Model Training
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("City_Data.csv")
    
    # Helper to clean currency strings into floats
    def clean_curr(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            cleaned = x.replace('₹', '').replace(',', '').strip()
            try:
                return float(cleaned)
            except ValueError:
                return np.nan
        return float(x)
        
    df['Estimated Sale Price (INR)'] = df['Estimated Sale Price (INR)'].apply(clean_curr)
    
    # Map City to State
    df['State'] = df['City'].map(CITY_TO_STATE).fillna("Other")
    
    # Select features for our regression model
    features = [
        'Built-up Area (sqft)', 
        'Bedrooms (BHK)', 
        'Bathrooms', 
        'Property Age (Years)', 
        'Distance to Metro (km)', 
        'Distance to IT Hub (km)'
    ]
    
    # Ensure all features are numeric and drop missing values for model training
    for f in features:
        df[f] = pd.to_numeric(df[f], errors='coerce')
        
    df = df.dropna(subset=features + ['Estimated Sale Price (INR)'])
    
    return df, features

@st.cache_resource
def train_model(df, features):
    X = df[features]
    y = df['Estimated Sale Price (INR)']
    
    model = LinearRegression()
    model.fit(X, y)
    
    preds = model.predict(X)
    r2 = r2_score(y, preds)
    mae = mean_absolute_error(y, preds)
    
    return model, r2, mae

# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    st.markdown("<h1>🏠 Real Estate 'What-If' Market Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem;'>An interactive tool to analyze Indian housing trends and estimate property values based on local features using a Machine Learning model.</p>", unsafe_allow_html=True)

    # Load Data & Train Model
    try:
        df, feature_names = load_data()
        if df.empty:
            st.error("No valid data found after cleaning. Please check your City_Data.csv file.")
            st.stop()
        model, r2, mae = train_model(df, feature_names)
    except Exception as e:
        st.error(f"Failed to load data or train the model: {e}")
        st.stop()

    # -------------------------------------------------------------------------
    # Sidebar - Location Workflow
    # -------------------------------------------------------------------------
    st.sidebar.markdown("## 📍 Location Workflow")
    st.sidebar.selectbox("Country", ["India"], disabled=True)
    
    # Filter flow: State -> City -> Locality
    states = sorted(df['State'].unique())
    selected_state = st.sidebar.selectbox("State", states)
    
    cities = sorted(df[df['State'] == selected_state]['City'].unique())
    selected_city = st.sidebar.selectbox("City", cities)
    
    localities = sorted(df[df['City'] == selected_city]['Locality'].dropna().unique())
    selected_locality = st.sidebar.selectbox("Locality", localities) if len(localities) > 0 else "All"
    
    # Filter data for the selected locality to set the default slider values
    local_df = df[(df['City'] == selected_city) & (df['Locality'] == selected_locality)]
    if local_df.empty:
        # Fallback to city defaults if locality has no valid rows after cleaning
        local_df = df[df['City'] == selected_city]

    def get_default(feature_name):
        return float(local_df[feature_name].median()) if not local_df.empty else float(df[feature_name].median())

    st.sidebar.markdown("---")
    
    # -------------------------------------------------------------------------
    # Sidebar - "What-If" Calculator
    # -------------------------------------------------------------------------
    st.sidebar.markdown("## 🎛️ 'What-If' Calculator")
    loc_display = f"{selected_locality}, {selected_city}" if selected_locality != "All" else selected_city
    st.sidebar.markdown(f"Adjust features for properties in **{loc_display}**.")
    
    # Sliders for features, dynamically defaulting to the local medians
    area = st.sidebar.slider("Built-up Area (sqft)", float(df['Built-up Area (sqft)'].min()), float(df['Built-up Area (sqft)'].max()), get_default('Built-up Area (sqft)'), step=50.0)
    beds = st.sidebar.slider("Bedrooms (BHK)", float(df['Bedrooms (BHK)'].min()), float(df['Bedrooms (BHK)'].max()), get_default('Bedrooms (BHK)'), step=0.5)
    baths = st.sidebar.slider("Bathrooms", float(df['Bathrooms'].min()), float(df['Bathrooms'].max()), get_default('Bathrooms'), step=1.0)
    age = st.sidebar.slider("Property Age (Years)", float(df['Property Age (Years)'].min()), float(df['Property Age (Years)'].max()), get_default('Property Age (Years)'), step=1.0)
    metro = st.sidebar.slider("Distance to Metro (km)", float(df['Distance to Metro (km)'].min()), float(df['Distance to Metro (km)'].max()), get_default('Distance to Metro (km)'), step=0.5)
    ithub = st.sidebar.slider("Distance to IT Hub (km)", float(df['Distance to IT Hub (km)'].min()), float(df['Distance to IT Hub (km)'].max()), get_default('Distance to IT Hub (km)'), step=0.5)
    
    # Prepare input for prediction
    input_data = pd.DataFrame({
        'Built-up Area (sqft)': [area],
        'Bedrooms (BHK)': [beds],
        'Bathrooms': [baths],
        'Property Age (Years)': [age],
        'Distance to Metro (km)': [metro],
        'Distance to IT Hub (km)': [ithub]
    })
    
    # Prediction
    predicted_price = model.predict(input_data)[0]
    predicted_price = max(0, predicted_price)
    
    # -------------------------------------------------------------------------
    # Main Dashboard
    # -------------------------------------------------------------------------
    
    # Top Row: Prediction Gauge & Metrics
    col1, col2 = st.columns([1.5, 1])
    
    max_price = df['Estimated Sale Price (INR)'].quantile(0.95)
    
    with col1:
        st.markdown("### 🎯 Estimated Property Value")
        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = predicted_price,
            number = {'prefix': "₹", 'valueformat': ",.0f"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Predicted Price (INR)", 'font': {'size': 20, 'color': '#cbd5e1'}},
            gauge = {
                'axis': {'range': [None, max(max_price, predicted_price * 1.2)], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#3b82f6"},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, max_price * 0.33], 'color': "rgba(59, 130, 246, 0.2)"},
                    {'range': [max_price * 0.33, max_price * 0.66], 'color': "rgba(59, 130, 246, 0.4)"},
                    {'range': [max_price * 0.66, max(max_price, predicted_price * 1.2)], 'color': "rgba(59, 130, 246, 0.6)"}
                ],
                'threshold': {
                    'line': {'color': "#ef4444", 'width': 4},
                    'thickness': 0.75,
                    'value': predicted_price
                }
            }
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=350, margin=dict(t=50, b=0, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
    with col2:
        st.markdown("### 🤖 Model Performance")
        st.info("The model is trained dynamically on the entire dataset using Multiple Linear Regression to ensure robust predictions.")
        
        st.metric(label="R² Score (Accuracy)", value=f"{r2:.2f}", help="Indicates how well the features explain the variance in price.")
        st.metric(label="Mean Absolute Error", value=f"₹{mae:,.0f}", delta_color="inverse", help="Average absolute difference between predicted and actual prices.")
        
    st.markdown("---")
    
    # Visual Analytics Row
    st.markdown("### 📊 Market Visual Analytics")
    
    v_col1, v_col2 = st.columns(2)
    
    with v_col1:
        st.markdown("#### Feature vs Price Relationship")
        feature_to_plot = st.selectbox("Select a Feature to compare against Price", feature_names, index=0)
        
        fig_scatter = px.scatter(df, x=feature_to_plot, y="Estimated Sale Price (INR)", 
                                 title=f"Price vs {feature_to_plot}",
                                 opacity=0.4,
                                 color="Estimated Sale Price (INR)",
                                 color_continuous_scale="Blues",
                                 template="plotly_dark")
        
        fig_scatter.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with v_col2:
        st.markdown(f"#### Median Prices in {selected_state}")
        # Group by City to find median prices for the selected state
        state_df = df[df['State'] == selected_state]
        if not state_df.empty:
            city_prices = state_df.groupby('City')['Estimated Sale Price (INR)'].median().reset_index()
            city_prices = city_prices.sort_values(by='Estimated Sale Price (INR)', ascending=False).head(10)
            
            fig_bar = px.bar(city_prices, x='City', y='Estimated Sale Price (INR)', 
                             title=f"City Comparison in {selected_state}",
                             color='Estimated Sale Price (INR)',
                             color_continuous_scale="Viridis",
                             template="plotly_dark")
            fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("Insufficient data for Bar Chart.")

if __name__ == "__main__":
    main()