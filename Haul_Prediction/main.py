import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle 
import joblib
from sklearn.metrics import mean_absolute_error,r2_score

st.set_page_config(
    page_title="Predictive Maintenance-Haul Machines",
    page_icon= "🚂",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp{
       background:#0e1117;
       color:#ffffff;
    }

    h1,h2,h3{
    font-weight:700;
    }

      h1{
    animation: fadeInDown 1.2s ease-in-out;
    }
    @keyframes fadeInDown{
    from{ opacity:0; transform:translateY(-20px);}
    to{opacity:1;transform:translateY(0);}
    }

    .card{
       background:#161b22;
       border-radius:16px;
       padding:20px;
       border:1px solid #30363d;
       margin-bottom:20px;
    }


    div[data-testid="metric-container"]{
        background:#0e4429;
        border-radius:14px;
        padding:15px;
        border:1px solid #2ea043;
    }

    
     .stButton > button{
    background:linear-gradient(90deg,#00c6ff,#0072ff);
    color:white;
    border-radius:12px;
    border:none;
    padding:10px,20px;
    transition: all 0.3s ease;
    }

    .stButton > button:hover{
    transform: scale(1.05);
    box-shadow: 0 6px 20px rgba(0,114,255,0.6)
    }

    section[data-testid="stSidebar"]{
      background-color:#010409;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_model():
    with open("Linear_regression_model.pkl","rb") as f:
        return pickle.load(f)
    model = load_model()    
model = joblib.load("Linear_regression_model.pkl")

def load_default_data():
    return pd.read_csv("Haul_Machines.csv")

st.sidebar.title("APP CONTROLS🔑")

uploaded_file = st.sidebar.file_uploader("Upload Haul Machine Dataset",type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom Dataset Loaded")
else:
    df = load_default_data()
    st.sidebar.info("Using default dataset")

if "Machine_ID" not in df.columns:
    st.error("Dataset must contain a 'Machine_ID' colums")
    st.stop()

st.title("Predictive Maintenance Dashboard")
st.markdown("Estimate Next Failure Hours Using Operational Data")   

st.markdown("<div class='card'>",unsafe_allow_html=True)
st.subheader("🔍Machine Filter")

machine_id = st.selectbox(
    "Select Machine id",df["Machine_ID"].unique()
)
filtered_df = df[df["Machine_ID"] == machine_id]
st.markdown("</div>",unsafe_allow_html=True)

st.markdown("<div class='card'>",unsafe_allow_html=True)
st.subheader("📊 Fleet Snapshot")

c1,c2,c3 = st.columns(3)

c1.metric("Machines",len(filtered_df))
c2.metric("Avg Hours",int(filtered_df["Hours_Operated"].mean()))
c3.metric("Avg Breakdowns",round(filtered_df["Breakdowns"].mean(),1))

st.markdown("</div>",unsafe_allow_html=True)

st.markdown("<div class='card'>",unsafe_allow_html=True)
st.subheader("Usage & Reliabity Pattern")

c1, c2 = st.columns(2)

with c1:
    fig, ax = plt.subplots(facecolor="#161b22")
    ax.hist(
        filtered_df["Hours_Operated"],
        bins=25,
        color="#58a6ff",
        edgecolor="white"
    )
    ax.set_title("Hours Operated Distribution",color="white")
    ax.tick_params(colors="white")
    st.pyplot(fig)

    with c2:
        fig, ax = plt.subplots(facecolor="#161b22")
        ax.scatter(
            filtered_df["Hours_Operated"],
            filtered_df["Breakdowns"],
            color = "#f85149",
            alpha=0.7
        )
        ax.set_title("Hours vs Breakdowns",color="white")
        ax.tick_params(colors="white")
        st.pyplot(fig)

st.markdown("</div>",unsafe_allow_html=True)

st.markdown("<div class='card'>",unsafe_allow_html=True)
st.subheader("Failure Prediction Tool")

c1,c2,c3 = st.columns(3)

with c1:
    hours = st.number_input("Hours_Operated",0,3000,1200)
with c2:
    breakdowns = st.number_input("Breakdown",0,50,5)
with c3:
    st.write("")
    Prediction_Button =st.button("Run Prediction")

if Prediction_Button:
    input_df = pd.DataFrame(
        [[hours,breakdowns]],
        columns= ["Hours_Operated","Breakdowns"
    ])
    prediction = model.predict(input_df)[0]

    st.metric(
        "Predicted Next Failure(Hours)",
        f"{prediction:,.0f}hrs"
    )  
st.markdown("</div>",unsafe_allow_html=True)
st.markdown("---")
