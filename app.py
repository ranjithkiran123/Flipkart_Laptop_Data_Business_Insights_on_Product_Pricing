import streamlit as st
import pickle
import numpy as np
import pandas as pd
pipe = pickle.load(open("pipe.pkl", "rb"))
df=pickle.load(open("df.pkl","rb"))
st.title("Laptop Price Predictor")
oss= st.selectbox("Operating System", df['Operating_System'].unique())
rams= st.selectbox("RAM Size", df['RAM_Size'].unique())
store= st.selectbox("Storage", df['Storage'].unique())
proc= st.selectbox("Processor", df['Processor'].unique())
ramt= st.selectbox("RAM Type", df['RAM_Type'].unique())
if st.button('Predict the Laptop Price'):
    query = np.array([proc,oss,rams,ramt,store])
    query = query.reshape(1,-1)
    p = pipe.predict(query)[0]
    result = np.exp(p)
    st.subheader("Laptop Predicted Prize : ")
    st.subheader(":blue[₹{}]".format(result.round(2)))