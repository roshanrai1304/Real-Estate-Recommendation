import streamlit as st
import numpy as np
import pandas  as pd
from src.data_cleaning import DataCleaning
from src.data_cleaning import DataPreProcessingStrategy
from src.data_cleaning import DataDivideStrategy
import joblib
import pickle


columns_for_df = ['closestEducationalInstituteDistance',
                'flatType',
                'Area',
                'pricePerSqFeet',
                'shoppingArea']
cluster_columns = ['builderName', 'State', 'District']
def make_predictions(inputs):
    
    data = pd.DataFrame([inputs], columns=columns_for_df)
    data['Area'] = np.log(data['Area']) 
    data['pricePerSqFeet'] = np.log(data['pricePerSqFeet'])
    with open("scalers/scaling_y.pkl", 'rb') as f:
      sc_y = pickle.load(f)
    with open("scalers/scaling_X.pkl", 'rb') as f:
      sc_X = pickle.load(f)
    with open("clusters/kmeans.pkl", 'rb') as f:
      kmeans = pickle.load(f)
    cluster_data = pd.read_csv(r"C:\Users\HP\Documents\mihir project\Real-Estate-Recommendation mlflow\clusters\cluster_data.csv")
    with open("saved_models/GradientBoost.pkl", "rb") as f:
      model = pickle.load(f)
    inputs_scaled = sc_X.transform(data.to_numpy()) 
    y = model.predict(inputs_scaled)
    cluster_group = kmeans.predict(inputs_scaled)[0]
    cluster_result = cluster_data[cluster_data['cluster'] == cluster_group].sample(5)[cluster_columns]
    y_inverse_scale = np.exp(sc_y.inverse_transform(y))
    print(f"y_inverse_scale = {y_inverse_scale}")
    return y_inverse_scale, cluster_result
    
    
    

def main():
    st.title("Real Estate Prediction")
    
    #Input fields
    
    flat_type = st.number_input('What type of flat are you looking?', min_value=1, max_value=4)
    pricePerSQFt = st.number_input('What is price per sql ft are you looking?', min_value=20000, max_value=30000)
    Area = st.number_input("What is the area you are looking?", min_value=400, max_value=1800)
    shopping_area = st.number_input("do you want shopping area nearby?", min_value=0, max_value=1)
    education_distance = st.number_input("What is the distance of education yoiu are looking?", min_value=20, max_value=1400)
    
    inputs = [education_distance, flat_type, Area, pricePerSQFt, shopping_area]
    
    if st.button('Predict'):
        prediction, cluster_results = make_predictions(inputs)
        builders = list(set(cluster_results['builderName'].to_list()))
        print(builders)
        st.success(f"Prediction: {prediction}")
        st.success(f"You can look following builders in Mumbai:")
        for i in builders :
          st.success(i)
        
        
if __name__ == "__main__":
    main()
    
