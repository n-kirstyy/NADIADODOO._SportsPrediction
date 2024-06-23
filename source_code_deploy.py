import numpy as np
import pandas as pd
import streamlit as st 
import joblib


def data_preparation (df):
    df.dropna(axis = 1, thresh=(1-30/100)*len(df), inplace = True) #drops all columns with more than 30% null values

    # split data into numeric and non-numeric data 
    numeric = df.select_dtypes (include = np.number) 
    non_numeric = df.select_dtypes (include = ['object'])

    #imputing numeric data
    imp = IterativeImputer(max_iter=10, random_state=0)
    numeric_imputed = pd.DataFrame(np.round(imp.fit_transform(numeric)), columns = numeric.copy().columns)

    #imputing non numeric data 
    imp2 = SimpleImputer(strategy = 'most_frequent')
    non_numeric_imputed = pd.DataFrame (imp2.fit_transform(non_numeric), columns=non_numeric.copy().columns)

    #encoding non numeric data
    non_num_encoded = BinaryEncoder().fit_transform(non_numeric_imputed)

    #creating dependent and independent variables
    y = numeric_imputed['overall']
    x = pd.concat([numeric_imputed, non_num_encoded], axis=1)

    correlation_matrix = x.corr()['overall'].abs().sort_values (ascending=False)
    selected_features = correlation_matrix[:16]
    x = x[selected_features.index]
    x.drop('overall', axis=1, inplace = True)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(x)
    x = pd.DataFrame(scaled, columns = x.columns)
    
    return x, y
   
  
model = joblib.load('random_forest_model.pkl')

st.title('Player Rating Prediction')


name = st.text_input('Name')
age = st.number_input('Age')
pos = st.text_input('Position')
yrs = st.number_input('Years of Experience')




if st.button('Predict Rating'):

    # Preprocess data (replace with your functions, ensure compatibility with 1.2.2)
    player_data = {'Name': name, 'Age': age, 'Position': pos, 'Years of Experience': yrs}
    data = pd.DataFrame.from_dict(player_data)
    x, y = data_preparation(data)  

    prediction = model.predict(x)

    st.success(f"Predicted Rating for {name}: {prediction:.2f}") 

      
if __name__=='__main__': 
    main()