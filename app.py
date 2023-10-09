import streamlit as st
#import pickle
#from model_file import Model
from sklearn.svm import LinearSVR, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor 
from sklearn.svm import LinearSVR
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def main():
    st.title("Predicting the 3D-Printing Parameters")

    # Get the input values from the user
    input1 = st.number_input("Roughness", value=0.0)
    input2 = st.number_input("Tension Strenght", value=0.0)
    input3 = st.number_input("Elongation", value=0.0)
    

         
        
    import pandas as pd
    df = pd.read_csv('3d_printing_data.csv')
    
    pla_df = df[df['material']=='pla']
    
    from sklearn.preprocessing import OrdinalEncoder
    enc1 = OrdinalEncoder()

    enc1.fit(pla_df[['infill_pattern']])
    pla_df[['infill_pattern']] = enc1.transform(pla_df[['infill_pattern']])
    pla_df = pla_df.drop(columns=['material'],axis=1)
    
    pla_df= pla_df.drop(['bed_temperature','fan_speed'],axis=1)
    
    
    from sklearn.model_selection import train_test_split
    d_train, d_test = train_test_split(pla_df, random_state=22, test_size=0.1)

    
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler2 = StandardScaler()


    x = d_train[['roughness','tension_strenght','elongation']]
    y = d_train.drop(['roughness','tension_strenght','elongation'], axis=1)

    x_1 = x.copy()
    y_1 = y.copy()

    scaler.fit(x)
    x_1[:] = scaler.transform(x)

    scaler2.fit(y)
    y_1[:] = scaler2.transform(y) 
    
    

    class Model(BaseEstimator, TransformerMixin):
        def fit(self, x, y):

            self.scaler = StandardScaler()
            self.scaler2 = StandardScaler()

            self.x_1 = x.copy()
            self.y_1 = y.copy()

            self.scaler.fit(x)        
            self.scaler2.fit(y)

            self.x_1[:] = self.scaler.transform(x) 
            self.y_1[:] = self.scaler2.transform(y) 

            from sklearn.svm import LinearSVR
            model = LinearSVR()

            self.wrapper = MultiOutputRegressor(model)
            self.wrapper.fit(self.x_1, self.y_1)


            return self.wrapper

        def predict_one_instance(self, x):

            predictions = self.wrapper.predict(self.scaler.transform(x))
            self.output = self.scaler2.inverse_transform(predictions)
            self.output = np.around(self.output,3)

            return np.around(self.output,3)
    
    model = Model()
    x = d_train[['roughness','tension_strenght','elongation']]
    y = d_train.drop(['roughness','tension_strenght','elongation'], axis=1)
    model.fit(x,y)
    
    
    if st.button("Calculate"):
        # Calculate the output values
        output1, output2, output3, output4, output5, output6 = model.predict_one_instance([[input1, input2, input3]])[0]

        # Display the output values
        st.subheader("The predicted parameters for the PLA material")
        st.write(f"Layer Height: {output1}")
        st.write(f"Wall Thickness: {output2}")
        st.write(f"Infill Density: {output3}")
        st.write(f"Infill Pattern: {enc1.inverse_transform([[output4]])[0][0]}")
        st.write(f"Nozzle Temperature: {output5}")
        st.write(f"Print Speed: {output6}")

if __name__ == "__main__":
    main()
