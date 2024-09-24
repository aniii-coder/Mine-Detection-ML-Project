
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split

# #Load the data
# sonar_data = pd.read_csv('/content/Copy of sonar data.csv', header=None)
# print(sonar_data.head())

# #Reviewing the details of data
# sonar_data.describe()
# sonar_data[60].value_counts()
# sonar_data.groupby(60).mean()

# #Splitting the data
# X=sonar_data.drop(columns=60,axis=1)
# Y = sonar_data[60]
# X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1,random_state=1)

# #TRain the model
# model = LogisticRegression()
# model.fit(X_train, Y_train)

# #Test the model
# int_data = (0.0269,0.0383,0.0505,0.0707,0.1313,0.2103,0.2263,0.2524,0.3595,0.5915,0.6675,0.5679,0.5175,0.3334,0.2002,0.2856,0.2937,0.3424,0.5949,0.7526,0.8959,0.8147,0.7109,0.7378,0.7201,0.8254,0.8917,0.9820,0.8179,0.4848,0.3203,0.2775,0.2382,0.2911,0.1675,0.3156,0.1869,0.3391,0.5993,0.4124,0.1181,0.3651,0.4655,0.4777,0.3517,0.0920,0.1227,0.1785,0.1085,0.0300,0.0346,0.0167,0.0199,0.0145,0.0081,0.0045,0.0043,0.0027,0.0055,0.0057)
# int_arrdata = np.asarray(int_data)
# data_reshaped = int_arrdata.reshape(1,-1)
# prediction= model.predict(data_reshaped)

# if(prediction=='M'):
#   print('Mine has been Discovered')
# else:
#   print('Rock has been Discovered')






import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the data
sonar_data = pd.read_csv(r'C:\Users\aksin\OneDrive\Desktop\Anonymous_coder\newProjects.py\github_python\Mine-Detection-ML-Project\Copy of sonar data.csv', header=None)

# Prepare the data
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=1)

# Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Streamlit UI
st.title('Sonar Data Prediction')
st.write('Input an array of sonar readings (60 values):')

# Input array
input_data = st.text_input(
    "Enter sonar readings (comma-separated):", 
   
)

# Button to predict
if st.button('Submit'):
    # Convert input to numpy array
    try:
        int_data = np.array([float(x.strip()) for x in input_data.split(',')])
        
        if len(int_data) != 60:
            st.error('Please enter exactly 60 values.')
        else:
            # Reshape for prediction
            data_reshaped = int_data.reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(data_reshaped)
            
            # Output prediction
            if prediction[0] == 'M':
                st.success('Mine has been Discovered')
            else:
                st.success('Rock has been Discovered')
    except ValueError:
        st.error('Please enter valid numbers separated by commas.')

# To run this app, save it in a file named 'app.py' and run `streamlit run app.py`
