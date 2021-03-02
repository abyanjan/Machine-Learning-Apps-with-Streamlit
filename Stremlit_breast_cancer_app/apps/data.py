import pandas as pd
import numpy as np
import sklearn
import streamlit as st

from sklearn.datasets import load_breast_cancer
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


def app():
    
    st.write(''' ## Data ''')
    st.write('Data Source: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)')
    
    
    # load data from sklearn
    #@st.cache 
    cancer = load_breast_cancer()
    data = pd.concat([pd.DataFrame(cancer.data),pd.DataFrame(cancer.target)], axis =1)
    data.columns = np.append(cancer.feature_names,['target'])
    
    st.subheader('Showing Few Data Examples')
    st.dataframe(data.head(5))
    
    st.subheader('Data Description')
    
    # Number of rows and columns
    st.write('Number of Data instances(Rows):', data.shape[0])
    st.write('Number of Data Features(Rows):', data.shape[1]-1)
    
    data_description = pd.DataFrame()
    #data_description['Features'] = np.append(cancer.feature_names,['target'])
    features = ['radius','texture','perimeter','area','smoothness','compactness',
                'concavity','concave points','symmetry','fractcal dimension']
    feature_info = ['mean of distances from center to points on the perimeter',
                    'standard deviation of gray-scale values',
                    '','','local variation in radius lengths',
                    'perimeter^2 / area - 1.0', 
                    'severity of concave portions of the contour',
                    'number of concave portions of the contour','',
                    '"coastline approximation" - 1']
    
    data_description['Attributes'] = features
    data_description['Description'] = feature_info
    data_description.set_index('Attributes')
    st.write(data_description)
    
    st.write('The mean, standard error, and "worst" or largest',
             '(mean of the three largest values) of these features were computed for each image',
             'resulting in 30 features.')
    
    #target
    st.write('**Target Label**')
    st.write('malignant:', 0, 'benign:', 1 )
    
    target_dist = data.target.value_counts()
    st.bar_chart(target_dist)
    
   
    
   
