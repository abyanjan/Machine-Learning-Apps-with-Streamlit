import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
from sklearn.metrics import recall_score, precision_score, plot_confusion_matrix

import streamlit as st


def app():
    
    st.title("Training Machine learning Models")
    
    #load_data
    @st.cache()
    def get_data():
        cancer = load_breast_cancer()
        data = pd.concat([pd.DataFrame(cancer.data),pd.DataFrame(cancer.target)], axis =1)
        data.columns = np.append(cancer.feature_names,['target'])
        
        return data
    
    # train test split
    data = get_data()
    train_x, test_x, train_y, test_y = train_test_split(data.drop('target', axis = 1), 
                                                        data['target'], 
                                                        stratify = data['target'],
                                                        test_size = 0.3,
                                                        random_state = 99)
               
    # normalization
    scalar = StandardScaler().fit(train_x)
    train_x_scaled = scalar.transform(train_x)
    test_x_scaled = scalar.transform(test_x)
    
    
    # Model training

    @st.cache(allow_output_mutation=True)
    def train_model(model_name, param_dict):
        if model_name =="Logistic Regression":
            clf = LogisticRegression(C = param_dict['C'],solver='liblinear').fit(train_x_scaled, train_y)
        elif model_name =="KNearest Neighbor":
            clf = KNeighborsClassifier(n_neighbors=param_dict['k']).fit(train_x_scaled, train_y)
        elif model_name == "Random Forest":
            clf = RandomForestClassifier(n_estimators=param_dict['n_est'],
                                         max_depth = param_dict['max_depth']).fit(train_x, train_y)
        return clf
        
    @st.cache()
    def eval_results(model_name:str,clf, data, true_labels):
        pred_prob = clf.predict_proba(data)
        pred_class = clf.predict(data)
        
        auc = roc_auc_score(true_labels, pred_prob[:,1])
        f1 = f1_score(true_labels, pred_class)
        precision = precision_score(true_labels, pred_class)
        recall = recall_score(true_labels, pred_class)
        
        return { model_name : [auc, f1, precision, recall]}
    
  
    param_dict = {}
    #results ={'Model': [model_name], "AUC":[auc], "F1-Score":[f1],
       #       "Precision":[precision], "Recall":[recall]}
    results = {}
    
    
    col1, col2, col3 = st.beta_columns(3)
    
    with col1:
        # Logistic Regression
        st.subheader("Logistic Regression")
        cost_param = st.select_slider(label="Select Cost parameter", 
                             options=np.logspace(-4,4, 4).tolist())
        param_dict.update({'C':cost_param})
        if st.checkbox("Train Logsitic Regression"):
            
            show_text_lr = st.text("Starting the training process ---------")
            lr_clf = train_model(model_name ="Logistic Regression", param_dict=param_dict)
            lr_results = eval_results(model_name="Logistic Regression", clf = lr_clf, 
                         data = test_x_scaled,true_labels=test_y)
            results.update(lr_results)
                
            show_text_lr.text("Finished Training ------ Done!!!")
            
    with col2:
        # Knearest neighbor
        st.subheader("KNearest Neighbor")
       
        k_param = st.slider(label="Select numer of neighbors(k)", 
                                min_value=1, max_value=50)   
        
        param_dict.update({'k':k_param})
        if st.checkbox("Train KNearest Neighbor"):
            
            show_text_knn = st.text("Starting the training process ---------")
            knn_clf = train_model(model_name ="KNearest Neighbor", param_dict=param_dict)
            knn_results = eval_results(model_name="KNearest Neighbor", clf = knn_clf, 
                         data = test_x_scaled,true_labels=test_y)
            results.update(knn_results)
                
            show_text_knn.text("Finished Training ------ Done!!!")
            
    with col3:
       # Random Forest
        st.subheader("Random Forest")
       
        n_est = st.slider(label = "Select Number of Estimators", min_value=50, max_value=500)
        max_depth = st.slider(label = "Select maximum depth", min_value=3, max_value=10) 
        
        param_dict.update({'n_est':n_est, 'max_depth':max_depth})
        if st.checkbox("Train Random Forest"):
            show_text_rf = st.text("Starting the training process ---------")
            rf_clf = train_model(model_name ="Random Forest", param_dict=param_dict)
            rf_results = eval_results(model_name="Random Forest", clf = rf_clf, 
                         data = test_x, true_labels=test_y)
            results.update(rf_results)
            show_text_rf.text("Finished Training ------ Done!!!")
            
    st.subheader("Evaluation Results on Validation Data")
    
    if st.checkbox("Show Results"):
        st.text("Showing Comparison of the model results....")
        st.write(pd.DataFrame(results, index=['AUC', 'F1-Score','Precision','Recall']))
            
    
    
    
       