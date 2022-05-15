import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')
import pickle
file_list = os.listdir(os.getcwd())
def model_fit():
    for i in file_list:
        if i == "pre_processed_test_data.csv":
            file_name = i
            data = pd.read_csv(file_name)
        if i == "fin_train_scores.csv":
            file_name = i
            score_data = pd.read_csv(file_name)
            score_data['Model_Used'][score_data['Test+f1_Score'] == score_data['Test+f1_Score'].max()]

    loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
    y_predc = loaded_model.predict(data_num)
    data_num['is_promoted'] = y_predc
    data_num.to_csv('Final_Data.csv', index=False)