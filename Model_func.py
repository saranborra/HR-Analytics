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

def model_hr():
    train_scores, test_scores = [], []
    file_list = os.listdir(os.getcwd())
    LR = LogisticRegression()
    DT = DecisionTreeClassifier()
    RF = RandomForestClassifier()
    Reg_list = [LR, DT, RF]
    for i in file_list:
        if i == "pre_processed_train_data.csv":
            file_name = i
            data = pd.read_csv(file_name)
            data_copy = data.copy()
            X = data_copy.drop(columns='is_promoted')
            y = data_copy['is_promoted']
            SM = SMOTE(random_state=2)
            X_sm, y_sm = SM.fit_resample(X, y)
            X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.3, random_state=424)
            for j in Reg_list:
                j.fit(X_train, y_train)
                y_pred = j.predict(X_test)
                y_pred_train = j.predict(X_train)
                train_scores.append(np.round(f1_score(y_train, y_pred_train), 2))
                test_scores.append(np.round(f1_score(y_test, y_pred), 2))
                if j == LR:
                    file_store_name = 'finalized_model_lr.sav'
                elif j == DT:
                    file_store_name = 'finalized_model_dt.sav'
                else:
                    file_store_name = 'finalized_model_rf.sav'
                pickle.dump(j, open(file_store_name, 'wb'))
            fin_train_scores = pd.DataFrame({'Model_Used': Reg_list,
                                             'Train_f1_Score': train_scores,
                                             'Test+f1_Score': test_scores}
                                            )
            fin_train_scores.to_csv('fin_train_scores.csv', index=False)
            print("DONE")

model_hr()