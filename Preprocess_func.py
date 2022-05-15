import os
import numpy as np
import pandas as pd
import warnings
from config import config_dict
warnings.filterwarnings('ignore')


def preprocess_hr():
    file_list = os.listdir(os.getcwd())
    for i in file_list:
        if i == "train.csv" or i == "test.csv":
            file_name = i
            data = pd.read_csv(file_name)
            data_copy = data.copy()
            department_new = data_copy['department'].value_counts().to_dict()
            data_copy['department_new'] = data_copy['department'].map(department_new)
            data_copy.drop(columns=['department', 'employee_id'], inplace=True)
            for j in range(len(data_copy)):
                data_copy['region'][j] = int(data_copy['region'][j][7:])
            data_copy['region'] = data_copy['region'].astype(int)
            data_copy['education'] = data_copy['education'].fillna(data_copy['education'].mode()[0])
            data_copy['education'] = data_copy['education'].map(config_dict['education_dict'])
            data_copy = pd.get_dummies(data=data_copy, columns=['gender'], drop_first=True)
            data_copy['recruitment_channel'] = data_copy['recruitment_channel'].map(
                config_dict['recruitment_channel_dict'])
            data_copy['previous_year_rating'] = data_copy['previous_year_rating'].fillna(
                data_copy['previous_year_rating'].median())
            if file_name == "train.csv":
                data_copy.to_csv('pre_processed_train_data.csv', index=False)
                print('train_file_completed')
                print(data_copy.shape)
            else:
                data_copy.to_csv('pre_processed_test_data.csv', index=False)
                print('test_file_completed')

preprocess_hr()
