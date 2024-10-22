# -*- coding:utf-8 -*-

import pickle
import os
import json
import pandas as pd
import numpy as np



DIR_NAME = os.path.dirname(os.path.abspath(__file__))

xgboost_file = open(f"{DIR_NAME}/xgboost_car_price_predict_polo_2015_2020_18-10-2024.pkl", "rb")
dt_file = open(f"{DIR_NAME}/dt_car_price_predict_polo_2015_2020_18-10-2024.pkl", "rb")
lr_file = open(f"{DIR_NAME}/lr_car_price_predict_polo_2015_2020_18-10-2024.pkl", "rb")
rf_file = open(f"{DIR_NAME}/rf_car_price_predict_polo_2015_2020_18-10-2024.pkl", "rb")
svr_file = open(f"{DIR_NAME}/svr_car_price_predict_polo_2015_2020_18-10-2024.pkl", "rb")

xgboost_model = pickle.load(xgboost_file)
dt_model    = pickle.load(dt_file)
lr_model    = pickle.load(lr_file)
rf_model    = pickle.load(rf_file)
svr_model   = pickle.load(svr_file)

file_name_car_le = open(f"{DIR_NAME}/name_car_le.pkl", 'rb')
file_engine_type_le = open(f"{DIR_NAME}/engine_type_le.pkl", 'rb')
file_transmission_type_le = open(f"{DIR_NAME}/transmission_type_le.pkl", 'rb')

Name_car_le             = pickle.load(file_name_car_le)
Engine_type_le          = pickle.load(file_engine_type_le)
Transmission_type_le    = pickle.load(file_transmission_type_le)


def models_predict(rf_model,xgboost_model,lr_model,dt_model,svr_model,new_data_df):
    ''' RandomForestRegressor '''
    y_pred_rf = rf_model.predict(new_data_df)
    ''' XGBRegressor '''
    y_pred_xgboost = xgboost_model.predict(new_data_df)
    ''' LinearRegression '''
    y_pred_lr = lr_model.predict(new_data_df)
    ''' DecisionTreeRegressor '''
    y_pred_dt = dt_model.predict(new_data_df)
    ''' SVR '''
    y_pred_svr = svr_model.predict(new_data_df)
    pred_dict = {}
    pred_dict['rf'] = np.expm1(y_pred_rf)
    pred_dict['xgboost'] = np.expm1(y_pred_xgboost)
    pred_dict['lr'] = np.expm1(y_pred_lr)
    pred_dict['dt'] = np.expm1(y_pred_dt)
    pred_dict['SVR'] = np.expm1(y_pred_svr)

    return pred_dict

def predict_price(year=None,milliage_km =None,engine_type=None,capacity=None,transmission_type=None):
    try:
        new_data = {
            "name_car": "Volkswagen Polo Sedan I · Рестайлинг",
            "engine_type":engine_type,
            "transmission_type":transmission_type,
            "capacity":capacity,
            "year": year,
            "mileage_km_log":np.log1p(milliage_km),
        }
        new_data['name_car'] = Name_car_le.transform([new_data["name_car"]])[0]
        new_data['engine_type'] = Engine_type_le.transform([new_data["engine_type"]])[0]
        new_data['transmission_type'] = Transmission_type_le.transform([new_data["transmission_type"]])[0]
        new_data_df = pd.DataFrame([new_data])
        pred_dict = models_predict(rf_model,xgboost_model,lr_model,dt_model,svr_model,new_data_df)
        print(pred_dict)
        return pred_dict
    except:
        return None

if __name__ == "__main__":
    pass
