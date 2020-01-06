import pandas as pd 
from sklearn.model_selection import train_test_split
from modelFactory import linear_Model
from modelFactory import random_tree_model
from modelFactory import xgboost_model
from modelFactory import tpot
import time







if __name__ == "__main__":
    startTime = time.time()
    #df = pd.read_csv("Data/csvData/Scrubbed_Data.csv")
    df = pd.read_csv("Data/csvData/Scrubbed_File.csv")
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)
    #mf = modelFactory(X_train,X_test,y_train,y_test)
    #prediction = mlinear_model()
    #print(prediction.shape)
    
    
    #mf = linear_Model(X_train,X_test,y_train,y_test)
    #prediction = mf.linear_model()
    #mae,mse,rmse = mf.model_eval()
    #print(mae,mse,rmse)

    #Random Forest Regressor - Model Testing 
    #rf = random_tree_model(X,y,X_train,X_test,y_train,y_test)
    #rf.hyper_paramenter_tuning()
    #rf.random_model_metrics()
    #rf.random_pickle()


    #XGBoostRegressor
    #xgbt = xgboost_model(X,y,X_train,X_test,y_train,y_test)
    #xgbt.xgb_hyper_paramenter_tuning()
    #xgbt.xgb_model_metrics()
    #xgbt.xgb_pickle()

    #TPOT Regressor
    tpt = tpot(X,y,X_train,X_test,y_train,y_test)
    #tpt.tp_model()
    tpt.tp_model_metrics()
    tpt.tp_pickle()

    stopTime = time.time()
    print("Time Taken: {}".format(stopTime-startTime))
   