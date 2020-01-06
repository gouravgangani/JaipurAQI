import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
import pickle
import os
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from tpot import TPOTRegressor



class modelFactory:
   def __init__(self,X,y,X_train,X_test,y_train,y_test):
       self.X_train = X_train
       self.X_test = X_test
       self.y_train = y_train
       self.y_test = y_test
       self.X = X
       self.y = y

   def variables(self,X_train,X_test):
        print(X_train.shape)

class linear_Model(modelFactory):
    from sklearn import metrics
  
    def __init__(self,*args):
        self.lm = LinearRegression()
        super(linear_Model,self).__init__(*args)
    
        
    def linear_model(self):
        #lm = LinearRegression()
        self.lm.fit(self.X_train,self.y_train)
        prediction = self.lm.predict(self.X_test)
        return prediction 

    def model_eval(self):
        prediction = self.linear_model()
        mae =  metrics.mean_absolute_error(self.y_test, prediction)
        mse =  metrics.mean_squared_error(self.y_test, prediction)
        rmse = np.sqrt(metrics.mean_squared_error(self.y_test, prediction))
        return mae,mse,rmse

    
class random_tree_model(modelFactory):
    def __init__(self, *args):
        self.rm = RandomForestRegressor()
        super(random_tree_model,self).__init__(*args)

    
    def rt_model(self):
        #self.rm = RandomForestRegressor()
        self.rm.fit(self.X_train,self.y_train)
        return self.rm
       

    def coef_of_determination(self):
        self.m = self.rt_model()
        print("Coefficient of Determination R^2 on Train Set: {}".format(self.m.score(self.X_train,self.y_train)))
        print("Coefficient of Determination R^2 on Test Set: {}".format(self.m.score(self.X_test,self.y_test)))

    def cross_val(self):
        from sklearn.model_selection import cross_val_score
        score = cross_val_score(self.rm,self.X,self.y)
        print(score.mean())

    def rt_model_predict(self):
        self.rt = self.rt_model()
        prediction = self.rt.predict(self.X_test)
        #sns.distplot(self.y_test-prediction)
        #plt.scatter(self.y_test,prediction)


    #Hyper Parameter Tuning
    def hyper_paramenter_tuning(self):
        n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
        max_features = ['auto','sqrt']
        max_depth = [int(x) for x in np.linspace(5,30,num = 6)]
        min_samples_split = [2,5,10,15,100]
        min_samples_leaf = [1,2,5,10]

        #Random Grid
        random_grid = { 'n_estimators': n_estimators,
                        'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf}

        #self.rm = RandomForestRegressor()
        self.rm_hyper = RandomizedSearchCV(estimator = self.rm, param_distributions = random_grid, scoring = 'neg_mean_squared_error', n_iter= 100, cv = 5, verbose= 2, random_state= 42, n_jobs = 1)
        self.rm_hyper.fit(self.X_train,self.y_train)
        print("Best Parameters: \n", self.rm_hyper.best_params_)
        print("Best Score:", self.rm_hyper.best_score_)
        predictions = self.rm_hyper.predict(self.X_test)
        return predictions

    
    #Retrieving the Metrics
    def random_model_metrics(self):
        prediction = self.hyper_paramenter_tuning()
        print('MAE:', metrics.mean_absolute_error(self.y_test, prediction))
        print('MSE:', metrics.mean_squared_error(self.y_test, prediction))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(self.y_test, prediction)))

    
    def random_pickle(self):
        if not os.path.exists("Data/Pickle_Models"):
            os.makedirs("Data/Pickle_Models")
        self.file = open("Data/Pickle_Models/randomForestRegressor.pkl",'wb')
        pickle.dump(self.rm_hyper,self.file)
    
class xgboost_model(modelFactory):
    def __init__(self,*args):
        self.xg = xgb.XGBRegressor()
        super(xgboost_model,self).__init__(*args)


    def xgb_coef_of_determination(self):
        self.xg.fit(self.X_train,self.y_train)
        print("Coefficient of Determination R^2 on Train Set: {}".format(self.xg.score(self.X_train,self.y_train)))
        print("Coefficient of Determination R^2 on Test Set: {}".format(self.xg.score(self.X_test,self.y_test)))

    def xgb_cross_val(self):
        from sklearn.model_selection import cross_val_score
        score = cross_val_score(self.xg,self.X,self.y,cv= 5)
        print("Score Mean",score.mean())
   

    #Hyper Parameter Tuning
    def xgb_hyper_paramenter_tuning(self):
        n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
        learning_rate = ['0.05','0.1','0.2','0.3','0.5','0.6']
        max_depth = [int(x) for x in np.linspace(5,30, num = 6)]
        subsample = [0.7,0.6,0.8]
        min_child_weight = [3,4,5,6,7]

        #Random Grid
        random_grid = { 'n_estimators': n_estimators,
                        'learning_rate': learning_rate,
                        'max_depth': max_depth,
                        'subsample': subsample,
                        'min_child_weight': min_child_weight,
                        } 

        #self.rm = RandomForestRegressor()
        self.xg_hyper = RandomizedSearchCV(estimator = self.xg, param_distributions = random_grid, scoring = 'neg_mean_squared_error', n_iter= 50, cv = 5, verbose= 2, random_state= 42, n_jobs = 5)
        self.xg_hyper.fit(self.X_train,self.y_train)
        print("Best Parameters: \n", self.xg_hyper.best_params_)
        print("Best Score:", self.xg_hyper.best_score_)
        predictions = self.xg_hyper.predict(self.X_test)
        return predictions


    #Retrieving the Metrics
    def xgb_model_metrics(self):
        prediction = self.xgb_hyper_paramenter_tuning()
        print('MAE:', metrics.mean_absolute_error(self.y_test, prediction))
        print('MSE:', metrics.mean_squared_error(self.y_test, prediction))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(self.y_test, prediction)))

    
    def xgb_pickle(self):
        if not os.path.exists("Data/Pickle_Models"):
            os.makedirs("Data/Pickle_Models")
        self.file = open("Data/Pickle_Models/xgboostRegressor.pkl",'wb')
        pickle.dump(self.xg_hyper,self.file)

    
class tpot(modelFactory):
    def __init__(self,*args):
        self.tpt = TPOTRegressor(generations=10, population_size=100)
        super(tpot,self).__init__(*args)

    def tp_model(self):
        self.tpt.fit(self.X_train, self.y_train)
        print(self.tpt.score(self.X_test, self.y_test))
        predictions = self.tpt.predict(self.X_test)
        return predictions
    
    #Retrieving the Metrics
    def tp_model_metrics(self):
        prediction = self.tp_model()
        print('MAE:', metrics.mean_absolute_error(self.y_test, prediction))
        print('MSE:', metrics.mean_squared_error(self.y_test, prediction))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(self.y_test, prediction)))

    def tp_pickle(self):
        if not os.path.exists("Data/Pickle_Models"):
            os.makedirs("Data/Pickle_Models")
        self.file = open("Data/Pickle_Models/TPOTRegressor.pkl",'wb')
        pickle.dump(self.tpt,self.file)


'''
if __name__ == "__main__":
    #df = pd.read_csv("Data/csvData/Scrubbed_Data.csv")
    df = pd.read_csv("Data/csvData/Scrubbed_File.csv")
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)
    #mf = modelFactory(X_train,X_test,y_train,y_test)
    #prediction = mlinear_model()
    #print(prediction.shape)
    mf = linear_Model(X_train,X_test,y_train,y_test)
    prediction = mf.linear_model()
    mae,mse,rmse = mf.model_eval()
    print(mae,mse,rmse)
    #print(prediction)

    '''

    