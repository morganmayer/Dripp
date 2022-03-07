import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import yaml
from datetime import datetime
import os
import shutil
import seaborn as sns
import pickle
from pickle import dump
from scipy.signal import find_peaks

from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

from mpl_toolkits.mplot3d import Axes3D
import collections
import multiprocessing
from multiprocessing import Pool

class Analysis:
            
    def calculate_save_rmsle(self):
        
        self.all_test_rmsle =  [[self.rmsle(self.y_train, self.all_train_predictions[t][i]) for i in range(len(self.models))] for t in range(len(self.feature_transformations))]
        self.all_train_rmsle = [[self.rmsle(self.y_test, self.all_test_predictions[t][i]) for i in range(len(self.models))] for t in range(len(self.feature_transformations))]
        
        dict_test_rmsle = {}
        for t, transform in enumerate(self.feature_transformations):            
            dict_test_rmsle[str(self.transform_names[t])] = self.all_test_rmsle[t]        
        df_test_rmsle = pd.DataFrame.from_dict(dict_test_rmsle, 
                                              orient='index',
                                              columns=self.models)        
        self.df_test_rmsle = df_test_rmsle.copy()        
        df_test_rmsle.to_csv(f"{self.path}/df_test_RMSLE_matrix.csv", float_format='%.3f')
        
        print("Test RMSLE", self.df_test_rmsle)
        
        dict_train_rmsle = {}        
        for t, transform in enumerate(self.feature_transformations):            
            dict_train_rmsle[str(self.transform_names[t])] = self.all_train_rmsle[t]        
        df_train_rmsle = pd.DataFrame.from_dict(dict_train_rmsle, 
                                              orient='index',
                                              columns=self.models)        
        self.df_train_rmsle = df_train_rmsle.copy()        
        df_train_rmsle.to_csv(f"{self.path}/df_train_RMSLE_matrix.csv", float_format='%.3f')
        
        #print("Train RMSLE", self.df_train_rmsle)
        
    def rmse(self, y_true, y_pred):
        return np.sqrt(np.sum(np.abs(y_pred - y_true)))
        
    def calculate_save_rmse(self):
        
        self.all_test_rmse =  [[self.rmse(self.y_train, self.all_train_predictions[t][i]) for i in range(len(self.models))] for t in range(len(self.feature_transformations))]
        self.all_train_rmse = [[self.rmse(self.y_test, self.all_test_predictions[t][i]) for i in range(len(self.models))] for t in range(len(self.feature_transformations))]
        
        dict_test_rmse = {}
        for t, transform in enumerate(self.feature_transformations):            
            dict_test_rmse[str(self.transform_names[t])] = self.all_test_rmse[t]        
        df_test_rmse = pd.DataFrame.from_dict(dict_test_rmse, 
                                              orient='index',
                                              columns=self.models)        
        self.df_test_rmse = df_test_rmse.copy()        
        df_test_rmse.to_csv(f"{self.path}/df_test_RMSE_matrix.csv", float_format='%.3f')
        
        print("Test RMSE", self.df_test_rmse)
        
        dict_train_rmse = {}        
        for t, transform in enumerate(self.feature_transformations):            
            dict_train_rmse[str(self.transform_names[t])] = self.all_train_rmse[t]        
        df_train_rmse = pd.DataFrame.from_dict(dict_train_rmse, 
                                              orient='index',
                                              columns=self.models)        
        self.df_train_rmse = df_train_rmse.copy()        
        df_train_rmse.to_csv(f"{self.path}/df_train_RMSE_matrix.csv", float_format='%.3f')
        
        ###########3
        
        # R2 coefficient_of_dermination = r2_score(y, p(x))
        
        self.all_test_r2 =  [[r2_score(self.y_train, self.all_train_predictions[t][i]) for i in range(len(self.models))] for t in range(len(self.feature_transformations))]
        self.all_train_r2 = [[r2_score(self.y_test, self.all_test_predictions[t][i]) for i in range(len(self.models))] for t in range(len(self.feature_transformations))]
        
        dict_test_r2 = {}
        for t, transform in enumerate(self.feature_transformations):            
            dict_test_r2[str(self.transform_names[t])] = self.all_test_r2[t]        
        df_test_r2 = pd.DataFrame.from_dict(dict_test_r2, 
                                              orient='index',
                                              columns=self.models)        
        self.df_test_r2 = df_test_r2.copy()        
        df_test_r2.to_csv(f"{self.path}/df_test_r2_matrix.csv", float_format='%.3f')
        
        print("Test R2", self.df_test_rmse)
        
        dict_train_r2 = {}        
        for t, transform in enumerate(self.feature_transformations):            
            dict_train_r2[str(self.transform_names[t])] = self.all_train_r2[t]        
        df_train_r2 = pd.DataFrame.from_dict(dict_train_r2, 
                                              orient='index',
                                              columns=self.models)        
        self.df_train_r2 = df_train_r2.copy()        
        df_train_r2.to_csv(f"{self.path}/df_train_r2_matrix.csv", float_format='%.3f')
        
        #print("Train RMSLE", self.df_train_rmsle)

    def calculate_error(self):
        
#        y_test_all = [[self.y_test for i in np.zeros_like(self.all_test_predictions)]
        # to work on: make 3d array with y_test for easier absolute difference to get errors

        self.all_test_errors = [[np.abs(self.y_test - self.all_test_predictions[t][i]) for i in range(len(self.models))] for t in range(len(self.feature_transformations))]
        self.all_train_errors = [[np.abs(self.y_train - self.all_train_predictions[t][i]) for i in range(len(self.models))] for t in range(len(self.feature_transformations))]
        
        self.all_test_performances =  [[np.mean(self.all_test_errors[t][i]) for i in range(len(self.models))] for t in range(len(self.feature_transformations))]
        self.all_train_performances = [[np.mean(self.all_train_errors[t][i]) for i in range(len(self.models))] for t in range(len(self.feature_transformations))]

    def rmsle(self, y_true, y_pred):
        sum_ = np.sum( np.log((y_pred + 1)/(y_true + 1))**2 )
        return np.sqrt((1/len(y_true)) * sum_ )
    
    def save_error(self): 
        # save quantitative performances and errors and percent error 
        
        self.all_df_errors = []
        
        for t in range(len(self.feature_transformations)):
            
            transform_df_errors = []
            for i, model in enumerate(self.models):
                
                model_path = self.model_paths[t][i]
        
                # train df
                df_train = self.df_y_train.copy()
                df_train["Predicted"] = self.all_train_predictions[t][i]
                df_train["Absolute error"] = self.all_train_errors[t][i]
                df_train["Train/Test"] = "Train"
                                
                # test df
                df_test = self.df_y_test.copy()
                df_test["Predicted"] = self.all_test_predictions[t][i]
                df_test["Absolute error"] = self.all_test_errors[t][i]
                df_test["Train/Test"] = "Test"
                
                # merge train and test
                df_error = pd.concat([df_train, df_test])
                df_error["Percent error"] = 100*(df_error["Absolute error"].astype(float))/df_error[str(self.target)].astype(float)
                # df_error["RMSLE"] = [rmsle(df_error[str(self.target)].astype(float), pred) for ]
                
                # add id columns
                df_error = df_error.join(self.df_id)
                
                # save in model path folder
                df_error.to_csv(f"{model_path}/df_error.csv", float_format='%.3f')
                
                transform_df_errors.append(df_error)
            
            self.all_df_errors.append(transform_df_errors)
            
    def save_test_performance_df(self):
        
        dict_test_perf = {}
        for t, transform in enumerate(self.feature_transformations):            
            dict_test_perf[str(self.transform_names[t])] = self.all_test_performances[t]
        
        df_test_perf = pd.DataFrame.from_dict(dict_test_perf, 
                                              orient='index',
                                              columns=self.models)        
        self.df_test_perf = df_test_perf.copy()        
        df_test_perf.to_csv(f"{self.path}/df_test_MAE_matrix.csv", float_format='%.3f')
        
        # print("Test MAE", self.df_test_perf)
        
        dict_train_perf = {}        
        for t, transform in enumerate(self.feature_transformations):            
            dict_train_perf[str(self.transform_names[t])] = self.all_train_performances[t]
        
        df_train_perf = pd.DataFrame.from_dict(dict_train_perf, 
                                              orient='index',
                                              columns=self.models)        
        self.df_train_perf = df_train_perf.copy()        
        df_train_perf.to_csv(f"{self.path}/df_train_MAE_matrix.csv", float_format='%.3f')
        
        #print("Train MAE", self.df_train_perf)
        
        # cross validation perf saved
        dict_cv = {}        
        for t, transform in enumerate(self.feature_transformations):            
            dict_cv[str(self.transform_names[t])] = -1.0 * self.all_cv_scores[t]
        
        df_cv = pd.DataFrame.from_dict(dict_cv, 
                                              orient='index',
                                              columns=self.models)        
        self.df_cv = df_cv.copy()        
        df_cv.to_csv(f"{self.path}/df_CV_score_{self.scoring}_matrix.csv", float_format='%.3f')
        
        # print("CV score", df_cv)
        
        return None
    
    def corr_features(self, df, corr_type="pearson", abs_=True):
        
        if abs_ == True:
            corr_matrix = df.corr(method=str(corr_type)).abs()
        else:
            corr_matrix = df.corr(method=str(corr_type))

        corrs = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
                  .stack()
                  .sort_values(ascending=False))
        # print(corrs) # can see greatest abs corr is removed each time in while loop
        
        return corrs
    
    def get_all_reduced_features(self, max_corr, corr_type="pearson", plot_corr=False):
        
        for t in range(len(self.feature_transformations)):
            
            label = f"{self.transform_names[t]}"
            
            if len(self.df_trains_transformed[t].columns) < 300:
                
                red_ftrs, corr_pairs = self.reduced_features_rm_corr(self.df_trains_transformed[t], max_corr, corr_type = corr_type)
            
                self.write_feature_list(red_ftrs, f"{self.data_plot_path}/{label}_reduced_ftrs_{corr_type}_max_{max_corr*100}.txt")
            
            # red_ftrs.to_csv(f"{self.data_plot_path}/{label}_sorted_corr_reduced_ftrs_{corr_type}_max_{max_corr*100}.csv", float_format='%.3f')
            
                # plot correlated pairs
                if plot_corr == True:
    
                    for corr_pair in corr_pairs:
                        
                        # plot correlation
                        self.plot_ftr_ftr(int(corr_pair[0]), int(corr_pair[1]))
            
        return None
    
    def reduced_features_rm_corr(self, df, max_corr, corr_type = "pearson"):
        
        df = df.apply(pd.to_numeric, errors="ignore")
        
        # remove target from df
        df = df.drop(str(self.target), axis=1)
        
        corrs = self.corr_features(df, corr_type=corr_type)
        
        corr_pairs = []
        
        while float(corrs[0]) >= float(max_corr):
            
            # get smaller wavenumber from dataframe, get corrs again
            left_ftr = float(corrs.iloc[[0]].index[0][0])
            right_ftr = float(corrs.iloc[[0]].index[0][1])
            less_ftr = min(left_ftr, right_ftr)
            
            # print(less_ftr)
            # print(type(less_ftr))
            # print(df.columns)
            # print(type(df.columns))
            
            # plot correlation
            corr_pairs.append([int(left_ftr), int(right_ftr)])
            
            # remove feature
            try:
                df = df.drop(less_ftr, axis=1)
            except:
                df = df.drop(str(int(less_ftr)), axis=1)
            
            # check new correlations
            corrs = self.corr_features(df, corr_type=corr_type)
            
        reduced_ftrs = df.columns.values.tolist()
        
        return reduced_ftrs, corr_pairs
    
    def write_feature_list(self, features, path):
        # save list of features to copy to another input file
        with open(f"{path}_list.txt", "w") as f:
            for idx, ftr in enumerate(features):
                f.write(f"'{ftr}', ")
    
    def note_important_features(self):
        # save list of nonzero wts for lasso and elasticnet
        # output RF, adaboost feature importances
        # possibly output tpot feature importances ?
        # or also save in one doc for easy comparison
        
        for t, transform in enumerate(self.feature_transformations):
        
            for i, model in enumerate(self.models):
                
                model_path = self.model_paths[t][i]
                    
                if model in ["Lasso", "ElasticNet", "Ridge", "LinearRegression"]: # uses weights because linear regression
                    
                    tuned_model = self.all_tuned_models[t][i] 
                    coeffs = tuned_model.coef_
                    
                    features = list(self.df_trains_transformed[t].columns)
                    features.remove(self.target)
                    
                    weights = [coeff for coeff in coeffs]   # was absolute                                      
                    sorted_features = [x for _,x in sorted(zip(weights,features), reverse=True)]
                    sorted_weights = sorted(weights, reverse=True)
                    
                    df_weights = pd.DataFrame(list(zip(sorted_features, sorted_weights)), columns=["Feature", "Weight"])
                    df_weights.to_csv(f"{model_path}/feature_weights.csv", float_format='%.3f')

                    self.feature_importance(weights, features, model_path)
                    
                    # save list of features to copy to another input file
                    with open(f"{model_path}/feature_list.txt", "w") as f:
                        for idx, ftr in enumerate(features):
                            if float(weights[idx]) != 0.0:
                                f.write(f"'{ftr}', ")
                
                elif model in ["RandomForest", "AdaBoost", "DecisionTree"]: # uses feature importances bc tree-based algorithm
                    
                    tuned_model = self.all_tuned_models[t][i]
                    importances = tuned_model.feature_importances_
                    
                    features = list(self.df_trains_transformed[t].columns)
                    features.remove(self.target)
                    
                    sorted_features = [x for _,x in sorted(zip(importances,features), reverse=True)]
                    sorted_importances = sorted(importances, reverse=True)
                    
                    df_importances = pd.DataFrame(list(zip(sorted_features, sorted_importances)), columns=["Feature", "Importance"])
                    df_importances.to_csv(f"{model_path}/feature_importances.csv", float_format='%.3f')
                    
                    self.feature_importance(importances, features, model_path)
                    
                    # save list of features to copy to another input file
                    with open(f"{model_path}/feature_list.txt", "w") as f:
                        for idx, ftr in enumerate(features):
                            if float(importances[idx]) != 0.0:
                                f.write(f"'{ftr}', ")
                    
                
    def plot_performances(self):
    
        for t, transform in enumerate(self.feature_transformations):
            
            self.box_performances_sns(t,
                                      by="Algorithm")
            
            self.box_performances_sns(t,
                                      by="Algorithm",
                                      plot_type="bar")
            
            self.bar_CV_test(t, 
                              by="Algorithm")
            
            # turn on for violin plots as well
            # self.box_performances_sns(t
            #                           by="Algorithm", 
            #                           plot_type="violinsplit")
    
            for i in range(len(self.models)):
                
                model_path = self.model_paths[t][i]
                                
                # self.pres_parity_plot(self.all_test_predictions[t][i],
                #                       self.all_train_predictions[t][i],
                #                         model_path)
                
                self.parity_plot(self.all_test_predictions[t][i],
                                 self.all_train_predictions[t][i], 
                                 model_path)
                
                # plot parity plots for specific fuels
                for additive in self.blend_additives:
                    
                    df_error = self.all_df_errors[t][i]
                    index = df_error.index.values
                    examples_to_keep = [example for example in index if additive in example]

                    df_error_blend = df_error.loc[examples_to_keep]
                    
                    self.blends_parity_plot(df_error_blend,
                                            model_path,
                                            additive
                                            )

        for i in range(len(self.models)):
            
            self.box_performances_sns(i, by="Transform")
            
            self.box_performances_sns(i,
                                      by="Transform",
                                      plot_type="bar")
            
            self.bar_CV_test(i, 
                             by="Transform")

            # turn on for violin plots as well
            # self.box_performances_sns(i,
            #                           by="Transform",
            #                           plot_type="violinsplit")
            
            
            
            
            
            
#
