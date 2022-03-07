import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import yaml
from datetime import datetime
import os
import shutil
import seaborn as sns
# from tpot import TPOTRegressor
import pickle
from pickle import dump
from scipy.signal import find_peaks

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn import metrics

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeRegressor
from dtreeviz.trees import dtreeviz

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge

from mpl_toolkits.mplot3d import Axes3D
import collections
import multiprocessing
from multiprocessing import Pool


class Train:
    
    def tune_model(self, model, X_train_):
        # print(model)
        if model in ["Lasso"]:
                    
            parameters_ = [self.parameters["Lasso"]]
            best_model = GridSearchCV(Lasso(), parameters_, cv=self.kf, scoring = self.scoring) # test that indices match
            best_model.fit(X_train_, self.y_train)    
            tuned_model = best_model.best_estimator_
            cv_score = best_model.best_score_
            
        elif model in ["MLPRegressor"]:
            
            best_model = MLPRegressor(random_state=1, max_iter=500)
            # best_model.fit(X_train, y_train)
            tuned_model = best_model.fit(X_train_, self.y_train)
            cv_score = np.nan
            #cv_score = best_model.best_score_
            
        elif model in ["DecisionTree"]:
            
            # parameters_ = [self.parameters["DecisionTree"]]
            # best_model = GridSearchCV(DecisionTreeRegressor(), parameters_, cv=self.kf) # test that indices match
            # best_model.fit(X_train_, self.y_train)
            # tuned_model = best_model.best_model.best_estimator_
            
            parameters_ = self.parameters["DecisionTree"]["parameters"]
            n_iter_dc = self.parameters["DecisionTree"]["n_iter"]
            
            for param in parameters_: # change lists to arrays in parameter grid
                if type(parameters_[str(param)]) == list:
                    # change to array
                    try:
                        parameters_[str(param)] = np.asarray(parameters_[str(param)])
                    except:
                        pass
                else:
                    pass
            
            best_model = RandomizedSearchCV(estimator = DecisionTreeRegressor(), 
                                            param_distributions = parameters_, 
                                            n_iter = n_iter_dc, 
                                            cv = self.kf, scoring = self.scoring,
                                            verbose=2, 
                                            random_state=self.seed, 
                                            n_jobs = -1, 
                                            refit=True)
            best_model.fit(X_train_, self.y_train)
            tuned_model = best_model.best_estimator_
            cv_score = best_model.best_score_
            
        elif model in ["KNN"]:
            
            parameters_ = [self.parameters["KNN"]]
            best_model = GridSearchCV(KNeighborsRegressor(), parameters_, cv=self.kf, scoring = self.scoring
                                      ) # test that indices match; scoring="accuracy"
            best_model.fit(X_train_, self.y_train) 
            tuned_model = best_model.best_estimator_
            cv_score = best_model.best_score_
            
        elif model in ["ElasticNet"]:
            
            parameters_ = [self.parameters["ElasticNet"]]
            best_model = GridSearchCV(ElasticNet(), parameters_, cv=self.kf, scoring = self.scoring) # test that indices match
            best_model.fit(X_train_, self.y_train)    
            tuned_model = best_model.best_estimator_
            cv_score = best_model.best_score_
            
        elif model in ["Ridge"]:
            
            parameters_ = [self.parameters["Ridge"]]
            best_model = GridSearchCV(Ridge(), parameters_, cv=self.kf, scoring = self.scoring) # test that indices match
            best_model.fit(X_train_, self.y_train)    
            tuned_model = best_model.best_estimator_
            cv_score = best_model.best_score_
            
        elif model in ["RandomForest"]:
        
            parameters_ = self.parameters["RandomForest"]["parameters"]
            n_iter_rf = self.parameters["RandomForest"]["n_iter"]
            
            for param in parameters_: # change lists to arrays in parameter grid
                if type(parameters_[str(param)]) == list:
                    # change to array
                    try:
                        parameters_[str(param)] = np.asarray(parameters_[str(param)])
                    except:
                        pass
                else:
                    pass
            
            best_model = RandomizedSearchCV(estimator = RandomForestRegressor(), 
                                            param_distributions = parameters_, 
                                            n_iter = n_iter_rf, 
                                            cv = self.kf, scoring = self.scoring,
                                            verbose=0, 
                                            random_state=self.seed, 
                                            n_jobs = -1, 
                                            refit=True)
            best_model.fit(X_train_, self.y_train)
            tuned_model = best_model.best_estimator_
            cv_score = best_model.best_score_
            
            # plot_tree(tuned_model.estimators_[0], 
            #           feature_names=X_train_,
            #           class_names=wine.target_names, 
            #           filled=True, impurity=True, 
            #           rounded=True) 
            
        elif model in ["AdaBoost"]:
            
            parameters_ = self.parameters["AdaBoost"]["parameters"]
            n_iter_ab = self.parameters["AdaBoost"]["n_iter"]
            
            for param in parameters_: # change lists to arrays in parameter grid
                if type(parameters_[str(param)]) == list:
                    # change to array
                    try:
                        parameters_[str(param)] = np.asarray(parameters_[str(param)])
                    except:
                        pass
                else:
                    pass
            
            best_model = RandomizedSearchCV(estimator = AdaBoostRegressor(), 
                                            param_distributions = parameters_, 
                                            n_iter = n_iter_ab, 
                                            cv = self.kf, scoring = self.scoring, 
                                            verbose=0, 
                                            random_state=self.seed, 
                                            n_jobs = -1, 
                                            refit=True)
            
            best_model.fit(X_train_, self.y_train)
            tuned_model = best_model.best_estimator_
            cv_score = best_model.best_score_
            
        elif model in ["PLS"]:
            
            parameters_ = [self.parameters["PLS"]]
            
            best_model = GridSearchCV(PLSRegression(), parameters_, cv=self.kf, scoring = self.scoring) # test that indices match
            best_model.fit(X_train_, self.y_train)    
            tuned_model = best_model.best_estimator_
            cv_score = best_model.best_score_
            
        elif model in ["LinearRegression"]:
            
            tuned_model = LinearRegression()
            tuned_model.fit(X_train_, self.y_train) 
            
            cv_scores = cross_val_score(tuned_model, X_train_, self.y_train, 
                                       cv=self.kf, 
                                       scoring = self.scoring
                                       )
            print(cv_scores)
            cv_score = np.mean(cv_scores)
            
        # elif model in ["TPOT", "tpot"]:
                            
        #     tuned_model = TPOTRegressor(generations=self.parameters["TPOT"]["generations"], 
        #                          population_size=self.parameters["TPOT"]["population_size"], 
        #                          verbosity=self.parameters["TPOT"]["verbosity"], 
        #                          random_state=self.seed, 
        #                          max_time_mins = self.parameters["TPOT"]["max_time_mins"], 
        #                          n_jobs=self.parameters["TPOT"]["n_jobs"])
            
        #    # tuned_model.fit(X_train_, self.y_train)                
        #    # tuned_model.export(f'{model_path}/{model}_pipeline.py')
        #     cv_score = np.nan

        elif model in ["Baseline_Average"]:
            
            tuned_model = DummyRegressor(strategy="mean")
            
            cv_scores = cross_val_score(tuned_model, X_train_, self.y_train, 
                                       cv=self.kf, 
                                       scoring = self.scoring
                                       )
            cv_score = np.mean(cv_scores)
            
        elif type(model) == dict: # expand to compare multiple trained models
            
            cv_score = np.nan    
            # try:
                # add file extension flexibility
            try:
                tuned_model = joblib.load(model["pretrained"]["filename"])
            except:
                tuned_model = pickle.load(open(model["pretrained"]["filename"]), "rb")
            # print(model["pretrained"]["filename"])
            # tuned_model = pickle.load(open(model["pretrained"]["filename"], 'rb'))
            # except:
            #     # exchange with other simple model and replace in list?
            #     pass
            
        else: 
            model_error = f'{model} model not supported'
            self.output_comments.append(model_error)
            raise NameError(model_error)
            tuned_model = None
        # print(model)
        
        return tuned_model, cv_score


    def tune_train_all_models(self):
        
        self.all_tuned_models = np.zeros((len(self.feature_transformations), len(self.models)), dtype=object)
        self.all_test_predictions = np.zeros((len(self.feature_transformations), len(self.models), len(self.y_test)))
        self.all_train_predictions = np.zeros((len(self.feature_transformations), len(self.models), len(self.y_train)))
        self.all_cv_scores = np.zeros((len(self.feature_transformations), len(self.models)), dtype=object)
        
        for t in range(len(self.feature_transformations)):
            
            X_train_ = self.X_trains_transformed[t]
            X_test_ = self.X_tests_transformed[t]
            
            for i, model in enumerate(self.models):
                
                model_path = self.model_paths[t][i]
                
                tuned_model, cv_score = self.tune_model(model, X_train_)
                
                # if model == "RandomForest":
                    
                    # for idx, estimator in enumerate(tuned_model.estimators_):
    
                        # plt.figure(figsize=(15, 10))
                        
                        # plot_tree(estimator, 
                        #       feature_names=self.df_trains_transformed[t].columns,
                        #       class_names=self.y_train, 
                        #       filled=True, impurity=True, 
                        #       rounded=True)
                        
                        # plt.savefig(f"{model_path}/RF_estimator{idx}.png")
                        
                      #  viz = dtreeviz(estimator.estimators_[idx], 
                       #                self.X_trains_transformed[t], 
                         #              self.y_train,
                         #  feature_names=self.df_trains_transformed[t].columns,
                         #  title=f"{idx} decision tree")
                        
                # elif model == "AdaBoost":
                    
                #     for idx, estimator in enumerate(tuned_model.estimators_):
    
                #         plt.figure(figsize=(15, 10))
                        
                #         plot_tree(estimator, 
                #               feature_names=self.df_trains_transformed[t].columns,
                #               class_names=self.y_train, 
                #               filled=True, impurity=True, 
                #               rounded=True)
                        
                #         plt.savefig(f"{model_path}/AB_estimator{idx}_weight{tuned_model.estimator_weights_[idx]: .2f}.png")
                
                # save model object written into text to see hyperparameters
                with open(f"{model_path}/model_object_parameters.txt", "w") as f:
                    f.write(f"{tuned_model}")
                
                if type(model) != dict:
                    
                    try: 
                        tuned_model.fit(X_train_, self.y_train)                
                        
                        test_predictions_i = tuned_model.predict(X_test_)
                        train_predictions_i = tuned_model.predict(X_train_)
    
                        if model in ["TPOT", "tpot"]:
                            tuned_model.export(f'{model_path}/{model}_pipeline.py')
                        else:
                       	 # save the model
                            dump(tuned_model, open(f'{model_path}/tuned_trained_model.pkl', 'wb'))
                                            
                    except:
                        model_error = f"Could not tune and train {model} model"
                        self.output_comments.append(model_error)
                        raise ValueError(model_error)
                        pass # add placeholder for undefined model in list ?
                        
                else: # pretrained models loaded in
                    # print(tuned_model)
                    test_predictions_i = tuned_model.predict(X_test_)
                    train_predictions_i = tuned_model.predict(X_train_)
                    
                self.all_tuned_models[t][i] = tuned_model
                self.all_cv_scores[t][i] = cv_score
                self.all_test_predictions[t][i] = test_predictions_i
                self.all_train_predictions[t][i] = train_predictions_i
                
                # save the respective feature transform scaler
                if type(self.scaler_objects[t]) != str:
                    dump(self.scaler_objects[t], open(f'{model_path}/scaler.pkl', 'wb'))
#                except:
#                    print("No scaler") # find way to save homemade scalers

                # erase tuned model just in case
                tuned_model = None
                train_predictions_i , test_predictions_i = None, None