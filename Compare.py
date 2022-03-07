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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn.model_selection import cross_val_score

from mpl_toolkits.mplot3d import Axes3D
import collections
import multiprocessing as mp

degree_sign= u'\N{DEGREE SIGN}'

# supress sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# modules
import clean
import feature_engineering
import train
import plot
import analysis
import uncertainty

class Compare(clean.Clean,              
                feature_engineering.FeatureEngineering, 
                train.Train,               
                plot.Plot,
                analysis.Analysis,
                uncertainty.Uncertainty):
    
    def __init__(self, input_file):
        
        self.input_file = "input_files/" + input_file
        
        # unique ID is datetime
        self.id_num = str(datetime.now())[2:19]
        self.id_num = self.id_num.replace(':', '_').replace(' ', '_') #.replace('-', '')
        
        self.read_input()
        self.make_paths()
        self.load_dataframe()
        
        self.labels = {"fp_c": {"units": f"{degree_sign}C", "name":"Flash point"},
                            "mp_c":  {"units": f"{degree_sign}C", "name":"Freezing point"},
                            "cn":  {"units": "", "name": "Cetane number"},
                            "MW":  {"units": "g/mol", "name":"Molecular weight"},
                            "wv":  {"units": r"$cm^{-1}$", "name":"wavenumber"},
                            "HC": {"units": "" , "name": "H/C ratio"}
                                }
        
        self.name = self.labels[str(self.target)]["name"]
        self.units = self.labels[str(self.target)]["units"]
        
    def make_paths(self):
        
        # make directory for case request analysis if it doesn't already exist
        try:
            os.mkdir("comparisons")
        except:
            pass
        
        # define the name of the directory to be created
        self.path = "comparisons/comparison_" + str(self.id_num) + "_" + str(self.target)+"_"+self.split_col
        
        # make directory for this case
        try:
            os.mkdir(self.path)
        except OSError:
            print("Creation of the directory %s failed" % self.path)
            
        # folder for data plots
        self.data_plot_path = f"{self.path}/data_plots"
        os.mkdir(self.data_plot_path)
        os.mkdir(f"{self.path}/performance_plots")
        
        # put copy of input request file into output folder    
        shutil.copy(self.input_file, self.path) 
        
        # Uncertainty methods
        try:
            self.uncertainty = self.input_["uncertainty"]
            self.uncertainty_path = f"{self.path}/uncertainty"
            os.mkdir(self.uncertainty_path)
            os.mkdir(f"{self.uncertainty_path}/UQ_df")
            os.mkdir(f"{self.uncertainty_path}/UQ_parity_bounds")
        except:
            pass
        
        # put copy of parameter file into output folder    
        shutil.copy(self.parameters_file, self.path) 
        
        # make path folders for model+transform combinations
        self.mkdir_model_paths()
        
    def read_input(self):
        
        # load input file with yaml
        with open(str(self.input_file), 'r') as file:
            input_ = yaml.load(file, Loader=yaml.FullLoader)
            
        self.input_ = input_
        
        # dictionaries for each phase of ML process
        self.data = input_["data"]
        self.cleaning = input_["cleaning"]
        self.validation = input_["validation"]
        self.parameters_file = "hyperparameter_searches/"+input_["validation"]["parameters_file"]
        
        # transforms and models
        self.feature_transformations = input_["transformations"]["feature_transformations"]
        self.transform_names = input_["transformations"]["transform_names"]
        self.models = input_["models"]
        
        # name of column to split data with
        self.split_col = input_["validation"]["split"]["split_col"]        
        
        # plot quality in dpi
        try:
            self.dpi = self.data["plot_quality_dpi"]
        except:
            self.dpi = 1200
        
        # load hyperparameter search settings file with yaml
        with open(str(self.parameters_file), 'r') as file:
            self.parameters = yaml.load(file, Loader=yaml.FullLoader)

        self.target = self.data["target_col_name"]
        if self.target == "H/C":
            self.target = "HC"
        
        self.seed = self.validation["random_seed"]
        
        try:
            self.blend_additives = self.data["blend_additives"]
        except:
            pass
        
        try:
            self.export_filtered = self.data["export_filtered"]
        except:
            pass
        
        self.output_comments = [] # error messages / modifications to output to file later, list of strings
        
    def load_dataframe(self):

        # dataframe with ID, target, features 
        self.df_load_features = pd.read_csv("data/"+self.data["feature_path"], 
                              header = 0, 
                              dtype=object, 
                              index_col=str(self.data["index_col_name"])) 
        
        self.df_load_id = pd.read_csv("data/"+self.data["id_path"], 
                              header = 0, 
                              dtype=object, 
                              index_col=str(self.data["index_col_name"]))
        
        self.df_load_split = pd.read_csv("data/"+self.data["split_path"], 
                              header = 0, 
                              dtype=object, 
                              index_col=str(self.data["index_col_name"]))
        
        # remove excluded examples, labeled 2 in split_col
        self.df_load_features = self.df_load_features[self.df_load_split[self.split_col].astype(int) <= 1]
        self.df_load_id = self.df_load_id[self.df_load_split[self.split_col].astype(int) <= 1]
        self.df_load_split = self.df_load_split[self.df_load_split[self.split_col].astype(int) <= 1]
        
        #rename for HC issue
        self.df_load_features.rename({"H/C": "HC"}, axis='columns', inplace=True) 
        self.df_load_id.rename({"H/C": "HC"}, axis='columns', inplace=True) 
        self.df_load_split.rename({"H/C": "HC"}, axis='columns', inplace=True) 
        
        # combine features with target df
        self.df_target = self.df_load_id[str(self.target)].to_frame()
        self.df = self.df_target.join(self.df_load_features)
        
        # do cleaning with features and target
        self.df = self.pre_split_cleaning(self.df)
        
        # dataframe of ID col info
        #self.df_id = self.df_load[self.id_cols]
        self.df_id = self.df_load_id.loc[list(self.df.index)]
        self.df_id = self.df_id.drop(str(self.target), axis=1)
        self.df_id = self.df_id.join(self.df_load_split)
        
        # get list of features
        self.features = list(self.df.columns)
        self.features.remove(self.target)

    def split(self):
        
        if self.validation["split"] == "random":
            
            self.df_train, self.df_test = train_test_split(self.df, 
                                            test_size=self.validation["holdout_fraction"], 
                                            random_state=self.seed
                                            )
            
        elif "split_col" in self.validation["split"]:
            
            self.split_col = self.validation["split"]["split_col"]
            
            self.ids = self.df_id[self.split_col].values
            self.ids_train = [int(i) == 0 for i in self.ids]
            self.ids_test = [int(i) == 1 for i in self.ids]
            
            self.df_train = self.df[self.ids_train]
            self.df_test = self.df[self.ids_test]
            
        self.X = self.df.drop(str(self.target), axis=1).values.astype(np.float)
        self.y = self.df[str(self.target)].copy().values.astype(np.float)
        
        self.X_train = self.df_train.drop(str(self.target), axis=1).values.astype(np.float)
        self.y_train = self.df_train[str(self.target)].copy().values.astype(np.float)
        
        self.X_test = self.df_test.drop(str(self.target), axis=1).values.astype(np.float)
        self.y_test = self.df_test[str(self.target)].copy().values.astype(np.float)
        
        self.df_y_train = self.df_train[[str(self.target)]]
        self.df_y_test = self.df_test[[str(self.target)]]

        # K-fold splitting
        if "Kfold" in self.validation:
            
            try:
                self.K = self.validation["Kfold"]["nsplits"]
            except ValueError:
                print("nsplits for K-fold validation not given")
            
            self.kf = KFold(n_splits = self.K, shuffle=True, random_state=self.seed)
            
            try:
                scoring = self.validation["Kfold"]["scoring"]
                if scoring == "RMSE":
                    self.scoring = "neg_root_mean_squared_error"
                elif scoring == "RMSLE":
                    self.scoring = "neg_mean_squared_log_error"
                elif scoring == "MAE":
                    self.scoring = "neg_mean_absolute_error"
                else: # RMSE default
                    self.scoring = "neg_root_mean_squared_error"
            except:
                self.scoring = "neg_root_mean_squared_error"
            
        # get lists for train and test indicies
        self.index_list_train = list(self.df_train.index.values)
        self.index_list_test = list(self.df_test.index.values)

    def mkdir_model_paths(self):
        
        self.model_paths = np.zeros((len(self.feature_transformations), len(self.models)), dtype=object)
        os.mkdir(f"{self.path}/models")
        
        for t in range(len(self.feature_transformations)):
        
            for i, model in enumerate(self.models):
                    
                # pretrained is dict type so save path differently
                if type(model) == dict: # call model by the joblib model number instead of i
                    model_path = f"{self.path}/models/pretrained{i}_{self.transform_names[t]}" # make into dictionary
                else:
                    model_path = f"{self.path}/models/{model}_{self.transform_names[t]}" # make into dictionary
                    
                os.mkdir(model_path)
                self.model_paths[t][i] = model_path
        
    def plot_data(self):
        
        if 'overall distribution' in self.data["plot"]:     
            # self.overall_dist()
            self.overall_dist_presentation()
            
        if 'train distribution' in self.data["plot"]:            
            self.set_dist(self.df_train, 'train')
            
        if 'test distribution' in self.data["plot"]:            
            self.set_dist(self.df_test, 'test')
            
        if "Kfold distribution" in self.data["plot"]:

            if "Kfold" not in self.validation:
                plot_error = "Cannot plot K-fold distribution if not using K-fold CV in validation"
                self.output_comments.append(plot_error)
                print(plot_error)
                
            else:
                self.kfold_dist()
           
        if "3D spectra" in self.data["plot"]:            
            self.spectra_3D()
            
        if "2D spectra" in self.data["plot"]:       

            self.spectra_2D(self.df_train, "train")  
            self.spectra_2D(self.df_test, "test")
            self.spectra_2D(self.df, "all")

        if "all spectra" in self.data["plot"]:            
            self.all_spectra()            
        else:
            pass
        
    def transform_save(self, transformation, i):
        
        df_train_transformed, df_test_transformed, scaler = self.transform(transformation)
                
        X_train_transformed = df_train_transformed.drop(str(self.target), axis=1).values.astype(np.float)
        X_test_transformed = df_test_transformed.drop(str(self.target), axis=1).values.astype(np.float)

        self.scaler_objects[i] = scaler
        
        self.df_trains_transformed[i] = df_train_transformed
        self.df_tests_transformed[i] = df_test_transformed
        
        self.X_trains_transformed[i] = X_train_transformed
        self.X_tests_transformed[i] = X_test_transformed

        X_train_transformed, X_test_transformed, scaler = None, None, None
        df_train_transformed, df_test_transformed = None, None
        
        return None
    
    def do_all_transformations(self, parallel=False):
        # do all feature transforms in list, save in list of dataframes
        
        self.df_trains_transformed = [None] * len(self.feature_transformations)
        self.df_tests_transformed = [None] * len(self.feature_transformations)
        
        self.X_trains_transformed = [None] * len(self.feature_transformations)
        self.X_tests_transformed = [None] * len(self.feature_transformations)
        
        self.scaler_objects = [None] * len(self.feature_transformations)  # save scaler objects to output
        
        if parallel == True: # broken at the moment. df_trains, etc are empty
            
            pool = mp.Pool(mp.cpu_count())
            
            _ = [pool.apply(self.transform_save, args=(transformation, t)) for t, transformation in enumerate(self.feature_transformations)]

            pool.close()
            
        else:
        
            for i, transformation in enumerate(self.feature_transformations):
                
                self.transform_save(transformation, i)
                