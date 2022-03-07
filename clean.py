import numpy as np
import pandas as pd

class Clean:
    
    def pre_split_cleaning(self, df):
        
        # drop range of wavenumbers
        try:
            if self.cleaning["pre_split_cleaning"]["remove_range"]:
                ranges = self.cleaning["pre_split_cleaning"]["remove_range"]
                for range_ in ranges:
                    #wv_range_remove = self.cleaning["pre_split_cleaning"]["remove_range"]
                    df = self.remove_wv_range(df, range_)
        except:
            pass
        
        # baseline correction
        # try:
        if self.cleaning["pre_split_cleaning"]["baseline_correction"] == True:
            df = self.baseline_correction(df)
        # except:
        #     print("Baseline correction not done")
        #     pass

        # if there is a missing value- feature or target
        if df.isnull().values.any() == True: 
            
            # add: remove example if target is missing
        
            if self.cleaning["pre_split_cleaning"]["nan"] == "remove example":
                presplit_cleaned_df = df.dropna(how='any') # double check this doesn't affect self.df
                
            elif self.cleaning["pre_split_cleaning"]["nan"] == "remove feature":
                presplit_cleaned_df = df.dropna(axis='columns')
            
            # Later: add other options for thresholds
            else:
                # use imputer function from sklearn
                strategy = self.cleaning["pre_split_cleaning"]["nan"] # show strategy options
                # strategies: mean
                try: # fix imputer. doesn't go to exception
                    presplit_cleaned_df = df.copy()
                    # imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
                    # self.presplit_cleaned_df = imputer.fit_transform(self.df)
                except:
                    imputer_error = f"pre-split cleaning method {strategy} not valid"
                    self.output_comments.append(imputer_error)
                    print(imputer_error)

        else: # no missing values or cleaning requested
            presplit_cleaned_df = df.copy()
            
        return presplit_cleaned_df

    def post_split_cleaning(self):

        if self.cleaning["post_split_cleaning"]['normalize'] == "overall min max":        
            self.df_train, self.df_test, self.scale = self.normalize_overall_min_max()

        elif self.cleaning["post_split_cleaning"]['normalize'] == "min max":        
            self.df_train, self.df_test, self.scale = self.normalize_scaler(MinMaxScaler())

        elif self.cleaning["post_split_cleaning"]['normalize'] == "standard scaler":        
            self.df_train, self.df_test, self.scale = self.normalize_scaler(StandardScaler())
            
    def baseline_correction(self, df):
        # subtracts minimum absorbance in a spectraum from all of the absorbances in a spectrum
        # so the minimum absorbance from each spectra is zero.
        
        features = list(df.columns)
        features.remove(self.target)
        
        X = df.drop(str(self.target), axis=1).values.astype(np.float)
        index_list= list(df.index.values)        
        corrected_spec = np.array(X)
        
        for s, spectra in enumerate(X):
        
            min_abs = min(spectra)
            spectra_bc = spectra - min_abs
            
            corrected_spec[s] = spectra_bc
            
        df_x = pd.DataFrame(data = corrected_spec, 
                                     columns = features, 
                                     index = index_list)
        
        df_bc = pd.merge(df[[str(self.target)]],
                         df_x, 
                         how='outer', 
                         left_index=True, 
                         right_index=True)
        
        return df_bc
        
        
    def normalize_overall_min_max(self):
        ''' This function takes the train and test dataframes and 
        normalizes all values by the overall min and max
        ----------
        '''
        min_ = np.amin(self.X_train, axis=None, out=None)
        max_ = np.amax(self.X_train, axis=None, out=None)
        scale = {"min": min_, "max": max_}
        
        X_train_norm = (self.X_train - min_) / (max_ - min_)
        X_test_norm = (self.X_test - min_) / (max_ - min_)
        
        df_x_train = pd.DataFrame(data = X_train_norm
                 , columns = self.features, index = self.index_list_train)
        df_x_test = pd.DataFrame(data = X_test_norm
                 , columns = self.features, index = self.index_list_test)
        
        df_train_norm = pd.merge(self.df_y_train, 
                                    df_x_train, 
                                    how='outer', 
                                    left_index=True, 
                                    right_index=True)
        df_test_norm = pd.merge(self.df_y_test, 
                                df_x_test,how='outer', 
                                left_index=True, 
                                right_index=True)
        
        return df_train_norm, df_test_norm, scale
    
    def normalize_scaler(self, scale_method):
        ''' This function takes the train and test dataframes and
        normalizes each individual feature by its mean and standard deviation.
        Scale method may be MinMaxScaler or StandardScaler
        ----------
            scale_method : sklearn object
                StandardScaler() or MinMaxScaler()
        '''
    
        X_train_norm = scale_method.fit_transform(self.X_train)
        X_test_norm = scale_method.transform(self.X_test)
        
        df_x_train = pd.DataFrame(data = X_train_norm
                 , columns = self.features, index = self.index_list_train)
        df_x_test = pd.DataFrame(data = X_test_norm
                 , columns = self.features, index = self.index_list_test)
        
        df_train_norm = pd.merge(self.df_y_train, df_x_train, 
                                 how='outer', left_index=True, right_index=True)
        df_test_norm = pd.merge(self.df_y_test, df_x_test, 
                                how='outer', left_index=True, right_index=True)
        
        return df_train_norm, df_test_norm, scale_method
    
    def remove_wv_range(self, df, wv_range_remove):
        
        wv_remove = np.arange(float(wv_range_remove[0]), float(wv_range_remove[1]), dtype=int)
        wv_remove = [str(i) for i in wv_remove]
        df_new = df.drop(columns=wv_remove, axis = 1)
        
        return df_new
    
    
    
    

    
    #
