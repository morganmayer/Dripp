from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt
import os

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

import multiprocessing as mp

class FeatureEngineering:
    
    def transform(self, transformation, 
                  X_train = None,
                  X_test = None):
        try:
            a_ = len(X_train)
            b_ = len(X_test)
        except:
            X_train = self.X_train
            X_test = self.X_test
        
        if "Multiple" in transformation: # do several transformations in a row, recursive
            
            df_train_transformed = self.df_train.copy()
            df_test_transformed = self.df_test.copy()
            
            for idx, indiv_transform in enumerate(transformation["Multiple"]): # recursive

                X_train = df_train_transformed.drop(str(self.target), axis=1).values.astype(np.float)
                X_test = df_test_transformed.drop(str(self.target), axis=1).values.astype(np.float)

                df_train_transformed, df_test_transformed, _ = self.transform(indiv_transform, 
                                                                              X_train = X_train, 
                                                                              X_test = X_test)
            scaler = "None"
            #df_train_transformed, df_test_transformed = None, None
            
        elif "pearson_reduced" in transformation:
            
            max_corr = transformation["pearson_reduced"]["max_corr"]
            try:
                plot_corr = transformation["pearson_reduced"]["plot_corr"]
            except:
                plot_corr = False
            
            reduced_features, corr_pairs = self.pearson_reduced_features(max_corr, plot_corr=plot_corr)
            
            # new df_train, df_test with new feature set with select_features
            df_train_transformed = self.select_features(self.df_train, reduced_features)
            df_test_transformed = self.select_features(self.df_test, reduced_features)
            
            # scaler can be reduced_features?
            scaler = reduced_features
            
        elif "PCA" in transformation:
            
            try:
                n_components = transformation["PCA"]["number"]
                if n_components == 'all': # max PCs i min(n features, n datapoints)
                    n_components = min(int(len(self.df_train) - 1), int(len(X_train[0]) - 1))
                    n_components = int(n_components)
            except ValueError:
                pca_error = "Number of PC's not specified"
                self.output_comments.append(pca_error)
                print(pca_error)

            df_train_transformed, df_test_transformed, scaler = self.do_pca(n_components)
            
            try:
                random_remainder = transformation["PCA"]["random_remainder"]
                max_components = min(int(len(self.df_train) - 1), int(len(X_train[0]) - 1))
                
                n_rand = max_components - n_components
    
                for idx in range(int(n_rand)):
                    df_train_transformed[f"Random_{idx}"] = np.random.rand(len(df_train_transformed))
                    df_test_transformed[f"Random_{idx}"] = np.random.rand(len(df_test_transformed))               
            except:
                pass

        elif "local_averaging" in transformation:
            
            try:
                n_points = transformation["local_averaging"]["n_points"]
            except ValueError:
                local_averaging_error = "Number of points to use for local averaging not specified"
                self.output_comments.append(local_averaging_error)
                print(local_averaging_error)
                
            index_list_train = list(self.df_train.index.values) 
                
            df_X_train = pd.DataFrame(data = X_train, 
                                     columns = self.features, index = index_list_train)
        
            df_train = pd.merge(self.df_y_train, 
                                   df_X_train, 
                                   how='outer', 
                                   left_index=True, 
                                   right_index=True)
            
            index_list_test = list(self.df_test.index.values) 
                
            df_X_test = pd.DataFrame(data = X_test, 
                                     columns = self.features, index = index_list_test)
        
            df_test = pd.merge(self.df_y_test, 
                                   df_X_test, 
                                   how='outer', 
                                   left_index=True, 
                                   right_index=True)
                
            df_train_transformed = self.local_averaging(df_train, n_points)
            df_test_transformed = self.local_averaging(df_test, n_points)
            scaler = {"n_points": n_points} # better way to do this for saving/evaluation ?
            
        elif "lowpass_filter" in transformation:
            
            index_list_train = list(self.df_train.index.values) 
                
            df_X_train = pd.DataFrame(data = X_train, 
                                     columns = self.features, index = index_list_train)
        
            df_train = pd.merge(self.df_y_train, 
                                   df_X_train, 
                                   how='outer', 
                                   left_index=True, 
                                   right_index=True)
            
            index_list_test = list(self.df_test.index.values) 
                
            df_X_test = pd.DataFrame(data = X_test, 
                                     columns = self.features, index = index_list_test)
        
            df_test = pd.merge(self.df_y_test, 
                                   df_X_test, 
                                   how='outer', 
                                   left_index=True, 
                                   right_index=True)
                
            df_train_transformed = self.lowpass_filter(df_train)
            df_test_transformed = self.lowpass_filter(df_test)
            
            df_all = pd.concat([df_train_transformed, df_test_transformed])
            df_all.to_csv(f"{self.path}/lowpass_filter_spectra_data.csv",
                          index_label="name",
                          )
            if self.export_filtered == True:
                self.export_spectra(df_all)
            
            scaler = {"lowpass": 0} # better way to do this for saving/evaluation ?
            
        elif "Selection" in transformation:
            
            try:
                selected_features = transformation["Selection"]["features"]
            except ValueError:
                error = "Could not read list of selected features"
                self.output_comments.append(error)
            
            df_train_transformed = self.select_features(self.df_train, selected_features)
            df_test_transformed = self.select_features(self.df_test, selected_features)
            scaler = selected_features # better way to do this for saving/evaluation ?
            
            pass
        
        elif "derivative" in transformation:
            
            try:
                df_train_transformed = self.derivative_spectra(self.df_train)
                df_test_transformed = self.derivative_spectra(self.df_test)
                scaler = "None" # better way to do this for saving/evaluation ?
            except ValueError:
                error = "Could not do derivative transformation"
                self.output_comments.append(error)
        
        elif "peaks" in transformation:
            # use scipy.find_peaks()
            pass
        
        elif "None" in transformation: # no transformation
            df_train_transformed, df_test_transformed = self.df_train, self.df_test
            scaler = "None" # better way to do this?
            
        else:
            raise ValueError(f"{transformation} Not valid transformation")
            
        return df_train_transformed, df_test_transformed, scaler
    
    
    def pearson_reduced_features(self, max_corr, plot_corr=False):
        
        # reuse lasso model if already trained, don't retrain every time
        try:
            tuned_model = self.lasso_model_tuned #.copy()
        except:
            # train lasso model with as-is spectra
            parameters_ = [self.parameters["Lasso"]]
            best_model = GridSearchCV(Lasso(), parameters_, cv=self.kf, scoring = self.scoring) 
            best_model.fit(self.X_train, self.y_train)
            tuned_model = best_model.best_estimator_
            self.lasso_model_tuned = tuned_model
        
        # get list of features that are not zero-- if weight is zero, feature is obsolete
        weights = [coeff for coeff in tuned_model.coef_]
        lasso_features, weights = zip(*((ftr, weight) for ftr, weight in zip(self.features, weights) if weight != 0.0))
        
        # make dataframe with only remaining features
        df_lasso_ftrs = self.select_features(self.df_train, lasso_features)
        
        # remove correlations above threshold: max_corr
        reduced_features, corr_pairs = self.reduced_features_rm_corr(df_lasso_ftrs, max_corr, corr_type="pearson")
        
        # plot correlated pairs, one in each pair was removed
        if plot_corr == True:
            for i, corr_pair in enumerate(corr_pairs):
                self.plot_ftr_ftr(int(corr_pair[0]), int(corr_pair[1]), index=i)

        return reduced_features, corr_pairs

    def plot_pca_variance(self):
        
        sum_explained = []
        n_components_all = []
        max_comp = min(int(len(self.df_train) - 1), int(len(self.X_train[0]) - 1))
        frac_variances = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 
                          max_comp]
        
        for frac in frac_variances:
            pca = PCA(n_components=frac, svd_solver='full')
            pca.fit(self.X_train)
            sum_explained.append(sum(pca.explained_variance_ratio_))
            n_components_all.append(int(len(pca.explained_variance_ratio_)))
        
        frac_variances[-1] = 1.0
        plt.plot(frac_variances, n_components_all)
        plt.xlabel("Variance explained")
        plt.ylabel("Number of components")
        plt.grid()
        
        var_dict = {"fraction_variance": frac_variances,
                    "sum_variance": sum_explained,
                    "number_components": n_components_all,
                    "frac_total_components": [i/max_comp for i in n_components_all]}
        
        df_var = pd.DataFrame(data=var_dict)
        df_var.to_csv(f"{self.path}/PCA_explained_variance.csv",
                      index=False)
        
        plt.savefig(f"{self.data_plot_path}/pca_explained_variance.png")
        
        return None
    
    def butter_lowpass_filter(self, data, cutoff, fs, order):
        
        T = 5.0         # Sample Period
        fs = 30.0       # sample rate, Hz
        cutoff = 2      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
        nyq = 0.5 * fs  # Nyquist Frequency
        order = 2       # sin wave can be approx represented as quadratic
        n = int(T * fs) # total number of samples
        
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients 
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        
        return y
    
    def lowpass_filter(self, df):
        
        T = 5.0         # Sample Period
        fs = 30.0       # sample rate, Hz
        cutoff = 2      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
        nyq = 0.5 * fs  # Nyquist Frequency
        order = 2       # sin wave can be approx represented as quadratic
        n = int(T * fs) # total number of samples
            
        features = list(df.columns)
        features.remove(self.target)  

        X = df.drop(str(self.target), axis=1).values.astype(np.float)
        index_list= list(df.index.values)
            
        # Filter the data, and plot both the original and filtered signals.
        X_transformed = np.empty_like(X)
        for i, data in enumerate(X):
            y = self.butter_lowpass_filter(data, cutoff, fs, order)
            X_transformed[i] = y
        
        df_x_transformed = pd.DataFrame(data = X_transformed, 
                                     columns = features, 
                                     index = index_list)
        
        df_filtered = pd.merge(df[[str(self.target)]], 
                                   df_x_transformed, 
                                   how='outer', 
                                   left_index=True, 
                                   right_index=True)
        
        return df_filtered
    
    def export_spectra(self, df):
        
        X = df.drop(str(self.target), axis=1).values.astype(np.float)
        index_list= list(df.index.values)
        features = list(df.columns)
        features.remove(self.target)  
        
        os.mkdir(f"{self.path}/filtered")
    
        for idx, example in enumerate(X):
            
            name = index_list[idx]
            if ".csv" in name:
                name = name.replace(".csv", "")
    
            filename = f"{self.path}/filtered/{name}_filtered.csv"
    
            df_ex = pd.DataFrame(example, index=features)
            df_ex.to_csv(filename, header=None, index=True)
            
        return None
    
    def do_pca(self, n_components, 
               X_train = None,
               X_test = None):
        """ This function uses train and test dataframes and performs PCA to reduce dims.
        It fits on the train and transforms the test df based on the train.
        ----------
            n_components : integer
                number of principal components to include in PCA transform.
                Max is one less than min(n features, n training datapoints)
                (old: the total number of data points)
        """
        try:
            a_ = len(X_train)
            b_ = len(X_test)
        except:
            X_train = self.X_train
            X_test = self.X_test
            
        try:
            #col_list = [f"PC_{c}" for c in range(1, 1 + int(n_components))]
            pca = PCA(n_components=n_components)
        except:
            pca = PCA(n_components=n_components, svd_solver='full')
        
        principalComponents_train = pca.fit_transform(X_train)
        
        col_list = [f"PC_{c}" for c in range(1, 1 + int(len(pca.explained_variance_ratio_)))]
        
        principalDf_train = pd.DataFrame(data = principalComponents_train,
                                             columns = col_list, 
                                             index = self.index_list_train)

        principalComponents_test = pca.transform(X_test)
        principalDf_test = pd.DataFrame(data = principalComponents_test,
                                        columns = col_list, 
                                        index = self.index_list_test)
        
        df_train_transformed = self.df_y_train.join(principalDf_train,
#                                                    on=self.index_list_train,
                                                    how='inner')
        df_test_transformed = self.df_y_test.join(principalDf_test,
#                                                  on=self.index_list_test,
                                                    how='inner')

            
        return df_train_transformed, df_test_transformed, pca
    
    def local_averaging(self, df, n):
        """ This function takes in a spectra dataframe and 'smooths' noise by replacing each 
        intensity measurement with an average of the 'n' points to the left and right of it.
        Returns a dataframe of same dimensions. End points are averaged with whatever is around 
        if it exists.
        ----------
            df : dataframe
                dataframe with spectra and target
            n : integer
                number of points to average to the left and right of each point    
        """
            
        features = list(df.columns)
        features.remove(self.target)  

        X = df.drop(str(self.target), axis=1).values.astype(np.float)
        index_list= list(df.index.values)        
        averaged_spectra = np.array(X)
        
        for s, spectra in enumerate(X):
        
            for i in range(len(spectra)):
                
                if i < n:
                    intensity = np.mean(spectra[0:i+n])
                else:
                    try:
                        intensity = np.mean(spectra[i-n : i+n])
                    except: 
                        intensity = np.mean(spectra[i-n : -1])
                    
                averaged_spectra[s][i] = intensity
                
        df_x_averaged = pd.DataFrame(data = averaged_spectra, 
                                     columns = features, index = index_list)
        
        df_averaged = pd.merge(df[[str(self.target)]], 
                                   df_x_averaged, 
                                   how='outer', 
                                   left_index=True, 
                                   right_index=True)
       
        return df_averaged
    
    def derivative_spectra(self, df):
            
        features = list(df.columns)
        features.remove(self.target)  
        
        X = df.drop(str(self.target), axis=1).values.astype(np.float)
        index_list= list(df.index.values)        
        derivative_spec = np.array(X)
        
        for s, spectra in enumerate(X):
        
            der = np.gradient(spectra)                    
            derivative_spec[s] = der 
            
        df_x_derivative = pd.DataFrame(data = derivative_spec, 
                                     columns = features, 
                                     index = index_list)
        
        df_derivative = pd.merge(df[[str(self.target)]], 
                                   df_x_derivative, 
                                   how='outer', 
                                   left_index=True, 
                                   right_index=True)
        
        return df_derivative
    
    def select_range(self, df, wv_numbers):
        """ Returns a spectra array of requested intervals """
        
        
        return None
    
    def change_spectra_resolution(self, df):
        """ This function takes in a spectra dataframe and changes the resolution of spectra. 
        It reduces the resolution by (either removing points) or (interpolating in a new range) 
        - decide later how,
        probably not useful to increase resolution via interpolation?
        ----------
            df : dataframe
                dataframe with spectra and target
        """
        return df
    
    def remove_spectra_range(self, df, ranges_to_remove):
        """ This function takes in a spectra dataframe and removes wavenumbers in ranges to reduce features'
        ----------
            df : dataframe
                dataframe with spectra and target
            ranges_to_remove : list of 2 numbered lists, ex- [[1, 5], [10, 12]]
                list of ranges of wavenumbers to remove ex remove wavenumber 1-5 and 10-12
        """
        return df
    
    def select_features(self, df, selected_features):
        """ This function takes in a dataframe and only keeps features requested in list'
        ----------
            df : dataframe
                dataframe with spectra and target
            selected_features : list
                list of features to remain in df
        """
        try:
        # print(type(selected_features[0]))
            df_reduced = df[[self.target, *selected_features]]
        except:
            # print(selected_features)
            # print(type(selected_features[0]))
            selected_features = [str(int(ftr)) for ftr in selected_features]
            df.columns = [self.target, *[str(int(float(ftr))) for ftr in df.columns[1:]]]
            df_reduced = df[[self.target, *selected_features]]
        
        return df_reduced
    
    def peak_detection(self, df):
    
        return df
    
    def recursive_feature_elimination(self, df):
    
        return df
    