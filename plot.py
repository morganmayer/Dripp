""" This module includes plotting functions. """

import numpy as np
import pandas as pd
import csv
import re 
import os
import matplotlib.pyplot as plt 
from matplotlib import cm
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.tree import plot_tree
import multiprocessing as mp

plt.tight_layout(h_pad = 3, w_pad=3)
degree_sign= u'\N{DEGREE SIGN}'

plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
#plt.rcParams['axes.xmargin'] = 0
#plt.rcParams['axes.ymargin'] = 0

plt.rcParams.update({'font.size': 12})

plt.rcdefaults()
#plt.rcParams['axes.xmargin'] = 0

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

####################################

class Plot:

    def set_dist(self, df, set_):
        """This function takes in a dataframe and plots the distribution
        of the target column. It is saved to the path
        ----------
            set_ : string
                name of split. "train" or "test". Used for file naming
            path : string
                path where png file will be stored
        """
        
        y = self.df[str(self.target)].values.astype(np.float)
        plt.hist([y])
        
        plt.hist(self.y_train, #bins = np.arange(min_val,max_val,20),
                 alpha=0.4)
        
        plt.ylabel('frequency')
        plt.xlabel(f"{self.target}")
        plt.grid(axis = 'y') 
        plt.savefig(f"{self.data_plot_path}/{set_}_dist.png", bbox_inches='tight', dpi=self.dpi)
        plt.clf();
    
    def overall_dist(self):
        """ This function takes in the train and test dataframes and plots both 
        target distributions stacked in a histogram, It is saved to the path
        ----------
        """
        plt.hist([self.y_train, self.y_test], 
                 label=['train', 'test'], 
                 stacked=True) # add auto bin number
        plt.ylabel('frequency')
        plt.xlabel(f"{self.target}")
        plt.legend(loc='upper right')
        plt.grid(axis = 'y') 
#        plt.savefig(f"{self.data_plot_path}/overall_dist.png", bbox_inches='tight', dpi=self.dpi)
        plt.show() # wont show or save ? fix !
#        print("plt savefig")
        plt.clf();
        
        
    def overall_dist_presentation(self):
        
        min_ = min(self.y)
        max_ = max(self.y)
        bins = np.arange(min_, max_, (max_ - min_) / 10)
        
        plt.hist(self.y_train, #bins = np.arange(min_val,max_val,20),
                 alpha=0.4,
                 label='Train',
                 bins = bins,
                 color=(0.12156, 0.4666, 0.7058));
        
        plt.hist(self.y_test,
                 #np.arange(min_val,max_val,20),
                 alpha=0.6,
                 label='Test',
                 bins = bins,
                 color=(1, 0.41176, 0.1176));
        
        try:
            name = self.labels[str(self.target)]["name"]
            units = self.labels[str(self.target)]["units"]
            target_axis = f"{name} {units}"
        except:
            target_axis = ""
                
        plt.ylabel('Number of fuels')
        plt.xlabel(f'{target_axis}')
        #plt.ylim(0,140)
        plt.legend()
        
        plt.savefig(f"{self.data_plot_path}/{self.target}_overall_dist_pres.png", bbox_inches='tight', dpi=self.dpi)
        
        plt.clf();
        
    def kfold_dist(self):
        """This function takes in the train data series and K-fold indices for 
        plotting the fold distribution. It is saved to the path
        """
        
        folds = []
        try: # stratified kfold .split() takes X, y
            for train_index, test_index in self.kf.split(self.X_train, self.y_train):
                y_test_ = self.y_train[test_index]
                folds.append(y_test_)
        except: # kfold .split() takes only X
            for train_index, test_index in self.kf.split(self.X_train):
                y_test_ = self.y_train[test_index]
                folds.append(y_test_)
                
        name = self.labels[str(self.target)]["name"]
        units = self.labels[str(self.target)]["units"]
                
        labels = [f"split {i+1}" for i in range(len(folds))]
        plt.hist(folds,
                 label=labels,
                 stacked = True);
        plt.ylabel('frequency')
        plt.xlabel(f"{name} {units}")
        plt.legend(loc='upper right')
        plt.grid(axis = 'y') 
        plt.savefig(f"{self.data_plot_path}/kfold_dist.png", bbox_inches='tight', dpi=self.dpi)
        plt.clf();
        
    def spectra_3D(self):    
        """This function takes in a dataframe and plots the spectra in a 3D plot.
        It is saved to the path. [in construction]
        ----------
        """        
        # To add: sort by target value, color gradient for target value or make target = y_
        # save higher quality image, fix fontsize / use tight layout, make interactive/moving
        # add train, test, split options to plot
        # add fuel names maybe? or remove numbers from y axis
        
        fig = plt.figure(figsize=(15,15))    
        ax = fig.add_subplot(111, projection='3d')
        
        features = np.asarray(self.features)
        
        x_ = features.copy().astype(float)
        y_ = np.arange(len(self.df)).astype(float)
        
        X_,Y_ = np.meshgrid(x_,y_)
        df = self.df.sort_values(by=[str(self.target)])
        Z = df.drop(str(self.target), axis=1).values.astype(np.float)
            
        ax.plot_surface(X_, Y_, Z, rstride=1, cstride=1000, shade=True, lw=.1, alpha=0.4) 
         
        ax.set_zlabel("Intensity")
        ax.set_xlabel("Wavenumber")
        ax.set_ylabel("Fuel")
    #        ax.view_init(20,-120)
        plt.savefig(f"{self.data_plot_path}/3D_spectra.png", dpi=self.dpi)
        plt.clf();
        
    def spectra_2D(self, df, label):
        """ This function takes in a spectra dataframe and plots the 
        staggered spectra on a 2D plot. It is saved to the path
        ----------
            label : string
                name of set. "train" or "test" for file naming.
        """
        plt.figure(figsize=(3,10))
        df = df.sort_values(by=[str(self.target)])
        
        X = df.drop(str(self.target), axis=1).values.astype(np.float)
        
        for i, example in enumerate(X):
            
            intensity = X[i] + 0.2*i
            plt.plot(self.features, intensity, alpha = 0.5, linewidth=1, c='k');
            
        plt.ylabel("Intensity")
        plt.xticks(color='w')
        #plt.yticks(np.arange(650, 4000, 150))
        plt.xlabel("Wavenumber")
        plt.grid()
        plt.savefig(f"{self.data_plot_path}/spectra_2D_{label}.png", bbox_inches='tight', dpi=self.dpi)
        plt.clf();

    def plot_spectra_in_folder(self, spectra_folder_path):
        
        for filename in os.listdir(spectra_folder_path):
            
            if filename.lower().endswith(".csv"):
                filepath = os.path.join(spectra_folder_path, filename)
                print(filepath)
                
                # read in array
                with open(filepath, newline='') as csvfile:
                    
                    label = filename.split(".")[0]
                    
                    data = list(csv.reader(csvfile))
                    
                    data = np.asarray(data, dtype=float).T
                    
                    plt.plot(data[0], data[1], 
                             c='k',
                             # s=3,
                             label = label);
            
                    plt.ylabel("Intensity")
                    plt.xlabel("Wavenumber")
                    
                    # plt.yticks([])
                    ticks = np.arange(650, 4000, 1000)
                    plt.xticks(ticks)
                    
                    plt.legend(prop={'size': 6})
                    plt.grid()
                    
                    plt.savefig(f"{self.data_plot_path}/spectra_{label}.png", 
                                bbox_inches='tight', 
                                dpi=self.dpi)
                    plt.clf();
                
            else:
                continue
    
        return None
    
    #####################################
        
    def error_hist(self, error_type="absolute"):
        
        for t in range(len(self.feature_transformations)):
            
            for i, model in enumerate(self.models):
                
                model_path = self.model_paths[t][i]                
                
                df_error = self.all_df_errors[t][i].copy()
                df_err_train = df_error[df_error["Train/Test"] == "Train"]
                df_err_test = df_error[df_error["Train/Test"] == "Test"]
                
                if error_type == "percent":
                    
                    err_train = df_err_train["Percent error"].values.astype(float)
                    err_test = df_err_test["Percent error"].values.astype(float)              
                    
                    target_axis = f"Percent error {self.name} [% {self.units}]"
                    
                else:
                    err_train = df_err_train['Absolute error'].values.astype(float)
                    err_test = df_err_test['Absolute error'].values.astype(float)
                    step = 1
                    
                    target_axis = f"Error {self.name} {self.units}"
                
                err_train = [e for e in err_train if float(e) <= 100]
                err_test = [e for e in err_test if float(e) <= 100]

                min_ = min(min(err_train), min(err_test))
                max_ = max(max(err_train), max(err_test))
                
                step = (max_ - min_) / 10
                
                bins = np.arange(min_, max_, step)
                
                plt.hist(err_train, #bins = np.arange(min_val,max_val,20),
                         alpha=0.4,
                         label='Train',
                         color= (0.121, 0.466, 0.7058),
                         bins = bins
                         ); 
                
                plt.hist(err_test,
                         #np.arange(min_val,max_val,20),
                         alpha=0.6,
                         label='Test',
                         color=(1, 0.41176, 0.1176),
                         bins = bins
                         );
                
                plt.ylabel('Number of fuels')
                plt.xlabel(f'{target_axis}')

                plt.legend()
                
                plt.savefig(f"{model_path}/error_hist_{error_type}.png", 
                    bbox_inches='tight', 
                    dpi=self.dpi) 
                
                plt.clf();
                
    def cross_sect(self, wavenumber):
        
        try:
            df_reduced = self.df[[self.target, wavenumber]]
        except:
            wavenumber = str(float(wavenumber))
            df_reduced = self.df[[self.target, wavenumber]]
        
        df_reduced = df_reduced.apply(pd.to_numeric, errors="ignore")
        df_reduced = df_reduced.join(self.df_load_id["label"])

        g = sns.scatterplot(x=str(self.target), y=str(wavenumber), 
                            hue= "label",
                            data=df_reduced,
                            sizes=(40,400),
                            size=str(self.target),
                            alpha = 0.5
                            );

        plt.legend(loc=(1, 0))
        plt.grid()
        
        plt.title(str(wavenumber))
        plt.xlabel(f"{self.target}")
        plt.ylabel("Absorbance")
        
        plt.savefig(f"{self.data_plot_path}/{wavenumber}_cross_sect.png", 
                    bbox_inches='tight', 
                    dpi=self.dpi)
                
        plt.clf();
        
        return None
    
    def plot_ftr_ftr(self, ftr1, ftr2, index=""):
        """ Plots two features from whole dataset. 
        index is order of pairs removed in pearson_reduced_features, default empty """
        # try:
        # # print(type(selected_features[0]))
        #     df_reduced = self.df[[self.target, ftr1, ftr2]]
        # except:
        #     ftr1 = str(float(ftr1))
        #     ftr2 = str(float(ftr2))
        #     print(ftr1, ftr2)
        #     df_reduced = self.df[[self.target, ftr1, ftr2]]
            
        df_reduced = self.df    
            
        df_reduced = df_reduced.apply(pd.to_numeric, errors="ignore")
        df_reduced = df_reduced.join(self.df_load_id["id_label"])
        # self.df_reduced = df_reduced

        plt.figure(figsize=(10,10))
        
        g = sns.scatterplot(x=str(ftr1), y=str(ftr2), 
                            hue= "id_label",
                            data=df_reduced,
                            sizes=(40,400),
                            size=str(self.target),
                            alpha = 0.5
                            );
        plt.legend(loc=(1, 0))

        # ftr1_vals = self.df[str(ftr1)].values.astype(float)
        # ftr2_vals = self.df[str(ftr1)].values.astype(float)
                
        # plt.scatter(ftr1_vals, ftr2_vals,
        #             facecolors='none', 
        #             edgecolors='b', 
        #             marker='o', 
        #             s= 40)
        
        plt.grid()
        plt.xlabel(f"{ftr1}")
        plt.ylabel(f"{ftr2}")
        
        plt.savefig(f"{self.data_plot_path}/{index}_features_{ftr1}_{ftr2}.png", 
                    bbox_inches='tight', 
                    dpi=self.dpi)
                
        plt.clf();
        
        return None
                
    def blends_parity_plot(self, df_error_blend, path, additive,
                    with_bounds=False, bounds=None, bound_label=None):
        '''  This function takes in the true and predicted values and plots a scatter
        plot alongside a y=x line.
        '''
        
        test = df_error_blend[df_error_blend["Train/Test"] == "Test"]
        train = df_error_blend[df_error_blend["Train/Test"] == "Train"]
        
        y_test = test[str(self.target)].values.astype(float)
        y_test_pred = test["Predicted"].values.astype(float)
        y_train = train[str(self.target)].values.astype(float)
        y_train_pred = train["Predicted"].values.astype(float)
            
        min_ = min([min(y_test), min(y_test_pred), min(y_train), min(y_train_pred)])
        max_ = max([max(y_test), max(y_test_pred), max(y_train), max(y_train_pred)])
            
        plt.figure(figsize=(6,6))
        plt.subplot(aspect='equal')
        
        plt.scatter(y_train, 
                    y_train_pred, 
                    facecolors='none', 
                    edgecolors='r', 
                    marker='o', 
                    s= 40, 
                    label= "Train"
                   # label=f'Train, MAE: {np.mean(train_errors): .2f}'
                   );
        
        plt.scatter(y_test, 
                    y_test_pred, 
                    facecolors='none', 
                    edgecolors='b', 
                    marker='*', 
                    s = 40, 
                    label = "Test"
                    #label=f'Test, MAE: {np.mean(test_errors): .2f}'
                    );
            
        if with_bounds == True:
            true = [*y_train, *y_test]
            pred = [*y_train_pred, *y_test_pred]
            #print(len(true), len(pred), len(bounds))
            plt.errorbar(true,
                         pred,
                         yerr=bounds,
                         ls='none',
                         ecolor='k',
                         elinewidth = 1,
                         capsize = 2,
                         alpha=0.6
                         );
            
        plt.plot([min_, max_],[min_, max_],
                 alpha=0.4,
                 c="k");
        
        target_axis = f"{self.name} {self.units}"
        
        plt.grid()
        plt.legend(loc="lower right")

        plt.xlabel(f'True {target_axis}')
        plt.ylabel(f'Predicted {target_axis}')
        
        if with_bounds == True:
            plt.savefig(f"{path}/parity_{additive}_{bound_label}_train_test.png", 
                             bbox_inches='tight', 
                             dpi=self.dpi)                 
        else:
            plt.savefig(f"{path}/parity_{additive}_train_test.png", 
                             bbox_inches='tight', 
                             dpi=self.dpi)
        plt.clf();
        
        
    def parity_plot(self, y_test_pred, y_train_pred, path, 
                    with_bounds=False, bounds=None, bound_label=None):
        '''  This function takes in the true and predicted values and plots a scatter
        plot alongside a y=x line.
        ------------
        y_test_pred : array-like
            the predicted test values
        y_train_pred : array-like
            the predicted train values
        model_name : string
            name of algorithm in yaml input file
        transform_name : string
            name of transform specified in transform_names in input file
        path : string
            path where png file will be stored
        '''
        test_errors = np.abs(self.y_test - y_test_pred)
        train_errors = np.abs(self.y_train - y_train_pred)
    
        min_ = min([min(self.y_test), min(y_test_pred), min(self.y_train), min(y_train_pred)])
        max_ = max([max(self.y_test), max(y_test_pred), max(self.y_train), max(y_train_pred)])
            
        plt.figure(figsize=(6,6))
        plt.subplot(aspect='equal')
        
        plt.scatter(self.y_train, 
                    y_train_pred, 
                    facecolors='none', 
                    edgecolors='r', 
                    marker='o', 
                    s= 20, 
                    alpha=0.6,
                    label=f'Train MAE: {np.mean(train_errors): .2f}');
        
        plt.scatter(self.y_test, 
                    y_test_pred, 
                    facecolors='none', 
                    edgecolors='b', 
                    marker='*', 
                    s = 35, 
                    alpha=0.6,
                    label=f'Test MAE: {np.mean(test_errors): .2f}');
            
        if with_bounds == True:
            true = [*self.y_train, *self.y_test]
            pred= [*y_train_pred, *y_test_pred]

            plt.errorbar(true,
                         pred,
                         yerr=bounds,
                         ls='none',
                         ecolor='k',
                         elinewidth = 1,
                         capsize = 2,
                         alpha=0.6
                         );
            
        plt.plot([min_, max_],[min_, max_],
                 alpha=0.4,
                 c="k");
        
        target_axis = f"{self.name} {self.units}"
        
        plt.grid()
        plt.legend(loc="lower right")

        plt.xlabel(f'True {target_axis}')
        plt.ylabel(f'Predicted {target_axis}')
        
        if with_bounds == True:
            plt.savefig(f"{path}/parity_{bound_label}_train_test.png", 
                             bbox_inches='tight', 
                             dpi=self.dpi)
                 
        else:
            plt.savefig(f"{path}/parity_{self.target}_train_test.png", 
                             bbox_inches='tight', 
                             dpi=self.dpi)
                    
        plt.clf();
        
        
    def pres_parity_plot(self, y_test_pred, y_train_pred, path):
        
        min_ = min([min(self.y_test), min(y_test_pred)])
        max_ = max([max(self.y_test), max(y_test_pred)])

        plt.figure(figsize=(6,6))
        plt.subplot(aspect='equal')
        
        target_axis = f"{self.name} {self.units}"
                
        plt.figure() #figsize=(6,6)
        plt.subplot(aspect='equal')
        plt.axes().set_aspect('equal')
        
        plt.scatter(self.y_test, 
                    y_test_pred, 
                    s=38,
                    zorder=10,
                    facecolors='none', 
                    edgecolors='b',
                    marker = "*",
                    alpha = 0.7,
                    # label='Blend'
                    )
        
        plt.scatter(np.arange(min_,max_,0.5),
                    np.arange(min_,max_,0.5),
                    s=2,
                    c="orange",
                    zorder=0)
            
            
        plt.grid()
        # plt.legend(loc = "lower right")
        # plt.margins(0, 0)
        plt.xlabel(f'Actual {target_axis}')
        plt.ylabel(f'Predicted {target_axis}')
        plt.tight_layout()
 
        plt.savefig(f"{path}/parity_train_test_pres.png", 
                        bbox_inches='tight', 
                        dpi=self.dpi)
            
        plt.clf();

    def box_performances_sns(self, idx, by=None, plot_type="boxplot"):
        ''' test with seaborn
        '''        
                                              
        if by == "Transform":            
            train_errors = [x[idx] for x in self.all_train_errors]
            test_errors = [x[idx] for x in self.all_test_errors]
            name = self.models[idx]
            x_axis = self.transform_names.copy()
            
        elif by == "Algorithm":
            train_errors = self.all_train_errors[idx]
            test_errors = self.all_test_errors[idx]            
            name = self.transform_names[idx]            
            x_axis = self.models.copy()

        else:
            raise TypeError(f"'by' option {by} not valid")
        
        concat_train_errors = np.asarray(train_errors).flatten()
        concat_test_errors = np.asarray(test_errors).flatten()
        errors = np.concatenate((concat_train_errors, concat_test_errors), axis=None)
        
        labels_train = ["Train" for i in concat_train_errors]
        labels_test = ["Test" for i in concat_test_errors]
        labels = np.concatenate((labels_train, labels_test), axis=None)
        
        names_train = np.asarray([np.asarray([x_axis[t] for i in train_errors[0]]) for t in range(len(x_axis))]).flatten()
        names_test = np.asarray([np.asarray([x_axis[t] for i in test_errors[0]]) for t in range(len(x_axis))]).flatten()
        names = np.concatenate((names_train, names_test), axis=None)
        
        error_dict = {"Absolute error": errors, "Train/Test": labels, str(by): names}
        
        df_error = pd.DataFrame(data = error_dict)
        
        flierprops = dict(markerfacecolor='None', 
                          marker='o',
                          markersize=3)
        
        if plot_type == "violinsplit":
            sns.catplot(x=str(by), 
                        y='Absolute error', 
                        hue="Train/Test", 
                        data=df_error,
                        width=1,
                        kind="violin", split=True, inner="quartile", 
                        linewidth=1,
                        palette='pastel',
                        legend=False,
                        scale_hue=True,
                        cut=0)
    #                    flierprops = flierprops

        elif plot_type == "bar":
            sns.barplot(x=str(by), 
                        y='Absolute error', 
                        hue="Train/Test", 
                        data=df_error,
                       # width=0.6,
                      #  linewidth=1,
                        palette='pastel',
                        #flierprops = flierprops
#                        ax = ax
                        )
        else:
            sns.boxplot(x=str(by), 
                        y='Absolute error', 
                        hue="Train/Test", 
                        data=df_error,
                        width=0.6,
                        linewidth=1,
                        palette='pastel',
                        flierprops = flierprops
#                        ax = ax
                        )
            
        plt.legend(title="")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.grid(axis = 'y', alpha = 0.3, which="both")

        plt.savefig(f"{self.path}/performance_plots/{self.target}_{name}_performances_{plot_type}_sns.png", 
                        bbox_inches='tight', dpi=self.dpi)
        plt.clf();
        
    def bar_CV_test(self, idx, by=None):
                
        if by == "Transform":
            cv_errors = np.asarray([x[idx] for x in self.all_cv_scores])
            test_errors = np.asarray([x[idx] for x in self.all_test_errors])
            name = self.models[idx]            
            x_axis = self.transform_names.copy()
            
        elif by == "Algorithm":
            cv_errors = self.all_cv_scores[idx]
            test_errors = self.all_test_errors[idx]
            name = self.transform_names[idx]
            x_axis = self.models.copy()
        else:
            raise TypeError(f"'by' option {by} not valid")
        
        # flierprops = dict(markerfacecolor='None', 
        #                   marker='o',
        #                   markersize=3)
#                          linestyle='none'

        fig, ax = plt.subplots()
        
        x = np.arange(len(x_axis))
        cvs = np.abs(cv_errors)
        maes = [np.mean(np.abs(errors)) for errors in test_errors]
        rmses = [np.sqrt(np.mean(np.asarray(errors)**2)) for errors in test_errors]
        
        width = 0.2
        ax.bar(x+width, cvs, width=width, color='b', alpha= 0.3, align='center', label = "CV Error")
        ax.bar(x, rmses, width=width, color='r', alpha= 0.3, align='center', label= "Test RMSE")
        ax.bar(x-width, maes, width=width, color='g', alpha= 0.3, align='center', label= "Test MAE")
        
        ax.set_xticks(x)
        ax.set_xticklabels(x_axis)
        ax.set_ylabel("Error")
        ax.set_xlabel(str(by))
        
        # ax.bar_label(rects1, padding=3, label_type='top')
        # ax.bar_label(rects2, padding=3, label_type='top')
        
        ax.legend(title="")
        
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.grid(axis = 'y', alpha = 0.3, which="both")

        plt.savefig(f"{self.path}/performance_plots/{self.target}_{name}_CV_test_performances.png", 
                        bbox_inches='tight', dpi=self.dpi)

        plt.clf();
    
    def PC_spectra(self, df, label):
        """This function takes in spectra dataframe that was transformed with PCA and plots the 
        staggered PCA-spectra on a 2D plot. It is saved to the path
        ----------
            df : pandas dataframe
                dataframe of set
            label : string
                name of set. "train" or "test" for file naming.
        """
        # spacing for PC's
        if label == "train":
            space = 10
        else:
            space = 5
        
        X = df.drop(str(self.target), axis=1).values.astype(np.float)
        
        features = list(df.columns)
        features.remove(self.target)
        features = np.asarray(features)
        
        for i, example in enumerate(X):
            
            intensity = X[i] + space*i
            plt.plot(self.features, intensity, alpha = 0.5, linewidth=1, c='k')
            
        plt.yticks([])
        plt.xticks([])
        plt.xlabel("Principal component")
        plt.savefig(f"{self.data_plot_path}/PCA_spectra_2D_{label}.png", bbox_inches='tight', dpi=self.dpi)
        plt.clf();
        
    def all_spectra(self):    
        """This function takes in a dataframe and plots each spectra
        individually and stores the images in a folder in the path.
        ----------
        """    
        # make folder for individual spectra
        spectra_path = f"{self.data_plot_path}/spectra_plots"
        os.mkdir(spectra_path)
        
        examples = self.df.index.values.tolist()
        
        features_wavenumber = np.asarray(self.features).astype(np.float)
        
        for i, spectra in enumerate(self.X):
            
            plt.plot(features_wavenumber, self.X[i])
            plt.xlabel("Wavenumber")
            plt.ylabel("Intensity")
            plt.title(f"{examples[i]}")
            name = examples[i].replace("/","-")
            plt.savefig(f"{spectra_path}/{name}_spectra.png", bbox_inches='tight', dpi=self.dpi)
            plt.clf();
            
    def importance_grouping(self, features, importances):
        
        bins = np.arange(700, 4050, 50) # change to auto binning depending on len(importances)
        summed_imps = []
        
        for b, bin_ in enumerate(bins):
            if b == 0:
                bin_importances = [importances[i] for i, ftr in enumerate(features) if float(ftr) < bin_]
#                print(np.sum(bin_importances))
            else:    
                bin_importances = [importances[i] for i, ftr in enumerate(features) if bins[b-1] < float(ftr) < bin_]
#            print(np.sum(bin_importances))
            
            summed_imps.append(np.sum(bin_importances)) 
            
        return bins, np.asarray(summed_imps)
    
    def feature_importance(self, importances, features, path):  
        
        # importances = importances / max(importances)
        #plt.rcParams['axes.xmargin'] = 0
        
        if len(features) < 10:
            
            #abs_importances = [abs(x) for x in importances]
            #importances = importances / sum(abs_importances)
            
            plt.barh(np.arange(len(importances)),
                importances, 
                color='slateblue',
                edgecolor='white')
            
            plt.yticks(range(len(importances)), features, 
                       fontsize=10,
                      # rotation=60
                       )
            
            # Add xticks on the middle of the group bars
            plt.ylabel(f'Wavenumber ({self.labels["wv"]["units"]})') #, fontweight='bold')
            
            # Create legend & Show graphic
            plt.xlabel('Weight')
           # plt.xlim(0, 1)
           
            plt.axis('tight')
            
            plt.grid(axis = 'x')
        
            plt.savefig(f"{path}/{self.target}_feature_importance_plot.png", bbox_inches='tight', dpi=self.dpi)
            plt.clf();
            
        elif len(features) < 250:
            
            # abs_importances = [abs(x) for x in importances]
            # importances = importances / sum(abs_importances)
            
            plt.barh(np.arange(len(importances)), 
                importances, 
                color='slateblue',
                edgecolor='white')
            plt.yticks(range(len(importances)), features, 
                       fontsize=10,
                    #   rotation=60
                       )
            
            # Add xticks on the middle of the group bars
            plt.ylabel(f'Wavenumber ({self.labels["wv"]["units"]})')
            
            # Create legend & Show graphic 
            plt.xlabel('Weight')
            
            plt.axis('tight')
            
         #   plt.xlim(0, 1)
            
            plt.grid(axis = 'x')
        
            plt.savefig(f"{path}/{self.target}_feature_importance_plot.png", bbox_inches='tight', dpi=self.dpi)
            plt.clf();
            
        elif self.data["spectra"] == True: # spectra, many points, so bin
            
            bins, summed_imps = self.importance_grouping(features, importances)

#            plt.bar(bins, summed_imps)
            plt.barh(np.arange(len(summed_imps)), 
                summed_imps, 
                color='slateblue',
                edgecolor='white')
            
            plt.yticks(range(len(summed_imps)), bins, 
                       fontsize=8
                      # rotation=60
                       )
            # Add xticks on the middle of the group bars
            plt.ylabel(f'Wavenumber range ({self.labels["wv"]["units"]})')
            
            # Create legend & Show graphic
            plt.xlabel('Weight sum')
            
            plt.axis('tight')
     
            plt.grid(axis = 'x')
        
            plt.savefig(f"{path}/{self.target}_feature_importance_plot.png", bbox_inches='tight', dpi=self.dpi)
            plt.clf();
            
        else:
            # too many features to plot
            error = "Too many features for bar plot in {path}"
            self.output_comments.append(error)
        
    def pca_feature_importance(self):
        
        pca = PCA(n_components=len(self.df_train)-1)
        pca.fit(self.X_train)
        
        components_array = abs(pca.components_)
        
        bins, summed_imps_0 = self.importance_grouping(self.features, components_array[0])
        
        binned_importances = np.ones((len(components_array), len(bins)))
        
        for i, pc_importances in enumerate(components_array):    
            if i == 0:
                binned_importances[0] = summed_imps_0
            else:
                _, summed_imps = self.importance_grouping(self.features, pc_importances)
                binned_importances[i] = summed_imps
        
        # plot staggered binned importance
        self.pca_importance_heatmap(bins, binned_importances)
        
    
    def pca_importance_staggered(self, bins, binned_importances):

        for i, pc_importances in enumerate(binned_importances):
            y_values = pc_importances + 0.2*i
            plt.plot(bins, y_values, alpha = 0.5, linewidth=1, c='k')
            
        plt.ylabel("Binned importance")
        plt.yticks([])
        plt.xlabel("Wavenumber range")
#        plt.show()
        plt.savefig(f"{self.data_plot_path}/pca_importance_staggered.png", 
                    bbox_inches='tight', dpi=self.dpi)
        plt.clf();
        
    
    def pca_importance_heatmap(self, bins, binned_importances):

        sns.heatmap(binned_importances)
        
        plt.xlabel("Wavenumber range")
        plt.ylabel("Principal component")
        plt.xticks(range(len(bins)), bins, 
                       fontsize=4,
                       rotation=90)

        plt.savefig(f"{self.data_plot_path}/pca_importance_heatmap.png", 
                    bbox_inches='tight', dpi=self.dpi)
        plt.clf();
        
    def plot_save_corr_features(self, t, corr_type="pearson"):
        
        label = f"{self.transform_names[t]}" 
        self.correlated_features_heatmap(self.df_trains_transformed[t], 
                                         label, 
                                         corr_type=corr_type)    
        
        if len(self.df_trains_transformed[t].columns) < 300: # dont want to save a huge csv
            # save correlation matrix to csv
            corr_matrix = self.df_trains_transformed[t].corr(method="pearson")
            corr_matrix.to_csv(f"{self.data_plot_path}/matrix_{label}_true_correlations_{corr_type}.csv", 
                                       float_format='%.3f')
        
        return None
        
    def plot_save_all_corr_features(self, corr_type="pearson", parallel=False):
        
        if parallel == True: # broken at the moment
            pool = mp.Pool(mp.cpu_count())
            _ = [pool.apply(self.plot_save_corr_features, args=(t), kwargs={corr_type: corr_type}) for t, _ in enumerate(self.feature_transformations)]
            pool.close()  
        
        else:
            for t in range(len(self.feature_transformations)):
                
                self.plot_save_corr_features(t, corr_type=corr_type)
            
        return None
    
    def plot_two_features(self, df_train, ftr_1, ftr_2):
        
        ftr_1_vals = df_train.loc[str(ftr_1)].values
        ftr_2_vals = df_train.loc[str(ftr_2)].values
        
        plt.plot(ftr_1_vals, ftr_2_vals)
        plt.xlabel(f"{ftr_1}")
        plt.ylabel(f"{ftr_2}")
        
        plt.savefig(f"{self.data_plot_path}/features_{ftr_1}_{ftr_2}.png", 
                    bbox_inches='tight', dpi=self.dpi)
        
        return None
    
        
    def correlated_features_heatmap(self, df, label, corr_type="pearson"):
        
        # if under certain number of features, make heatmap 
        # option spearman or pearson

        # https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas
        
        # save sorted abs correlation matrix as dataframe
        corrs = self.corr_features(df, corr_type=corr_type, abs_ = True)
        corrs.to_csv(f"{self.data_plot_path}/sorted_{label}_absolute_correlations_{corr_type}.csv", float_format='%.3f')
        
        # save sorted true correlation matrix as dataframe, not absolute
        corrs = self.corr_features(df, corr_type=corr_type, abs_ = False)
        corrs.to_csv(f"{self.data_plot_path}/sorted_{label}_true_correlations_{corr_type}.csv", float_format='%.3f')
        
        # plot heatmap, not absolute value of corrs
        if len(df.columns) <= 200: # plot heatmap if less features so can actually read it
            plt.figure(figsize=(12,10))
            cor = df.corr(method=str(corr_type))
            sns.heatmap(cor, annot=(len(df.columns) <= 30), cmap=plt.cm.Reds)
            plt.savefig(f"{self.data_plot_path}/heatmap_{label}_correlation_{corr_type}.png", 
                    bbox_inches='tight', dpi=self.dpi)
            plt.clf();
            
        return None


    # def plot_randomforest_viz(self):
        
    #     for t, transform in enumerate(self.feature_transformations):
        
    #         for i, model in enumerate(self.models):
                
    #                 tuned_model = self.all_tuned_models[t][i] 
                
    #                 model_path = self.model_paths[t][i]
            
    #                 fig = plt.figure(figsize=(15, 10))
                    
                    
    #                 plot_tree(rf.estimators_[0], 
    #                   feature_names=wine.feature_names,
    #                   class_names=wine.target_names, 
    #                   filled=True, impurity=True, 
    #                   rounded=True)
        
    #     return None

        
#