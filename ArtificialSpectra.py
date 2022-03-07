#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pint
from pint import UnitRegistry
degree_sign= u'\N{DEGREE SIGN}'
import os

import math
import seaborn as sns

plt.rcParams.update({'font.size': 12})

def warn(*args, **kwargs): #supress warnings
    pass
import warnings
warnings.warn = warn

plt.tight_layout(h_pad = 3, w_pad=3)

import scipy.optimize
import scipy.interpolate as interp
from scipy.interpolate import griddata
from scipy.interpolate import interp2d
from sklearn.metrics import r2_score

class ArtificialSpectra:
    
    def __init__(self, additives):
        
        self.additives = additives
        self.binary_name = additives[0]+"_"+additives[1]
        try:
            os.mkdir(f"data/interpolated_spectra_analysis/{self.binary_name}")
        except:
            pass
        try:
            os.mkdir(f"plots/interpolated_spectra_analysis/{self.binary_name}")
        except:
            pass
        
        # read in spectra and properties dataframes
        df_spectra = pd.read_csv("data/df_spectra.csv", 
                               header = 0, dtype=object, index_col="name")
        df_props_id = pd.read_csv("data/df_props_id.csv", 
                               header = 0, dtype=object, index_col="name")
               
        # wavenumbers in array
        self.wavenumbers = np.asarray(df_spectra.columns.values.astype(float))
        
        # read in blend composition csv file
        self.df_blend = pd.read_csv("data/blend_composition.csv", 
                                header = 0, dtype=object, index_col="name")
        
        #remove interpolated spectra rows
        self.df_blend = self.df_blend.loc[self.df_blend['id_spectra_source'] == "exp"]
        df_blend_comp = self.df_blend.loc[:, ~self.df_blend.columns.str.contains('id')] #remove id col like filename
        df_blend_comp = df_blend_comp.apply(pd.to_numeric, errors='coerce')/100   
        
        # keep only blends with the two aditives in it
        df_blend_comp = df_blend_comp[additives]
        df_blend_comp = df_blend_comp[df_blend_comp.sum(axis=1) == 1.0]
        
        # make df including blends and neat additives
        df_additives = pd.DataFrame(data=[[1,0], [0,1]],
                      index=additives,
                      columns=additives)
        df_comp = pd.concat([df_blend_comp, df_additives])
        self.df_comp = df_comp.sort_values(additives[0], axis=0)
        
        # spectra dataframe of neat and blends, sorted
        self.df_spectra = df_spectra.loc[self.df_comp.index]

        # array of additive volume fractions, array of names
        self.additive_comp = self.df_comp[str(additives[0])].values.astype(np.float)
        self.names = self.df_comp.index.values.tolist()
        
    def test_all_interp(self, degrees=[1, 2, 3]):
        
        for d in degrees:
            for i in range(len(self.additive_comp)):
                all_spec, perfs_ = self.test_interpolation([int(i)], degree=d)

    def spectra_vertical_interpolation(self, abs_train_, comp_train, comp_test, degree):
        
        interp_spectra = np.empty_like(self.wavenumbers)
        polyfits = []
        
        for idx, wv in enumerate(self.wavenumbers):

            coeffs = np.polyfit(comp_train, abs_train_[:, idx], degree)
            poly = np.poly1d(coeffs)
            abs_interp = float(poly(comp_test))
            interp_spectra[idx] = abs_interp
            polyfits.append(poly)
        
        return interp_spectra, polyfits
        
    def test_interpolation(self, idx_test, degree=2, save=True):
        
        absorbances = self.df_spectra.values.astype(np.float)
        test_names = np.asarray(self.names)[idx_test]
        idx_train = np.arange(0, len(self.additive_comp))
        idx_train = np.delete(idx_train, idx_test)

        # absorbances arrays
        abs_train = absorbances[idx_train] 
        abs_test = absorbances[idx_test]

        # composition array
        comp_train = self.additive_comp[idx_train] # x
        
        # empty arrays for spectra and metrics
        all_interp_spectra = np.empty_like(abs_test)
        fit_r2 = np.empty_like(abs_test)
        diffs = np.empty_like(abs_test)
        percent_diffs = np.empty_like(abs_test)

        # loop through every test example, usually only 1
        for j, example in enumerate(abs_test):
            
            comp_test = self.additive_comp[idx_test][j]
            interp_spectra, poly_ = self.spectra_vertical_interpolation(abs_train, 
                                                            comp_train, 
                                                            comp_test, 
                                                            degree)
            for idx, wv in enumerate(self.wavenumbers):
                
                poly = poly_[idx] # polynomial fit for wavelength
                
                # save R2
                abs_pred = np.asarray([poly(comp_train_i) for comp_train_i in comp_train])
                r2 = r2_score(abs_train[:, idx], abs_pred)
                fit_r2[j][idx] = float(r2)

                # save diff
                diff = float(abs_test[j][idx] - interp_spectra[idx])
                diffs[j][idx] = diff

                # save percent diff
                percent_diff = 100 * diff / abs_test[j][idx]
                percent_diffs[j][idx] = float(percent_diff)

            all_interp_spectra[j] = interp_spectra

            if save == True:
                self.export(interp_spectra, f"data/interpolated_spectra/{self.names[idx_test[j]]}_polyfit{degree}.csv")
                self.export(fit_r2[j], f"data/interpolated_spectra_analysis/{self.binary_name}/r2_fit_{self.names[idx_test[j]]}_polyfit{degree}.csv")
                self.export(percent_diffs[j], f"data/interpolated_spectra_analysis/{self.binary_name}/percent_diff_{self.names[idx_test[j]]}_polyfit{degree}.csv")

        perfs = {"r2": fit_r2,
                 "diffs": diffs,
                 "percent_diffs": percent_diffs}
        
        if save == True:
            # Plot R2, abs diff, percent diff
            self.plot_agreement_overlay(test_names, perfs, degree, "diffs")
            self.plot_agreement_overlay(test_names, perfs, degree, "percent_diffs")
        
        return all_interp_spectra, perfs
    
    def export(self, X, name):
        
        df_ex = pd.DataFrame(X, index=self.wavenumbers)
        df_ex.to_csv(name, header=None, index=True)        
        
        return None
    
    def plot_agreement_overlay(self, test_names, perf_, degree, kind):

        r2 = perf_["r2"]
        diffs = perf_["diffs"]
        percent_diffs = perf_["percent_diffs"]

        for idx, name in enumerate(test_names):

            # create figure and axis objects with subplots()
            fig = plt.figure()

            ax = fig.add_subplot(111)
            ax2 = fig.add_subplot(111, frame_on=False)

            ax.scatter(self.wavenumbers,
                    r2[idx],
                    s=8,
                    marker='o',
                   # mfc="none"
                   alpha=0.5,
                   #label=f"Mean R-squared: {np.mean(r2[idx]): .2f}"
                    );
            
            # ax.legend(bbox_to_anchor=(1.2, -0.1))

            ax.set_xlabel("Wavenumber") #,fontsize=14          
            ax.set_ylabel("R-squared")
            ax.set_ylim(0,1.01)

            if kind == "diffs":
                y = diffs.copy()
                label = f"MAE: {np.mean(np.abs(y[idx])): .2E}"
            elif kind == "percent_diffs":
                y = percent_diffs.copy()
                label = f"MAPE: {np.mean(np.abs(y[idx])): .2f}%"

            ax2.scatter(self.wavenumbers,
                    y[idx], 
                    color="red",
                    s=8,
                   # mfc='none',
                    marker='*',
                  #  label="Train",
                   alpha=0.5,
                   label = label
                 );

            #ax2.xaxis.tick_top()
            #ax2.set_xticks([])
            #ax2.xaxis.set_label_position("top")

            ax2.tick_params(axis='y', which='both', color='r')
            ax2.yaxis.tick_right()
            ax2.yaxis.set_label_position("right")

            ax2.grid(zorder = 2)

            if kind == "diffs":
                ax2.set_ylabel("Absorbance difference: (exp - pred)", color="red")
                ax2.set_ylim(-0.006, 0.006)
                filename = f"polyfit{degree}_diffs"
            elif kind == "percent_diffs":
                ax2.set_ylabel("Percent absorbance difference", color="red")
                ax2.set_ylim(-200, 200)
                filename = f"polyfit{degree}_percent_diffs"
           # ax2.legend(bbox_to_anchor=(1.2, -0.1))
            #ax2.set_xlim() 


            plt.title(f"{test_names[idx]} Polyfit degree {degree}, Mean R2: {np.mean(r2[idx]): .2f}, {label}")
            plt.savefig(f"plots/interpolated_spectra_analysis/{self.binary_name}/{filename}_{test_names[idx]}.png",
                        bbox_inches='tight', 
                        dpi=800)

            #plt.clf()

        return None
    
    def plot_abs_conc(self, wv_to_plot, test_spectras, idx_test, degrees):

        X_ = self.df_spectra.values.astype(np.float)
        idx_train = np.arange(0, len(self.additive_comp))
        idx_train = np.delete(idx_train, idx_test)

        #compositions
        comp_train = self.additive_comp[idx_train] 
        comp_test = self.additive_comp[idx_test]

        # absorbances of important wavenumbers
        idx_wv = np.asarray([np.argwhere(self.wavenumbers==wv)[0][0] for wv in wv_to_plot])
    
        # plot for each important wavelength
        for i, idx in enumerate(idx_wv):

            # absorbances for exp and artificial spectra
            abs_exp = [X_[int(c)][idx] for c in np.arange(0, len(self.additive_comp))]
            all_abs_pred = []
            for d, degree in enumerate(degrees):
                abs_pred = [test_spectras[d][int(c)][idx] for c in np.arange(0, len(comp_test))]
                all_abs_pred.append(abs_pred)

            plt.scatter(self.additive_comp, abs_exp,
                       label="Experimental",
                       marker = "o",
                        s = 30,
                       facecolors='none', 
                        edgecolors='b');

            markers = ["*", "^", "D"]
            for d, degree in enumerate(degrees):
                plt.scatter(comp_test, all_abs_pred[d],
                           label=f"degree {degree}",
                           marker = markers[d],
                            s = 30,
                           facecolors='none', 
                            edgecolors='r');

            plt.legend(loc=(1, 0))
            plt.grid()
            plt.xlabel(f"Volume fraction {self.additives[0]}")
            plt.ylabel("Absorbance")
            
            name = f"{self.additives[0]} and {self.additives[1]}"

            plt.title(f"{wv_to_plot[i]} cm-1, {name}")

            filename = f"plots/interpolated_spectra_analysis/wv_plots_{name}_{wv_to_plot[i]}_cm-1.png"
            plt.savefig(filename,
                        bbox_inches='tight',
                       dpi=800)

            #plt.clf()

        return None
    
    def just_wv_plots(self, wv_to_plot, idx_test, degrees=[1, 2]):
    
        X_ = self.df_spectra.values.astype(np.float)
        test_names_ = np.asarray(self.names)[idx_test]

        test_spectras = []

        for d, degree in enumerate(degrees):

            test_spectra, perfs = self.test_interpolation(idx_test, save=False)
            test_spectras.append(test_spectra)

        self.plot_abs_conc(wv_to_plot, test_spectras, idx_test, degrees)

        return None
    
    def make_all_interp_spectra(self, comps, degree):
        
        make_names = [f"{self.additives[0]}_{str(i).zfill(2)}_{self.additives[1]}_{str(100-i).zfill(2)}_interp" for i in comps]
        self.artificial_dicts = {}
        
        for idx, comp in enumerate(comps):
            self.artificial_dicts[make_names[idx]] = {}
            comp = float(comp/100)
            interp_spectra_, filename = self.make_interp_spectra(comp, make_names[idx], degree=degree)
            self.artificial_dicts[make_names[idx]]["id_spectra_filename"] = filename 
            self.artificial_dicts[make_names[idx]][str(self.additives[0])] = comp*100
            self.artificial_dicts[make_names[idx]][str(self.additives[1])] = 100 - comp*100
            self.artificial_dicts[make_names[idx]]["id_n_components"] = len(self.additives)
            self.artificial_dicts[make_names[idx]]["id_spectra_source"] = f"interpolated polyfit degree {degree}"
                              
        return self.artificial_dicts
    
    def make_interp_spectra(self, comp, name, degree=2):

        abs_train = self.df_spectra.values.astype(np.float)
        interp_spectra = np.empty_like(self.wavenumbers)
        
        interp_spectra, _ = self.spectra_vertical_interpolation(abs_train, self.additive_comp, comp, degree)

        filename = f"data/interpolated_spectra/{name}_polyfit{degree}.csv"
        self.export(interp_spectra, filename)

        return interp_spectra, filename.replace("data", "..")
