""" This module is meant for control sequence """
# import os
# import shutil
import argparse
# import yaml
# from datetime import datetime
# import numpy as np
import Compare as compare

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str,
                        help='name of input yaml file')

    args = parser.parse_args()
    input_file = args.input_file

    c = compare.Compare(input_file)
    
    c.split()
    
    c.plot_data()
    
    # c.plot_ftr_ftr(1467, 2927)
    
    c.post_split_cleaning()
    
    c.do_all_transformations()
    
    c.plot_save_all_corr_features(corr_type="pearson")
    # c.plot_save_all_corr_features(corr_type="spearman")
    
    # c.get_all_reduced_features(0.9, corr_type="pearson")
    
    c.tune_train_all_models()

    c.calculate_error()
    
    c.save_error()

    c.plot_performances()

    c.calculate_save_rmse()
    
    # c.calculate_save_rmsle()

    c.error_hist()
    
    c.note_important_features()
    
    c.save_test_performance_df()
    
   # c.plot_save_all_corr_features(corr_type="spearman")
    
    # c.plot_pca_variance()
    
    # c.pca_feature_importance()
    
    # c.plot_save_all_corr_features()
    
  #  c.PC_spectra(c.df_train, "train")
  
    n_clusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # c.overlay_plot(0, 0, n_clusters, metric="mae", method="kmeans")
    # c.overlay_plot(0, 0, n_clusters, metric="max", method="kmeans")
    # c.overlay_plot(0, 0, n_clusters, metric="1stdev", method="kmeans")
    c.overlay_plot(0, 0, n_clusters, metric="2stdev", method="kmeans")
    # c.overlay_plot(0, 0, n_clusters, metric="3stdev", method="kmeans")

    # c.n_clusters_plot(0, 0, n_clusters, metric="mae")
    # # c.n_clusters_plot(1, 0, n_clusters, metric="mae")
    # c.bounds_dist_plot(0, 0, n_clusters, metric="mae")
    # # c.bounds_dist_plot(1, 0, n_clusters, metric="mae")
    
    # c.n_clusters_plot(0, 0, n_clusters, metric="max")
    # # c.n_clusters_plot(1, 0, n_clusters, metric="max")
    # c.bounds_dist_plot(0, 0, n_clusters, metric="max")
    # c.bounds_dist_plot(1, 0, n_clusters, metric="max")
    
    # # c.n_clusters_plot(0, 0, n_clusters, metric="1stdev")
    # c.n_clusters_plot(1, 0, n_clusters, metric="1stdev")
    # # c.bounds_dist_plot(0, 0, n_clusters, metric="1stdev")
    # c.bounds_dist_plot(1, 0, n_clusters, metric="1stdev")
    
    c.n_clusters_plot(0, 0, n_clusters, metric="2stdev")
    # c.n_clusters_plot(1, 0, n_clusters, metric="2stdev")
    c.bounds_dist_plot(0, 0, n_clusters, metric="2stdev")
    # c.bounds_dist_plot(1, 0, n_clusters, metric="2stdev")
    
    # c.n_clusters_plot(0, 0, n_clusters, metric="3stdev")
    # # c.n_clusters_plot(1, 0, n_clusters, metric="3stdev")
    # c.bounds_dist_plot(0, 0, n_clusters, metric="3stdev")
    # c.bounds_dist_plot(1, 0, n_clusters, metric="3stdev")
    
    
    # c.n_clusters_plot(2, 0, metric="mae")
    # c.n_clusters_plot(3, 0, metric="mae")
    # c.n_clusters_plot(4, 0, metric="mae")
    # c.n_clusters_plot(5, 0)
    # c.n_clusters_plot(6, 0)
    
    