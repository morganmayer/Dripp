""" This module is meant for control sequence """

import Compare as compare

def sequence(input_file):
    
    c = compare.Compare(input_file)
    c.split()
    c.plot_data()
    c.post_split_cleaning()
    c.do_all_transformations()
    
    c.plot_save_all_corr_features(corr_type="pearson")
    
    # c.get_all_reduced_features(0.95, corr_type="pearson")
    # c.get_all_reduced_features(0.9, corr_type="pearson")
    # c.get_all_reduced_features(0.85, corr_type="pearson")
    # c.get_all_reduced_features(0.8, corr_type="pearson")
    # c.get_all_reduced_features(0.75, corr_type="pearson")
    # c.get_all_reduced_features(0.7, corr_type="pearson", plot_corr=True)
    
    c.tune_train_all_models()
    c.calculate_error()
    c.save_error()
    c.plot_performances()
    c.calculate_save_rmse()
    c.error_hist()
    c.note_important_features()
    c.save_test_performance_df()
    
    n_clusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    c.overlay_plot(0, 0, n_clusters, metric="2stdev", method="kmeans")
    
    c.n_clusters_plot(0, 0, n_clusters, metric="2stdev")
    # c.bounds_dist_plot(0, 0, n_clusters, metric="2stdev")

if __name__ == '__main__':


    # input_file = "cn_best.yaml"
    # sequence(input_file)
    
    
    input_file = "fp_c_best.yaml"
    sequence(input_file)
    
    
    # input_file = "mp_c_best.yaml"
    # sequence(input_file)
    
    
    # input_file = "MW_best.yaml"
    # sequence(input_file)
    
    
    # input_file = "HC_best.yaml"
    # sequence(input_file)