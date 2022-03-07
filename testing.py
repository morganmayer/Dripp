import Compare as compare

if __name__ == '__main__':
    
    input_file = "test_cn.yaml"
    
    c = compare.Compare(input_file)
    c.split()
    c.plot_data()
    c.post_split_cleaning()
    
    c.do_all_transformations()
    
    c.plot_save_all_corr_features(corr_type="pearson")
    
    c.tune_train_all_models()
    
    c.calculate_error()
    c.save_error()
    c.plot_performances()
    c.calculate_save_rmse()
    c.error_hist()
    
    c.note_important_features()
    c.save_test_performance_df()
