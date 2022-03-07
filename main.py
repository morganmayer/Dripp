import Compare as compare

import multiprocessing as mp

def sequence(input_file):
    
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
    c.save_test_performance_df()
    
    c.note_important_features()
    

if __name__ == '__main__':
    
    pool = mp.Pool(mp.cpu_count())
    
    property_inputs = ["cn_input.yaml",
                        # "fp_c_input.yaml",
                        # "mp_c_input.yaml",
                        # "MW_input.yaml",
                        # "HC_input.yaml"
                        ]
    
    pool.map(sequence, property_inputs)

    pool.close()
    
    # pool = mp.Pool(mp.cpu_count())
    
    # property_inputs = ["cn_input_more.yaml",
    #                     "fp_c_input_more.yaml",
    #                     "mp_c_input_more.yaml",
    #                     "MW_input_more.yaml",
    #                     "HC_input_more.yaml"
    #                     ]
    
    # pool.map(sequence, property_inputs)

    # pool.close()
    

    # input_file = "cn_input.yaml"
    # sequence(input_file)
    
    
    # input_file = "fp_c_input.yaml"
    # sequence(input_file)
    
    
    # input_file = "mp_c_input.yaml"
    # sequence(input_file)
    
    
    # input_file = "MW_input.yaml"
    # sequence(input_file)
    
    
    # input_file = "HC_input.yaml"
    # sequence(input_file)