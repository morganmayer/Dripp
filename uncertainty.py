import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering

class Uncertainty:
    
    def clustering_uncertainty_comparison(self, model_idx, 
                                          transform_idx, n_clusters=5, 
                                          method="kmeans", 
                                          metric="mae", 
                                          save=True, 
                                          plot_parity_bounds=True):
        
        if method == "kmeans":            
            clustering = KMeans(n_clusters=n_clusters, random_state=self.seed) 
            
        elif method == "spectral": # can't use bc no predict method...
            clustering = SpectralClustering(n_clusters=2,
                                            assign_labels="discretize", #or "kmeans" (more sensitive to randomization, morep)
                                            random_state=self.seed)
        elif method == "agglomerative":
            clustering = AgglomerativeClustering()
            n_clusters=0
            
        elif method == "meanshift":
            bandwidth = estimate_bandwidth(self.X_train, quantile=0.2, n_samples=500)
            clustering =  MeanShift(bandwidth=bandwidth, bin_seeding=True)
            n_clusters=0
            
        clustering.fit(self.X_train) # change to do clustering with transformed data not just cleaned?
        train_clusters = clustering.predict(self.X_train)
        test_clusters = clustering.predict(self.X_test)        
        all_clusters = [*train_clusters, *test_clusters]
        
        #print(transform_idx, model_idx)
        #print(len(self.all_df_errors))
        df_error = self.all_df_errors[transform_idx][model_idx]
        df_clusters = df_error.copy()
        df_clusters["cluster"] = all_clusters
        df_clusters_train = df_clusters.loc[df_clusters['Train/Test'] == "Train"]
        
        cluster_means = df_clusters_train.groupby("cluster")["Absolute error"].mean().values
        # print(cluster_means)
        cluster_stdevs = df_clusters_train.groupby("cluster")["Absolute error"].std().values
        # print(type(cluster_stdevs), len(cluster_stdevs))
        cluster_maxs = df_clusters_train.groupby("cluster")["Absolute error"].max().values
        
        cluster_bounds = np.empty_like(df_clusters.cluster.values.astype(float))
        
        if metric == "mae":
            
            for idx, cluster in enumerate(df_clusters.cluster.values):
                cluster_bounds[idx] = float(cluster_means[cluster])
                if type(cluster_bounds[idx]) != float:
                    cluster_bounds[idx] = cluster_means[cluster]
                    
            df_clusters['cluster mae'] = cluster_bounds 
            df_clusters["In bounds?"] = df_clusters["Absolute error"] <= df_clusters['cluster mae']
            
        elif metric == "1stdev":
            
            for idx, cluster in enumerate(df_clusters.cluster.values):
                try:
                    cluster_bounds[idx] = float(cluster_means[cluster] + cluster_stdevs[cluster])
                except:
                    cluster_bounds[idx] = cluster_maxs[cluster]
                    
            df_clusters['cluster stdev'] = cluster_bounds           
            df_clusters["In bounds?"] = df_clusters["Absolute error"] <= df_clusters['cluster stdev']
            
        elif metric == "2stdev":
            
            for idx, cluster in enumerate(df_clusters.cluster.values):
                try:
                    cluster_bounds[idx] = float(2*cluster_stdevs[cluster] + cluster_means[cluster])
                except:
                    # print("Case where only one in cluster")
                    cluster_bounds[idx] = cluster_maxs[cluster]
                    
            df_clusters['cluster 2*stdev'] =   cluster_bounds  
            df_clusters["In bounds?"] = df_clusters["Absolute error"] <= df_clusters['cluster 2*stdev']
            
        elif metric == "3stdev":
            
            for idx, cluster in enumerate(df_clusters.cluster.values):
                try:
                    cluster_bounds[idx] = float(3*cluster_stdevs[cluster] + cluster_means[cluster])
                except:
                    cluster_bounds[idx] = cluster_maxs[cluster]
                    
            df_clusters['cluster 3*stdev'] = cluster_bounds         
            df_clusters["In bounds?"] = df_clusters["Absolute error"] <= df_clusters['cluster 3*stdev']
            
        elif metric == "max":
            cluster_bounds = [float(cluster_maxs[cluster]) for cluster in df_clusters.cluster.values]
            df_clusters['cluster max error'] = cluster_bounds
            df_clusters["In bounds?"] = df_clusters["Absolute error"] <= df_clusters['cluster max error']

        n_in_bounds = df_clusters.groupby("Train/Test")["In bounds?"].sum()
        frac_inbounds = {}
        frac_inbounds["Train"] = n_in_bounds["Train"]/len(self.X_train)
        frac_inbounds["Test"] = n_in_bounds["Test"]/len(self.X_test)
        
        # save to path??
        # model_path = f"{self.path}/{self.models[model_idx]}_{self.transform_names[transform_idx]}" 
        if save == True:
            df_clusters.to_csv(f"{self.uncertainty_path}/UQ_df/df_clusters_model{model_idx}_transform{transform_idx}_{method}_{metric}_{n_clusters}clusters.csv",
                               float_format='%.3f')
        
        if plot_parity_bounds == True:
                
            self.parity_plot(self.all_test_predictions[transform_idx][model_idx],
                                             self.all_train_predictions[transform_idx][model_idx],
                                             f'{self.uncertainty_path}/UQ_parity_bounds',
                                            with_bounds=True,
                                            bounds=cluster_bounds,
                                            bound_label=f"{method}_{metric}_{n_clusters}clusters")
                                
            
        return frac_inbounds, cluster_bounds
    
    
    def overlay_plot(self, model_idx, transform_idx, n_clusters, metric="mae", method="kmeans"):
        
        all_cluster_bounds = []
        fracs = []
        
        for n in n_clusters:
            
            frac, cluster_bounds = self.clustering_uncertainty_comparison(model_idx, 
                                                          transform_idx, 
                                                          n_clusters=n, 
                                                          metric=metric,
                                                          method=method
                                                          )
            all_cluster_bounds.append(cluster_bounds)
            fracs.append(frac)
            
        train_fracs = [frac["Train"] for frac in fracs]
        test_fracs = [frac["Test"] for frac in fracs]            
            
        # create figure and axis objects with subplots()
        fig = plt.figure()
        
        ax = fig.add_subplot(111)
        ax2 = fig.add_subplot(111, frame_on=False)
        
        flierprops = dict(marker='o', 
                          markersize=4,
                          linestyle='none', 
                          color="grey")
        
        ax.boxplot(all_cluster_bounds, 
                    labels = n_clusters,
                    flierprops=flierprops
                    )
        
        ax.set_xlabel("Number of clusters") #,fontsize=14
        # ax.set_xlim(0, len(n_clusters)+1)
        
        ax.set_ylabel("Error bound magnitude"
                      #color="red"
                      # , fontsize=14
                      )
        
        try:
            ax.set_ylim(np.asarray(all_cluster_bounds).min() - 0.5, 
                        np.asarray(all_cluster_bounds).max() + 0.5)
        except:
            ax.set_ylim(0.0 - 0.5, 
                        1.0 + 0.5)
        
        ax2.plot(n_clusters, 
                train_fracs, 
                color="red",
                marker = 'o', 
                markersize=8,
                mfc='none',
                label="Train",
                alpha=0.5)  
        
        ax2.plot(n_clusters, 
                test_fracs, 
                color="red",
                mfc='none',
                marker = '*', 
                markersize=8,
                label = "Test",
                alpha=0.5)
        
        ax2.xaxis.tick_top()
        ax2.set_xticks([])
        ax2.xaxis.set_label_position("top")
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        
        # ax2.set_xlabel("Number of clusters")
        ax2.set_ylabel("Fraction of data within its error bounds", color="red")
        #ax2.set_ylim(0,1)
        ax2.legend(bbox_to_anchor=(1.2, -0.1))
        ax2.margins(1, 1)
        ax2.set_xlim(n_clusters[0]-0.5, n_clusters[-1]+0.5) #  -3.2 +2.5 for 1, 5, 10, 15, ... 80 # figure out how to do this better!
        ax2.set_ylim(0, 1)
        
        try:
            plt.savefig(f"{self.uncertainty_path}/clusteringUQ_overlay_{self.target}_model{model_idx}_transform{transform_idx}_{method}_{metric}.png", 
                        bbox_inches='tight', 
                        dpi=self.dpi)
        except:
            # plt.show() # not sure why but putting this here solves an issue for plotting the pretrained case
            plt.savefig(f"{self.uncertainty_path}/clusteringUQ_overlay_{self.target}_pretrained_{method}_{metric}.png", 
                        bbox_inches='tight', 
                        dpi=self.dpi)
        plt.clf()
        
        
    def bounds_dist_plot(self, model_idx, transform_idx, n_clusters, metric="mae", method="kmeans", plot_type="boxplot"):
        
        all_cluster_bounds = []
        
        for n in n_clusters:
            
            _, cluster_bounds = self.clustering_uncertainty_comparison(model_idx, 
                                                          transform_idx, 
                                                          n_clusters=n, 
                                                          metric=metric,
                                                          method=method
                                                          )
            all_cluster_bounds.append(cluster_bounds)
        
        
        # plt.ylim(np.asarray(all_cluster_bounds).min() - 0.5, 
        #             np.asarray(all_cluster_bounds).max() + 0.5)
        
        try:
            plt.ylim(np.asarray(all_cluster_bounds).min() - 0.5, 
                        np.asarray(all_cluster_bounds).max() + 0.5)
        except:
            plt.ylim(0.0 - 0.5, 
                        1.0 + 0.5)
                
        # print(len(all_cluster_bounds[-1]), n_clusters)    
        plt.boxplot(all_cluster_bounds, 
                    labels = n_clusters
                    )
            
        # plt.legend(title="")
        # plt.xticks(rotation=30)
        plt.ylabel("Error bound")
        plt.xlabel("Number of clusters")
        # plt.tight_layout()
        # plt.grid(axis = 'y', alpha = 0.3, which="both")
#        plt.show()

        try:
            plt.savefig(f"{self.uncertainty_path}/clusteringUQ_{self.target}_model{model_idx}_transform{transform_idx}_{method}_{metric}_{plot_type}.png", 
                        bbox_inches='tight', 
                        dpi=self.dpi)
        except:
            # plt.show() # not sure why but putting this here solves an issue for plotting the pretrained case
            plt.savefig(f"{self.uncertainty_path}/clusteringUQ_{self.target}_pretrained_{method}_{metric}_{plot_type}.png", 
                        bbox_inches='tight', 
                        dpi=self.dpi)
        plt.clf()
            
    
    def n_clusters_plot(self, model_idx, transform_idx, n_clusters, metric="mae", method="kmeans"):
        
        fracs = []
        for n in n_clusters:
            
            frac, _ = self.clustering_uncertainty_comparison(model_idx, 
                                                          transform_idx, 
                                                          n_clusters=n, 
                                                          metric=metric,
                                                          method=method
                                                          )
            fracs.append(frac)

        train_fracs = [frac["Train"] for frac in fracs]
        test_fracs = [frac["Test"] for frac in fracs]
        
        fig, ax = plt.subplots()
        
        
        ax.plot(n_clusters, 
                train_fracs, 
                marker = 'o', 
                mfc='none',
                label="Train")  
        
        ax.plot(n_clusters, 
                test_fracs, 
                mfc='none',
                marker = 'o', 
                label = "Test")
        
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("Fraction within bounds")
        ax.set_ylim(0,1)
        ax.legend()
        
        # save to path??
        # model_path = f"{self.path}/{self.models[model_idx]}_{self.transform_names[transform_idx]}" 
        plt.savefig(f"{self.uncertainty_path}//n_clusters_plot_model{model_idx}_transform{transform_idx}_{method}_{metric}.png")
        plt.clf()

            
            
            
            #