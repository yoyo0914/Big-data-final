import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import gc
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SimpleClustering:
    def __init__(self, n_clusters, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        
    def load_data(self, filepath):
        print(f"Loading {filepath}...")
        data = pd.read_csv(filepath)
        print(f"Loaded {len(data):,} samples")
        return data
    
    def preprocess_data(self, data):
        feature_cols = [col for col in data.columns if col.lower() not in ['id', 'index']]
        X = data[feature_cols].values.astype(np.float32)
        X_scaled = self.scaler.fit_transform(X)
        del data
        gc.collect()
        return X_scaled
    
    def cluster_data(self, X):
        print(f"Clustering {len(X):,} samples into {self.n_clusters} clusters...")
        
        with tqdm(desc="Key-dimension analysis", unit="step") as pbar:
            X_key = X[:, [1, 2]]  # S2, S3維度
            
            main_kmeans = MiniBatchKMeans(
                n_clusters=5, 
                random_state=self.random_state,
                batch_size=5000,
                max_iter=300,
                n_init=10  # 增加嘗試次數
            )
            main_groups = main_kmeans.fit_predict(X_key)
            pbar.update(1)
        
        with tqdm(desc="Hierarchical subdivision", unit="group") as pbar:
            final_labels = np.zeros(len(X), dtype=int)
            current_label = 0
            
            for group_id in range(5):
                mask = main_groups == group_id
                group_size = np.sum(mask)
                
                if group_size == 0:
                    continue
                
                X_group = X[mask]  
                
                if group_size >= 3:
                    sub_kmeans = MiniBatchKMeans(
                        n_clusters=3,
                        random_state=self.random_state + group_id, 
                        batch_size=min(2000, group_size),
                        max_iter=200,
                        n_init=5
                    )
                    sub_labels = sub_kmeans.fit_predict(X_group)
                    
                    group_indices = np.where(mask)[0]
                    for i, idx in enumerate(group_indices):
                        final_labels[idx] = current_label + sub_labels[i]
                else:
                    group_indices = np.where(mask)[0]
                    for i, idx in enumerate(group_indices):
                        final_labels[idx] = current_label + (i % 3)
                
                current_label += 3
                pbar.update(1)
        
        unique_clusters = len(np.unique(final_labels))
        if unique_clusters != self.n_clusters:
            with tqdm(desc="Final adjustment", unit="step") as pbar:
                adjust_kmeans = MiniBatchKMeans(
                    n_clusters=self.n_clusters,
                    random_state=self.random_state,
                    batch_size=3000,
                    max_iter=200,
                    n_init=5
                )
                final_labels = adjust_kmeans.fit_predict(X)
                pbar.update(1)
        
        print(f"Generated {len(np.unique(final_labels))} clusters")
        return final_labels
    
    def process_dataset(self, filepath, output_filepath):
        data = self.load_data(filepath)
        X = self.preprocess_data(data)
        final_labels = self.cluster_data(X)
        
        print("Saving results...")
        result_df = pd.DataFrame({
            'id': range(len(final_labels)),
            'label': final_labels
        })
        result_df.to_csv(output_filepath, index=False)
        print(f"Saved to {output_filepath}")
        
        del X, final_labels, result_df
        gc.collect()
        return True

def main():
    print("Starting clustering analysis...")
    
    try:
        clusterer = SimpleClustering(n_clusters=15)
        clusterer.process_dataset('public_data.csv', 'r13944045_public.csv')
        print("Public dataset completed")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        clusterer = SimpleClustering(n_clusters=23)
        clusterer.process_dataset('private_data.csv', 'r13944045_private.csv')
        print("Private dataset completed")
        
    except FileNotFoundError:
        print("Private dataset not found")
    except Exception as e:
        print(f"Private dataset error: {e}")
    
    print("Done")

if __name__ == "__main__":
    main()