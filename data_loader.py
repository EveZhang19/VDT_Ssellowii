import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, data_root):
        """
        Initializes paths for various GRN inference method outputs.
        """
        self.data_root = data_root
        self.scaler = MinMaxScaler()
        
        # Standardized internal paths
        self.paths = {
            'dream4': os.path.join(data_root, 'benchmarks/dream4_100'),
            'gnw': os.path.join(data_root, 'benchmarks/gnw_simu'),
            'gold_standard': os.path.join(data_root, 'benchmarks/gold_standard'),
            'swing': os.path.join(data_root, 'methods/swing'),
            'genie3': os.path.join(data_root, 'methods/genie3'),
            'dyngenie3': os.path.join(data_root, 'methods/dyngenie3'),
            'peak': os.path.join(data_root, 'methods/peak'),
            'op': os.path.join(data_root, 'methods/outpredict')
        }

    def load_dream4_data(self, network_id):
        path = os.path.join(self.paths['dream4'], f'insilico_size100_{network_id}_timeseries.tsv')
        data = pd.read_csv(path, sep='\t')
        return self._split_experiments(data, 210)

    def load_gnw_data(self, network_id):
        path = os.path.join(self.paths['gnw'], f'insilico_size100_{network_id}_dream4_timeseries.tsv')
        data = pd.read_csv(path, sep='\t')
        return self._split_experiments(data, 210)

    def load_gold_standard(self, network_id):
        path = os.path.join(self.paths['gold_standard'], f'gold_standard_size100_{network_id}.tsv')
        return pd.read_csv(path, sep='\t', header=None, names=['Gene1', 'Gene2', 'Interaction'])

    def _split_experiments(self, data, step):
        return [data.iloc[i:i+step] for i in range(0, len(data), step)]

    def get_gene_names(self, data):
        return data.columns[1:].tolist()

    # --- Method-specific data loaders ---

    def load_swing_data(self, net, exp):
        path = os.path.join(self.paths['swing'], f'net{net}_exp_{exp}.txt')
        df = pd.read_csv(path, sep='\t')
        return df[['Source', 'Target', 'mean_importance']].rename(
            columns={'Source': 'Gene1', 'Target': 'Gene2', 'mean_importance': 'Confidence'}
        )

    def load_genie3_data(self, net, exp):
        path = os.path.join(self.paths['genie3'], f'net{net}_exp_{exp}.txt')
        return pd.read_csv(path, sep='\t', names=['Gene1', 'Gene2', 'Confidence'])

    def load_dyngenie3_data(self, net, exp):
        path = os.path.join(self.paths['dyngenie3'], f'net{net}_exp_{exp}.txt')
        return pd.read_csv(path, sep='\t', names=['Gene1', 'Gene2', 'Confidence'])

    def load_peak_data(self, net, exp):
        path = os.path.join(self.paths['peak'], f'net{net}_{exp}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path, sep='\t', names=['Gene1', 'Gene2', 'Confidence'])
            df['Confidence'] = df['Confidence'].abs() 
            return df
        return None
            
    def load_op_data(self, net, exp):
        path = os.path.join(self.paths['op'], f'net{net}_{exp}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df[['TF', 'Target', 'Importance']].rename(
                columns={'TF': 'Gene1', 'Target': 'Gene2', 'Importance': 'Confidence'}
            )
        return None

    def load_all_method_data(self, network, experiment):
        """
        Merges predictions from all methods into a single feature matrix.
        """
        swing = self.load_swing_data(network, experiment)
        genie3 = self.load_genie3_data(network, experiment)
        dyngenie3 = self.load_dyngenie3_data(network, experiment)
        peak = self.load_peak_data(network, experiment)
        op = self.load_op_data(network, experiment)
        
        all_genes = set(swing['Gene1']).union(set(swing['Gene2']))
        all_pairs = pd.DataFrame(
            [(g1, g2) for g1 in all_genes for g2 in all_genes if g1 != g2],
            columns=['Gene1', 'Gene2']
        )
        
        # Incremental merging
        combined = all_pairs.merge(swing, on=['Gene1', 'Gene2'], how='left')
        combined = combined.merge(genie3, on=['Gene1', 'Gene2'], how='left', suffixes=('_swing', '_genie3'))
        combined = combined.merge(dyngenie3, on=['Gene1', 'Gene2'], how='left', suffixes=('', '_dyngenie3'))
        combined = combined.merge(peak, on=['Gene1', 'Gene2'], how='left', suffixes=('', '_peak'))
        combined = combined.merge(op, on=['Gene1', 'Gene2'], how='left', suffixes=('', '_op'))
        
        combined = combined.rename(columns={
            'Confidence': 'Confidence_dyngenie3'
        }).fillna(0)
        
        # Normalize confidence scores
        conf_cols = [col for col in combined.columns if 'Confidence' in col]
        combined[conf_cols] = self.scaler.fit_transform(combined[conf_cols])
        
        return combined
