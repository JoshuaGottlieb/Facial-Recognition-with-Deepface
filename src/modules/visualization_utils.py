import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_distribution(preprocess_type, metric, ax):
    intra_path = os.path.join(f'./data/vectorized/{preprocess_type}_intra_distances_{metric}.csv')
    intra_df = pd.read_csv(intra_path)
    inter_df = dm.get_unbatched_frame(f'{preprocess_type}_', metric)
    intra_df['Type'] = 'Match'
    inter_df['Type'] = 'Mismatch'
    df = pd.concat([intra_df, inter_df])
    
    if metric == 'l2':
        x = 'l2_distances'
        title = f'L2 Distances between Images, {preprocess_type.spli
    elif metric == 'cos':
        x = 'cosine_distances'
    
    sns.histplot(data = l2_df, x = f'l2_distances', hue = 'Type',
             stat = 'proportion', common_norm = False, bins = 25, ax = ax);

    ax.set_title('L2 Distances between Images, Padded', fontsize = 20);
    ax.set_xlabel('Euclidean Distance', fontsize = 16);
    ax.set_ylabel('Proportion', fontsize = 16);

    plt.setp(ax.get_legend().get_texts(), fontsize = 16);
    plt.setp(ax.get_legend().get_title(), fontsize = 18);

    ax.tick_params(axis = 'both', labelsize = 14);