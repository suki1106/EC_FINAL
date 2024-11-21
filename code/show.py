import pandas as pd
import matplotlib.pyplot as plt
import os


def process_testcases_in_directory():

    for i in range(10):
        output_csv_path = f'./result/pred_{i}.csv'
        target_csv_path = f'./result/target_{i}.csv'

        output_png_path = f'./result/output_{i}.png'
        
        output_df = pd.read_csv(output_csv_path)
        target_df = pd.read_csv(target_csv_path)
        

        
        output_data = output_df.to_numpy(dtype=float)
        target_data = target_df.to_numpy(dtype=float)

        
        vmin = min(output_data.min(), target_data.min())
        vmax = max(output_data.max(), target_data.max())
        
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        

        output_heatmap = axes[0].imshow(output_data, interpolation='nearest', vmin=vmin, vmax=vmax)
        axes[0].set_title('Output Heatmap')
        plt.colorbar(output_heatmap, ax=axes[0])  
        
        target_heatmap = axes[1].imshow(target_data, interpolation='nearest', vmin=vmin, vmax=vmax)
        axes[1].set_title('Target Heatmap')
        plt.colorbar(target_heatmap, ax=axes[1])  
        
        plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
        plt.close() 


if __name__ == "__main__":
    process_testcases_in_directory()