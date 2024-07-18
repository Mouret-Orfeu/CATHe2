import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib

# Set the backend to 'Agg' for non-interactive plotting
matplotlib.use('Agg')

def save_confusion_matrix(csv_file, output_image):
    # Read the sparse confusion matrix CSV file
    df = pd.read_csv(csv_file, header=None, names=['row', 'col', 'value'])
    
    # Determine the size of the confusion matrix
    size = max(df['row'].max(), df['col'].max()) + 1
    print(f"Size of confusion matrix: {size}")
    
    # Create an empty confusion matrix
    confusion_matrix = np.zeros((size, size), dtype=int)
    
    # Fill the confusion matrix
    for _, row in df.iterrows():
        confusion_matrix[row['row'], row['col']] = row['value']
    
    
    # Create a custom color map that makes zero cells white
    cmap = plt.cm.viridis
    cmap.set_under('white')
    
    # Plot the heatmap with logarithmic scaling
    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_matrix, annot=False, fmt="d", cmap=cmap, vmin=0.01, cbar_kws={'label': 'Log Scale'})
    
    # Create a purple to yellow color map for the annotations
    norm = mcolors.Normalize(vmin=confusion_matrix.min(), vmax=confusion_matrix.max())
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
    
    # Emphasize non-zero values by adding colored annotations
    for i in range(size):
        for j in range(size):
            if confusion_matrix[i, j] > 0:
                plt.text(j + 0.5, i + 0.5, confusion_matrix[i, j],
                         horizontalalignment='center',
                         verticalalignment='center',
                         fontsize=6, color=sm.to_rgba(confusion_matrix[i, j]))
    
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix Heatmap (Log Scale)')
    plt.savefig(output_image, bbox_inches='tight')
    plt.close()

# Example usage
save_confusion_matrix('./results/confusion_matrices/ProstT5_full.csv', './results/confusion_matrices/ProstT5_full.png')
