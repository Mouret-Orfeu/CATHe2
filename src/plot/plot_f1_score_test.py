import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Switch backend to Agg
import matplotlib
matplotlib.use('Agg')

# Load the dataset
file_path = './results/perf_ProstT5.csv'
data = pd.read_csv(file_path)

# Determine the most common values for each parameter to use as fixed values
fixed_values = {
    'Nb_Layer_Block': data['Nb_Layer_Block'].mode()[0],
    'Dropout': data['Dropout'].mode()[0],
    'Layer_size': data['Layer_size'].mode()[0],
    'pLDDT_threshold': data['pLDDT_threshold'].mode()[0]
}

# Create the directory if it doesn't exist
output_dir = './results/f1_score_plots/ProstT5_full'
os.makedirs(output_dir, exist_ok=True)

# Filter data based on the selected fixed values
def filter_data(fixed_params, varying_param):
    filtered_data = data.copy()
    for param, value in fixed_params.items():
        if param != varying_param:
            filtered_data = filtered_data[filtered_data[param] == value]
    return filtered_data

# Function to save plots
def save_plot(x_param, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=x_param, y='F1_Score', hue='Input_Type', data=filter_data(fixed_values, x_param), marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    
    # Save the plot
    filename = f"{output_dir}/F1_Score_vs_{x_param}_fixed_NbLayerBlock_{fixed_values['Nb_Layer_Block']}_Dropout_{fixed_values['Dropout']}_LayerSize_{fixed_values['Layer_size']}_pLDDT_{fixed_values['pLDDT_threshold']}.png"
    plt.savefig(filename)
    plt.close()

# Generate and save each plot
save_plot('Nb_Layer_Block', 'F1 Score vs Number of Layer Blocks (Comparison Across Input Types)', 'Number of Layer Blocks', 'F1 Score')
save_plot('Dropout', 'F1 Score vs Dropout Rate (Comparison Across Input Types)', 'Dropout Rate', 'F1 Score')
save_plot('Layer_size', 'F1 Score vs Layer Size (Comparison Across Input Types)', 'Layer Size', 'F1 Score')
save_plot('pLDDT_threshold', 'F1 Score vs pLDDT Threshold (Comparison Across Input Types)', 'pLDDT Threshold', 'F1 Score')
