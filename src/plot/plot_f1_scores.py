import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for environments without a display server
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


# Function to plot all F1 scores in the dataframe on the same plot as a bar plot
def plot_all_f1_scores(dataframe, list_model_to_show):
    """
    Plots all F1 scores in the dataframe on the same plot as a bar plot and saves the plot.
    
    :param dataframe: pandas DataFrame containing the results.
    :param list_model_to_show: List of model names to show in the plot.
    """
    # Define color mapping based on Input_Type
    color_map = {
        'AA': 'blue',
        '3Di': 'orange',
        'AA+3Di': 'red'
    }

    # Filter the dataframe to include only the models in the list_model_to_show
    dataframe = dataframe[dataframe['Model'].isin(list_model_to_show)]

    # Create a unique identifier for each combination of parameters
    dataframe['Parameters'] = dataframe.apply(lambda row: f'Nb_Layer_Block={row["Nb_Layer_Block"]}, Dropout={row["Dropout"]}, Input_Type={row["Input_Type"]}, Layer_size={row["Layer_size"]}, pLDDT_threshold={row["pLDDT_threshold"]}', axis=1) 
    
    # Sort the dataframe by Model, then by F1_Score in descending order
    dataframe = dataframe.sort_values(by=['Model', 'F1_Score'], ascending=[True, False])

    # Calculate the maximum F1 score for each model
    max_f1_scores = dataframe.groupby('Model')['F1_Score'].max().sort_values(ascending=False)
    sorted_models = max_f1_scores.index.tolist()
    
    # Add a column for the sorted model order
    dataframe['Model_Order'] = pd.Categorical(dataframe['Model'], categories=sorted_models, ordered=True)
    
    # Sort the dataframe by Model_Order to ensure correct order in plot
    dataframe = dataframe.sort_values(by=['Model_Order', 'F1_Score'], ascending=[True, False])
    
    fig = go.Figure()

    # Add bars for each parameter combination with color based on Input_Type
    for _, row in dataframe.iterrows():
        fig.add_trace(go.Bar(
            x=[row['Model_Order']],
            y=[row['F1_Score']],
            name=row['Parameters'],
            marker_color=color_map.get(row['Input_Type'], 'gray'),  # Use gray if Input_Type is not in the color_map
            hoverinfo='text',
            text=f'Model={row["Model"]}<br>Nb_Layer_Block={row["Nb_Layer_Block"]}<br>Dropout={row["Dropout"]}<br>Input_Type={row["Input_Type"]}<br>Layer_size={row["Layer_size"]}<br>pLDDT_threshold={row["pLDDT_threshold"]}<br>F1_Score={row["F1_Score"]:.4f}'
        ))

    fig.update_layout(
        title='F1 Score Evolution for Selected Models and Configurations',
        xaxis_title='Model',
        yaxis_title='F1 Score',
        barmode='group',
        legend_title='Parameters',
        bargap=0.2,  # Increase the gap between bars
        bargroupgap=0.1  # Increase the gap between groups of bars
    )

    # Save the plot as an HTML file
    plot_filename = './results/f1_score_plots/f1_score_selected_models.html'
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)

    fig.write_html(plot_filename)

    # Optionally, display the plot in the browser
    fig.show()



# Function to plot the evolution of F1 scores
def plot_f1_score_evolution(dataframe, x_param, models_to_plot, title=None, **conditions):
    """
    Plots the F1 score evolution for selected models along a specified parameter.
    
    :param dataframe: pandas DataFrame containing the results.
    :param x_param: The parameter to plot on the x-axis (e.g., 'Nb_Layer_Block', 'Dropout', 'Input_Type').
    :param models_to_plot: List of models to include in the plot.
    :param title: Optional title for the graph. If not provided, a title will be generated based on conditions.
    :param conditions: Dictionary of conditions to filter the data (optional).
    """
    # Filter the dataframe for the selected models
    df_filtered = dataframe[dataframe['Model'].isin(models_to_plot)]
    
    # Apply the condition filters if provided
    condition_str = ''
    for param, value in conditions.items():
        if value is not None:
            df_filtered = df_filtered[df_filtered[param] == value]
            condition_str += f'_{param}_{value}'    
    
    # Determine the models being plotted
    all_models = {'ProtT5', 'ESM2', 'Ankh_large', 'Ankh_base', 'ProstT5_full', 'ProstT5_half', 'TM_Vec'}
    models_in_plot = set(models_to_plot)
    if models_in_plot == all_models:
        models_str = 'all'
    else:
        models_str = '_'.join(sorted(models_in_plot))

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_filtered, x=x_param, y='F1_Score', hue='Model', marker='o', errorbar=None)

    # If a title is provided, use it; otherwise, construct the title
    if title:
        plot_title = title
    else:
        # Construct the title based on conditions
        plot_title = f'F1 Score Evolution along {x_param}: '
        condition_str = ', '.join(f'{param}={value}' for param, value in conditions.items() if value is not None)
        if condition_str:
            plot_title += condition_str  # Directly use the formatted condition_str

    # DEBUG
    print(f'Plotting: {plot_title}')

    plt.title(plot_title)
    plt.xlabel(x_param)
    plt.ylabel('F1 Score')
    plt.legend(title='Model')
    plt.grid(True)
    
    # Save the plot
    plot_filename = f'./results/f1_score_plots/f1_score_evolution_{x_param}_{models_str}{condition_str}.png'

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)

    plt.savefig(plot_filename)
    plt.close()  # Use plt.close() instead of plt.show() for 'Agg' backend


# Load the dataframe
df_results_path = './results/perf_dataframe.csv'
df = pd.read_csv(df_results_path)

# Example usage
models_to_plot = ['ProtT5', 'ESM2', 'Ankh_large', 'Ankh_base', 'ProstT5_full', 'ProstT5_half', 'TM_Vec']

list_model_to_show = ['ProtT5', 'ESM2', 'Ankh_large', 'Ankh_base', 'ProstT5_full', 'ProstT5_half', 'TM_Vec']

# Plot the F1 score evolution along 'Dropout' for all models with the specified conditions

models_to_plot = ['ProstT5_full']


# Example usage:
input_types = 'AA+3Di'
x_param = 'pLDDT_threshold'
Dropout = 0.3
layer_size = 2048
nb_layer_block = 2
is_top_50_SF = True


# Plot F1 score evolution
plot_f1_score_evolution(
    dataframe=df, 
    x_param=x_param, 
    models_to_plot=models_to_plot, 
    title = 'F1 Score relatively to pLDDT threshold, top 50 SF training',
    Input_Type=input_types, 
    Dropout=Dropout,
    Layer_size=layer_size, 
    Nb_Layer_Block=nb_layer_block,
    is_top_50_SF = is_top_50_SF
)

def plot_f1_score_evolution_unique_model(dataframe, x_param, model, input_types, **conditions):
    """
    Plots the F1 score evolution for a selected model along a specified parameter,
    with different curves for each input type.

    :param dataframe: pandas DataFrame containing the results.
    :param x_param: The parameter to plot on the x-axis (e.g., 'Nb_Layer_Block', 'Dropout', 'Input_Type').
    :param model: The model to plot (single model name as a string).
    :param input_types: List of input types to plot (e.g., ['AA', '3Di', 'AA+3Di']).
    :param conditions: Dictionary of additional conditions to filter the data (optional).
    """
    # Filter the dataframe for the selected model
    df_filtered = dataframe[dataframe['Model'] == model]
    
    # Apply the condition filters with a special case for 'AA'
    condition_str = ''
    filtered_dfs = []
    for input_type in input_types:
        df_subset = df_filtered[df_filtered['Input_Type'] == input_type]
        for param, value in conditions.items():
            if value is not None:
                if input_type == 'AA' and param == 'pLDDT_threshold':
                    # Force pLDDT_threshold to be 0 for 'AA'
                    df_subset = df_subset[df_subset[param] == 0]
                else:
                    df_subset = df_subset[df_subset[param] == value]

                    # DEBUG
                    print(f'Filtering: {param}={value}')

                condition_str += f'_{param}_{value}'
        filtered_dfs.append(df_subset)
    
    # Concatenate all the filtered subsets
    df_filtered = pd.concat(filtered_dfs)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_filtered, x=x_param, y='F1_Score', hue='Input_Type', marker='o')
    
    # Construct the plot title
    plot_title = f'F1 Score Evolution along {x_param} for {model}: '
    if condition_str:
        plot_title += condition_str.replace('_', ', ').replace('=', '=')

    plt.title(plot_title)
    plt.xlabel(x_param)
    plt.ylabel('F1 Score')
    plt.legend(title='Input_Type')
    plt.grid(True)
    
    # Save the plot with the model name in the filename
    plot_filename = f'./results/f1_score_plots/f1_score_evolution_{x_param}_{model}{condition_str}.png'

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)

    plt.savefig(plot_filename)
    plt.close()  # Use plt.close() instead of plt.show() for 'Agg' backend

# Example usage:
# input_types = ['AA', '3Di']
# model = 'ProstT5_full'
# x_param = 'Dropout'
# plddt_threshold = 24
# layer_size = 1024
# dropout = 0.1
# nb_layer_block = 2

# plot_f1_score_evolution_unique_model(
#     dataframe=df, 
#     x_param=x_param, 
#     model=model, 
#     input_types=input_types, 
#     Layer_size=layer_size, 
#     Dropout=dropout,
#     pLDDT_threshold=plddt_threshold  # Apply pLDDT_threshold=24 except for 'AA'
# )


