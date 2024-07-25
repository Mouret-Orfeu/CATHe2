import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for environments without a display server
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Function to plot all F1 scores in the dataframe on the same plot as a bar plot
def plot_all_f1_scores(dataframe):
    """
    Plots all F1 scores in the dataframe on the same plot as a bar plot and saves the plot.
    
    :param dataframe: pandas DataFrame containing the results.
    """
    # Create a unique identifier for each combination of parameters
    dataframe['Parameters'] = dataframe.apply(lambda row: f'Nb_Layer_Block={row["Nb_Layer_Block"]}, Dropout={row["Dropout"]}', axis=1)
    
    # Calculate the maximum F1 score for each model
    max_f1_scores = dataframe.groupby('Model')['F1_Score'].max().sort_values(ascending=False)
    sorted_models = max_f1_scores.index.tolist()
    
    # Add a column for the sorted model order
    dataframe['Model_Order'] = pd.Categorical(dataframe['Model'], categories=sorted_models, ordered=True)
    
    # Sort the dataframe by the new Model_Order column
    dataframe = dataframe.sort_values(by=['Model_Order', 'Parameters'])
    
    fig = go.Figure()

    # Add bars for each parameter combination
    params = dataframe['Parameters'].unique()

    for param in params:
        param_data = dataframe[dataframe['Parameters'] == param]
        fig.add_trace(go.Bar(
            x=param_data['Model_Order'],
            y=param_data['F1_Score'],
            name=param,
            hoverinfo='text',
            text=param_data.apply(lambda row: f'Model={row["Model"]}<br>Nb_Layer_Block={row["Nb_Layer_Block"]}<br>Dropout={row["Dropout"]}<br>F1_Score={row["F1_Score"]:.4f}', axis=1)
        ))

    fig.update_layout(
        title='F1 Score Evolution for All Models and Configurations',
        xaxis_title='Model',
        yaxis_title='F1 Score',
        barmode='group',
        legend_title='Parameters',
        bargap=0.2,  # Increase the gap between bars
        bargroupgap=0.1  # Increase the gap between groups of bars
    )

    # Save the plot as an HTML file
    plot_filename = './results/f1_score_plots/f1_score_all_models.html'
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)

    fig.write_html(plot_filename)

    # Optionally, display the plot in the browser
    fig.show()



# Function to plot the evolution of F1 scores
def plot_f1_score_evolution(dataframe, x_param, models_to_plot, **conditions):
    """
    Plots the F1 score evolution for selected models along a specified parameter.
    
    :param dataframe: pandas DataFrame containing the results.
    :param x_param: The parameter to plot on the x-axis (e.g., 'Nb_Layer_Block', 'Dropout').
    :param models_to_plot: List of models to include in the plot.
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
    sns.lineplot(data=df_filtered, x=x_param, y='F1_Score', hue='Model', marker='o')
    plot_title = f'F1 Score Evolution along {x_param}'
    if condition_str:
        plot_title += condition_str.replace('_', ' ').replace('=', ' = ')
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

# Plot F1 score evolution along 'Nb_Layer_Block' with Dropout = 0.1 for all models
plot_f1_score_evolution(df, 'Nb_Layer_Block', models_to_plot, Dropout=0.5)

# Plot F1 score evolution along 'Nb_Layer_Block' with Dropout = 0.1 for all models
plot_f1_score_evolution(df, 'Nb_Layer_Block', models_to_plot, Dropout=0.1, Nb_Layer_Block=1)

# Plot F1 score evolution along 'Nb_Layer_Block' with Dropout = 0 for all models
plot_f1_score_evolution(df, 'Nb_Layer_Block', models_to_plot, Dropout=0, Nb_Layer_Block=1)

# Plot all F1 scores in the dataframe
plot_all_f1_scores(df)

# Plot F1 score evolution along 'Dropout' for ProstT5_half with 1 Layer block
plot_f1_score_evolution(df, 'Dropout', ['ProstT5_half'], Nb_Layer_Block=1)

# Plot F1 score evolution along 'Dropout' for ProstT5_full with 1 Layer block
plot_f1_score_evolution(df, 'Dropout', ['ProstT5_full'], Nb_Layer_Block=1)

# Plot F1 score evolution along 'Dropout' for Ankh_base with 1 Layer block
plot_f1_score_evolution(df, 'Dropout', ['Ankh_base'], Nb_Layer_Block=1)

# Plot F1 score evolution along 'Dropout' for Ankh_large with 1 Layer block
plot_f1_score_evolution(df, 'Dropout', ['Ankh_large'], Nb_Layer_Block=1)

# Plot F1 score evolution along 'Dropout' for TM_Vec with 1 Layer block
plot_f1_score_evolution(df, 'Dropout', ['TM_Vec'], Nb_Layer_Block=1)
