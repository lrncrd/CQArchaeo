
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math
import seaborn as sns
import os
from tqdm import tqdm


def cqa_analysis(data, min_data_sample = 7, max_data_sample = 200, min_quantum = 4, max_quantum = 24, step = 0.02, Montecarlo_sim = True, mc_parameter = 0.15, mc_iterations = 100):
    """
    function to perform the CQA analysis on a given dataset and return the results into a quantogram plot.

    Parameters
    ----------
    data : str
        Path to the file to be imported. The file must be in .csv or .xlsx format and must have only one column.
    min_data_sample : int, optional
        Minimum boundary of the weight to be analysed. The default is 7.
    max_data_sample : int, optional
        Maximum boundary of the weight to be analysed. The default is 200.
    min_quantum : int, optional
        Minimum boundary of the quantum to be analysed. The default is 4.
    max_quantum : int, optional
        Maximum boundary of the quantum to be analysed. The default is 24.
    step : float, optional
        Step of the quantum to be analysed. The default is 0.02.
    Montecarlo_sim : bool, optional
        If True, the Montecarlo simulation is performed. The default is True.
    mc_parameter : float, optional
        Parameter of the Montecarlo simulation. The default is 0.15.
    mc_iterations : int, optional
        Number of iterations of the Montecarlo simulation. The default is 100.
        
    """

    (_, ext) = os.path.splitext(data)

    if ext == '.csv':
        data = pd.read_csv(data)
    elif ext == '.xlsx':
        data = pd.read_excel(data)

    else:
        raise Exception("Format not supported, please use csv (.csv) or excel (.xlsx) file.")

    if len(data.columns) > 1:
        raise Exception("The file to be imported must have only one column.")
    

    cqa_params = [min_data_sample, max_data_sample, min_quantum, max_quantum]
    for param in cqa_params:
        if type(param) != int:
            raise Exception("The values 'min_data_sample', 'max_data_sample', 'min_quantum', 'max_quantum' must be int.")

    # basic data handling
    db_2_analyse = data[(data >= min_data_sample) & (data < max_data_sample)]
    df=db_2_analyse.dropna()
    quanta_arr = np.arange(min_quantum, max_quantum + step, step)


    # Kendall formula
    cosine_results = pd.DataFrame()
    phi_q_df = pd.DataFrame()
    ###
    sample_size = df.count()
    coeff = 2 / sample_size
    sample_coefficient = np.sqrt(coeff).values[0]



    cosine_results = pd.concat([(2 * math.pi * df.iloc[:, 0]) / q for q in quanta_arr], axis=1)
    cosine_results.columns = [f'Results {round(q, 2)}' for q in quanta_arr]
    cosine_results = np.cos(cosine_results)



    cosine_sum_series = cosine_results.sum(axis=0)
    cosine_sum = cosine_sum_series.to_frame()

    phi_q_df = cosine_sum.multiply(sample_coefficient)
    phi_q_df.columns = ['Phi_q_values']
    phi_q_df['Quanta'] = quanta_arr

    phi_q_df.reset_index(drop=True, inplace=True)


    ### Select best quantum and print results

    max_value = phi_q_df['Phi_q_values'].max()
    quantum_max = phi_q_df.loc[phi_q_df['Phi_q_values'].idxmax(), 'Quanta']
    print(f"Highest 'Phi_q_values': {max_value:.2f}, corrisponding to quantum: {quantum_max:.2f}")

    if Montecarlo_sim == True:
    ### Montecarlo simulation
        def apply_variation(df_copy, mc_parameter):
            percent_diff = df_copy * mc_parameter
            random_matrix = np.random.uniform(-1, 1, size=df_copy.shape)
            variations = random_matrix * percent_diff
            return df_copy + variations

        final_mc_df = pd.DataFrame()
        intermediate_dfs = []

        for i in range(mc_iterations):
            df_copy = df.copy()
            mc_df = apply_variation(df_copy, mc_parameter)
            intermediate_dfs.append(mc_df[df.columns[0]]) 
            
        final_mc_df = pd.concat(intermediate_dfs, axis=1)
        final_mc_df.columns = [f'MC_{i+1}' for i in range(mc_iterations)]


        final_mc_df.reset_index(drop=True, inplace=True)

        Montecarlo_phi_q = pd.DataFrame()

        def calculate_Phi_q_values_mc(df, quanta_arr):
            Phi_q_values_mc_list = []
            
            for q in quanta_arr:
                cosine_result_mc = np.cos((2 * math.pi * df.iloc[:, 0]) / q)
                Phi_q_values_mc = np.sum(cosine_result_mc) * sample_coefficient
                Phi_q_values_mc_list.append(Phi_q_values_mc)
                
            return Phi_q_values_mc_list

        Montecarlo_phi_q_values = []
        completed_cols = 0

        for col in tqdm(final_mc_df.columns):
            df = final_mc_df[[col]]  
            Phi_q_values_mc_list = calculate_Phi_q_values_mc(df, quanta_arr)
            Montecarlo_phi_q_values.append(Phi_q_values_mc_list)
            
            completed_cols += 1

        Montecarlo_phi_q = pd.DataFrame(Montecarlo_phi_q_values, columns=quanta_arr).T

        Montecarlo_phi_q.columns = final_mc_df.columns

        alpha_1_share = round(mc_iterations * 0.01)
        alpha_5_share = round(mc_iterations * 0.05)
        mc_max_column = Montecarlo_phi_q.max()
        alpha_1 = mc_max_column.nlargest(alpha_1_share).iloc[-1]
        alpha_5 = mc_max_column.nlargest(alpha_5_share).iloc[-1]


    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(data=phi_q_df, x=phi_q_df['Quanta'], y=phi_q_df['Phi_q_values'], color='black', label='Quantogram', ax=ax, linewidth=2)
    if Montecarlo_sim == True:
        ax.axhline(y=alpha_1, color='r', linestyle='--', label='alpha_1', linewidth=1)
        ax.axhline(y=alpha_5, color='g', linestyle='--', label='alpha_5', linewidth=1)

    quantum_max = phi_q_df.loc[phi_q_df['Phi_q_values'].idxmax(), 'Quanta']
    ax.axvline(x=quantum_max, color='grey', linestyle='--', label='Best quantum', linewidth=1)

    ax.set_xlim(quanta_arr[0], quanta_arr[-1])
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelsize=8)

    plt.xticks(np.linspace(quanta_arr[0], quanta_arr[-1], 5))
    plt.legend()
    plt.grid(axis='y', linewidth=0.5)

    plt.xlabel('Quanta')
    plt.ylabel('Phi_q_values')
    plt.title('Quantogram')
    



    return phi_q_df