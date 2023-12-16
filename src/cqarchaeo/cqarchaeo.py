
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math
import seaborn as sns
import os
from tqdm import tqdm


class CQAnalysis:
    def __init__(self, data, min_value = 5, max_value = 200, 
                 min_quantum = 5, max_quantum = 24, step = 0.02, 
                 Montecarlo_sim = True, 
                 mc_parameter = 0.15, mc_iterations = 100):
        """
        Perform a quantogram analysis on a given dataset.

        Parameters
        ----------
            data (str) 
                path to the file to be imported. The file must be a csv or excel file with only one column.
            min_value (int) 
                minimum number of data to be considered (default: 5).
            max_value (int) 
                maximum number of data to be considered (default: 200).
            min_quantum (int)
                minimum quantum to be considered (default: 5). 
            max_quantum (int) 
                maximum quantum to be considered (default: 24).
            step (float) 
                step between two consecutive quanta (default: 0.02).
            Montecarlo_sim (bool) 
                if True, a Montecarlo simulation is performed (default: True).
            mc_parameter (float) 
                The percentage by which you want to randomize the initial sample data for Monte Carlo simulation. Each element of the initial sample will take a value within a probability range determined by `mc_parameter`.(default: 0.15).
            mc_iterations (int)
                number of iterations for the Montecarlo simulation. Must be a minimum of 100 (default: 100).

        Returns
        -------
            CQAnalysis object

        Examples
        --------
        >>> from cqarcheo import CQAnalysis
        >>> cqa = CQAnalysis(data = 'data.csv', min_value = 5, max_value = 200,
                            min_quantum = 5, max_quantum = 24, step = 0.02,
                            Montecarlo_sim = True, mc_parameter = 0.15, mc_iterations = 100)
        >>> cqa
        Quantum: 12.020, φ(q) value: 0.997        
        
        Notes
        -----
        Quantogram analysis is performed according to Kendall (1974):
  
        Bibliography
        ------------
        Kendall, D.G., 1974. Hunting quanta. Phil. Trans. R. Soc. Lond. A 276, 231–266. https://doi.org/10.1098/rsta.1974.0020
        
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
        
        if mc_iterations < 100:
            raise Exception("The number of Montecarlo iterations must be at least 100.")
        

        cqa_params = [min_value, max_value, min_quantum, max_quantum]
        for param in cqa_params:
            if type(param) != int:
                raise Exception("The values 'min_value', 'max_value', 'min_quantum', 'max_quantum' must be int.")
            

        ### Other parameters???
            
        self.data = data
        self.min_value = min_value
        self.max_value = max_value
        self.min_quantum = min_quantum
        self.max_quantum = max_quantum
        self.step = step
        self.Montecarlo_sim = Montecarlo_sim
        self.mc_parameter = mc_parameter
        self.mc_iterations = mc_iterations

        self.data_processed = self.data[(self.data >= self.min_value) & (self.data < self.max_value)].dropna()

        ### Kendall formula
        self.quanta_arr = np.arange(min_quantum, max_quantum + step, step)


        cosine_results = pd.DataFrame()
        phi_q_df = pd.DataFrame()

        sample_size = self.data_processed.count()
        coeff = 2 / sample_size
        sample_coefficient = np.sqrt(coeff).values[0]

        cosine_results = pd.concat([(2 * math.pi * self.data_processed.iloc[:, 0]) / q for q in self.quanta_arr], axis=1)
        cosine_results.columns = [f'Results {round(q, 2)}' for q in self.quanta_arr]
        cosine_results = np.cos(cosine_results)

        cosine_sum_series = cosine_results.sum(axis=0)
        cosine_sum = cosine_sum_series.to_frame()

        phi_q_df = cosine_sum.multiply(sample_coefficient)
        phi_q_df.columns = ['Phi_q_values']
        phi_q_df['Quanta'] = self.quanta_arr

        phi_q_df.reset_index(drop=True, inplace=True)

        phi_q_max_value = phi_q_df['Phi_q_values'].max()
        quantum_max = phi_q_df.loc[phi_q_df['Phi_q_values'].idxmax(), 'Quanta']
        print(f"Highest 'φ(q)': {phi_q_max_value:.2f}, corrisponding to quantum: {quantum_max:.2f}")


        self.phi_q_df = phi_q_df
        self.quantum_max = quantum_max
        self.phi_q_max_value = phi_q_max_value

        self.tab_res = pd.DataFrame(np.array([self.quantum_max, self.phi_q_max_value]).reshape(1, 2), columns=['Associated Quantum', 'Highest Phi_q_values'])
        
        if self.Montecarlo_sim == True:
        ### Montecarlo simulation

        # Define some useful functions:
            def apply_variation(df_copy, mc_parameter):
                percent_diff = df_copy * mc_parameter
                random_matrix = np.random.uniform(-1, 1, size=df_copy.shape)
                variations = random_matrix * percent_diff
                return df_copy + variations
            
            def calculate_Phi_q_values_mc(df, quanta_arr):
                Phi_q_values_mc_list = []
                
                for q in quanta_arr:
                    cosine_result_mc = np.cos((2 * math.pi * df.iloc[:, 0]) / q)
                    Phi_q_values_mc = np.sum(cosine_result_mc) * sample_coefficient
                    Phi_q_values_mc_list.append(Phi_q_values_mc)
                    
                return Phi_q_values_mc_list
            

            final_mc_df = pd.DataFrame()
            intermediate_dfs = []

            for _ in range(self.mc_iterations):
                df_copy = self.data_processed.copy()
                mc_df = apply_variation(df_copy, self.mc_parameter)
                intermediate_dfs.append(mc_df[self.data_processed.columns[0]]) 
                
            final_mc_df = pd.concat(intermediate_dfs, axis=1)
            final_mc_df.columns = [f'MC_{i+1}' for i in range(self.mc_iterations)]


            final_mc_df.reset_index(drop=True, inplace=True)

            Montecarlo_phi_q = pd.DataFrame()

            Montecarlo_phi_q_values = []
            completed_cols = 0

            for col in tqdm(final_mc_df.columns, desc='Computing Montecarlo simulation'):
                df = final_mc_df[[col]]  
                Phi_q_values_mc_list = calculate_Phi_q_values_mc(df, self.quanta_arr)
                Montecarlo_phi_q_values.append(Phi_q_values_mc_list)
                
                completed_cols += 1

            Montecarlo_phi_q = pd.DataFrame(Montecarlo_phi_q_values, columns=self.quanta_arr).T

            Montecarlo_phi_q.columns = final_mc_df.columns

            alpha_1_share = round(self.mc_iterations * 0.01) # LE ITERAZIONI MONTECARLO DEVONO ESSERE ALMENO 100?? 
            alpha_5_share = round(self.mc_iterations * 0.05)
            mc_max_column = Montecarlo_phi_q.max()
            alpha_1 = mc_max_column.nlargest(alpha_1_share).iloc[-1]
            alpha_5 = mc_max_column.nlargest(alpha_5_share).iloc[-1]


            self.alpha_1 = alpha_1
            self.alpha_5 = alpha_5
            self.Montecarlo_phi_q = Montecarlo_phi_q
        
        else:
            self.alpha_1 = []
            self.alpha_5 = []
            self.Montecarlo_phi_q = []
        
    def plot_quantogram(self, figsize=(10, 6), title = "Quantogram", 
                        plot_best_quantum = True,
                        legend_outside = True, 
                        plot_alpha_1 = True, plot_alpha_5 = True,
                        fill_between = True,
                        x_step = 1, save=False, dpi=300):
        """
        Plot a quantogram using matplotlib.

        Parameters
        ----------
            figsize (tuple)
                size of the figure (default: (10, 6)).
            title (str)
                title of the figure (default: "Quantogram").
            plot_best_quantum (bool)
                if True, the best quantum is plotted (default: True).
            legend_outside (bool)
                if True, the legend is placed outside the figure (default: True).
            plot_alpha_1 (bool)
                if True, the alpha_1 value is plotted (default: True).
            plot_alpha_5 (bool)
                if True, the alpha_5 value is plotted (default: True).
            x_step (int)
                step between x-axis ticks (default: 1).
            fill_between (bool)
                if True, the area under the quantogram is filled (default: True).
            save (bool)
                if True, the figure is saved (default: False).
            dpi (int)
                resolution of the figure (default: 300).
        
        Returns
        -------
            matplotlib figure
        
        Examples
        --------
        >>> from cqarcheo import CQAnalysis
        >>> cqa = CQAnalysis(data = 'data.csv', min_value = 5, max_value = 200,
                            min_quantum = 5, max_quantum = 24, step = 0.02,
                            Montecarlo_sim = True, mc_parameter = 0.15, mc_iterations = 100)
        
        >>> cqa.plot_quantogram(figsize=(10, 6), title = "Quantogram",
                                plot_best_quantum = True, save=False,
                                legend_outside = True,
                                plot_alpha_1 = True, plot_alpha_5 = True,
                                x_step = 1, dpi=300)
        """   
        fig, ax = plt.subplots(figsize=figsize)

        sns.lineplot(data=self.phi_q_df, x=self.phi_q_df['Quanta'], y=self.phi_q_df['Phi_q_values'], color='black', label='Quantogram', ax=ax, linewidth=2)
        
        if fill_between == True:
            ax.fill_between(self.phi_q_df['Quanta'], 0, self.phi_q_df['Phi_q_values'], color="black", alpha=0.2)
            
        
        if self.Montecarlo_sim == True:
            if plot_alpha_1 == True:
               ax.axhline(y=self.alpha_1, color='r', linestyle='--', label='alpha_1', linewidth=1)
            if plot_alpha_5 == True:
                ax.axhline(y=self.alpha_5, color='g', linestyle='--', label='alpha_5', linewidth=1)

        quantum_max = self.phi_q_df.loc[self.phi_q_df['Phi_q_values'].idxmax(), 'Quanta']
        
        if plot_best_quantum == True:
            ax.axvline(x=quantum_max, color='grey', linestyle='--', label='Best quantum', linewidth=1)

        ### zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        ax.set_xlim(self.quanta_arr[0], self.quanta_arr[-1])
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='x', labelsize=8)


        xticks = np.arange(self.quanta_arr[0], self.quanta_arr[-1], x_step)
        plt.xticks(xticks)

                
        plt.legend()
        plt.grid(axis='y', linewidth=0.5)

        plt.xlabel('Quanta', fontsize=12)
        plt.ylabel('$\phi$(q)', fontsize=12)
        plt.title(title)

        plt.text(0.12, -0.02, f"Quantum: {self.quantum_max:.2f}, $\phi$(q) value: {self.phi_q_max_value:.2f}", transform=plt.gcf().transFigure, fontsize=8, ha='left')


        if legend_outside == True:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        if save == True:
            plt.savefig('Quantogram.png', dpi=dpi, bbox_inches='tight')

        plt.show()

    def save_quanta(self, extension = 'xlsx',):
        """
        Save the quantogram data to a file.

        Parameters
        ----------
            extension (str)
                extension ('xlsx' or 'csv') of the file to be saved (default: 'xlsx').
        
        Examples
        --------
        >>> from cqarcheo import CQAnalysis
        >>> cqa = CQAnalysis(data = 'data.csv', min_value = 5, max_value = 200,
                            min_quantum = 5, max_quantum = 24, step = 0.02,
                            Montecarlo_sim = True, mc_parameter = 0.15, mc_iterations = 100)
        >>> cqa.save_quanta(extension = 'xlsx')
        """
        if extension == 'xlsx':
            self.phi_q_df.to_excel('Quanta_table.xlsx', index=False)
        elif extension == 'csv':
            self.phi_q_df.to_csv('Quanta_table.csv', index=False)
        else:
            raise Exception("Format not supported, please use csv (.csv) or excel (.xlsx) file.")


    def __repr__(self):
        return f"Quantum: {self.quantum_max:.3f}, φ(q) value: {self.phi_q_max_value:.3f}"
    


def compare_quantograms(quantogram_list, figsize=(10, 6), color_list=None, label_list=None, 
                        plot_montecarlo_bound=True, alpha_list = None, plot_alpha_1 = True, plot_alpha_5 = True,
                        fill_between = True, legend_outside=True, x_step = 1, save= True, dpi = 300):
    """
    Comprare multiple quantograms using matplotlib.

    Parameters
    ----------
        quantogram_list (list)
            list of quantograms (CQAnalysis instances) to be compared.
        figsize (tuple)
            size of the figure (default: (10, 6)).
        color_list (list)
            list of colors (str) for the quantograms. If None, default colors will be used (default: None).
        label_list (list)
            list of labels (str) for the quantograms. If None, default labels will be used (default: None).
        plot_montecarlo_bound (bool or list)
            if True, the Montecarlo bounds are plotted (default: True). If a list (including bool values) is passed, the bounds are plotted only for the quantograms in the list.
        alpha_list (list)
            list of alpha values (float) for the quantograms. If None, default values will be used (default: None).
        plot_alpha_1 (bool)
            if True, the alpha_1 value is plotted (default: True).
        plot_alpha_5 (bool)
            if True, the alpha_5 value is plotted (default: True).
        fill_between (bool)
            if True, the area between the quantogram and the x-axis is filled (default: True).
        legend_outside (bool)
            if True, the legend is placed outside the figure (default: True).
        x_step (int)
            step between x-axis ticks (default: 1).
        save (bool)
            if True, the figure is saved (default: False).
        dpi (int)
            resolution of the figure (default: 300).

    Returns
    -------
        matplotlib figure

    Examples
    --------
    >>> from cqarcheo import CQAnalysis, compare_quantograms
    >>> cqa1 = CQAnalysis(data = 'data.csv', min_value = 5, max_value = 200,
                          min_quantum = 5, max_quantum = 24, step = 0.02,
                          Montecarlo_sim = True, mc_parameter = 0.15, mc_iterations = 100)  
    >>> cqa2 = CQAnalysis(data = 'data.csv', min_value = 5, max_value = 200,
                            min_quantum = 5, max_quantum = 24, step = 0.02,
                            Montecarlo_sim = True, mc_parameter = 0.15, mc_iterations = 100)
    >>> compare_quantograms([cqa1, cqa2], figsize=(10, 6), color_list=["black", "red"], 
                            label_list=["Quantogram 1", "Quantogram 2"],
                            plot_montecarlo_bound=True, alpha_list = None,
                            legend_outside=True, x_step = 1, save= True, dpi = 300)
    
    Notes
    -----
    color_list supports all the colors supported by matplotlib (https://matplotlib.org/stable/gallery/color/named_colors.html)
    """
    
    fig, ax = plt.subplots(figsize=figsize)

    if color_list is None:
        color_palette = sns.color_palette("Set1", len(quantogram_list))
    else:
        color_palette = color_list

    for i, quantogram in enumerate(quantogram_list):
        
        line_color = color_palette[i]

        sns.lineplot(data=quantogram.phi_q_df, x='Quanta', y='Phi_q_values', ax=ax, linewidth=2, 
                     color=line_color, 
                     label=f'Quantogram {i}' if label_list is None else label_list[i],
                     alpha = 1 if alpha_list is None else alpha_list[i])
        
        if fill_between is True:     
            ax.fill_between(quantogram.phi_q_df['Quanta'], 0, quantogram.phi_q_df['Phi_q_values'], color=line_color, alpha=0.2)

       
        if plot_montecarlo_bound is True:
            if plot_alpha_1:
                ax.axhline(y=quantogram.alpha_1, color=line_color, linestyle=':', label='alpha_1', linewidth=1)
            if plot_alpha_5:
                ax.axhline(y=quantogram.alpha_5, color=line_color, linestyle='--', label='alpha_5', linewidth=1)


        elif isinstance(plot_montecarlo_bound, list):
            if plot_montecarlo_bound[i] == True:
                if plot_alpha_1: 
                    ax.axhline(y=quantogram.alpha_1, color=line_color, linestyle=':', label='alpha_1', linewidth=1)
                if plot_alpha_5:    
                    ax.axhline(y=quantogram.alpha_5, color=line_color, linestyle='--', label='alpha_5', linewidth=1)



    ax.set_xlim(quantogram_list[0].quanta_arr[0], quantogram_list[0].quanta_arr[-1])
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelsize=8)


    xticks = np.arange(quantogram_list[0].quanta_arr[0], quantogram_list[0].quanta_arr[-1], x_step)
    plt.xticks(xticks)


    plt.legend()
    plt.grid(axis='y', linewidth=0.5)

    plt.xlabel('Quanta', fontsize=12)
    plt.ylabel('$\phi$(q)', fontsize=12)
    plt.title("Quantograms")
    
    if legend_outside == True:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    ### zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    
    if save == True:
            plt.savefig('Quantogram_compare.png', dpi=dpi, bbox_inches='tight')
    
    
    plt.show()