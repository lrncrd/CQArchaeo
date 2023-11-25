<div style="text-align: center;">
  <img src="imgs/logo.jpg" alt="logo" width=500/>
</div>

# CQArcheo
A simple Python package for calculating and plotting Kendall CQA results (Kendall 1974).

## Installation

```bash
pip install cqarchaeo
```
Now you can import the package in your Python scripts.

## Usage

Perform a CQA analysis on a given dataset:

```python
### Import the package
from cqarchaeo import CQAnalysis

### Perform the analysis
cqa = CQAnalysis(r'data.xlsx', min_value = 5, max_value = 200, min_quantum = 5, 
        max_quantum = 24, step = 0.02, Montecarlo_sim = True, 
        mc_parameter = 0.15, mc_iterations = 100)

### View the quantogram
cqa.plot_quantogram(figsize=(10, 6), plot_best_quantum=True,    
        dpi=300, plot_alpha_5=True)
```
You get the following quantogram:

![](imgs/Quantogram.png)

You can also compare multiple quantograms using the `compare_quantograms` function:

```python
### Import the package
from cqarchaeo import CQAnalysis, compare_quantograms

### Perform the analysis
cqa1 = CQAnalysis(r'data1.xlsx', min_value = 5, max_value = 200, min_quantum = 5, 
        max_quantum = 24, step = 0.02, Montecarlo_sim = True, 
        mc_parameter = 0.15, mc_iterations = 100)
cqa2 = CQAnalysis(r'data2.xlsx', min_value = 5, max_value = 200, min_quantum = 5, 
        max_quantum = 24, step = 0.02, Montecarlo_sim = True, 
        mc_parameter = 0.15, mc_iterations = 100)

### Compare the quantograms
compare_quantograms(quantogram_list = [cqa1, cqa_2], figsize=(10, 6), 
                    color_list=["black", "green"], alpha_list=[0.2, 1],
                    label_list=None, plot_montecarlo_bound=[True, True])
```

You get the following plot:

![](imgs/Quantogram_compare.png)




### References
Kendall, D.G., 1974. Hunting quanta. Phil. Trans. R. Soc. Lond. A 276, 231–266. https://doi.org/10.1098/rsta.1974.0020

