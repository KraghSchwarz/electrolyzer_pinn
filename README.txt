
####################################################################################
####										####
####   				electrolyzer_PINN				####
####										####
####										####
####   Developed by Marcus V. Kragh-Schwarz, Florian Kuschel, 			####
####   Lukas Feierabend, Paolo Lamagni, and Natalia Levin			####
####   										####
####   In collaboration between ZBT, HydrogenPro and Aarhus University		####
####										####
####################################################################################



Structure:

- electrolyzer_PINN/
|-- .github/
|-- .venv/ (library root)
|-- configs/
    |-- configs.yaml
|-- datasets/
|-- docs/
|-- examples/
|   |-- example1_notebook.txt
|-- src/
|   |-- elec_pinn/
|       |-- data/
|           |-- loader.py
|           |-- preprocessing.py
|       |-- models/
|           |-- ann.py
|           |-- base.py
|           |-- fpinn.py
|           |-- full_pinn.py
|           |-- gpinn.py
|       |-- utils/
|           |-- __init__.py
|           |-- logging.py
|           |-- visualization.py
|       |-- __init__.py
|       |-- cli.py
|-- tests/
|-- .gitignore.txt
|-- CHANGELOG.txt
|-- README.txt



Explanations:




Datasets:
Four datasets are included to play around with. Three of the four data sets are based on the synthethic NREL electrolyzer module. The last is based on experimental data.

The synthetic data sets contain time, current density and cell voltage.
The experimental data set contains time, current density, temperature, cell voltage, and gas purities. 

The datasets are:
 - Square wave AST data (synthetic)
 - Ramp-and-hold (synthetic)
 - SolarPV power profile (synthetic)
 - Experimental PEMWE electrolyzer data



Examples



