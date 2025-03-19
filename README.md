# Wismut_ME_Bayes

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15050372.svg)](https://doi.org/10.5281/zenodo.15050372)
---

Code was developed and executed under Debian GNU/Linux 12 (bookworm).

*Code will not run for the application. Data is not openly available.*
Results of all Markov chains can be downloaded at [https://zenodo.org/records/15050372](https://zenodo.org/records/15050372).

Plots and results of the tables are generated from

- `analyze_application.ipynb`
- `analyze_simulation_study.ipynb`

Results can be reproduced from the raw chain samples.
It is necessary to adapt the paths to the outputs beforehand.
Alternatively save the results in a new directory in the root path with the name `results`.

## Structure

#### Directories

- `application/`
    Contains the code, that was used to run get the results on the data of the Wismut cohort. `run_application.sh` is a shell script that runs alls jupyter notebooks.

- `simulation_study/`
    Contains the code, that was used run the simulation_study. The file `simulate_data.ipynb` uses an *R kernel*. Not python. It can be used to generate all data sets of the simulation study. Afterwards, it is possible to run the full simulation study. See also the shell script `run_simulation_study.sh`.

- `wismut/`
    Contains the MCMC algorithm and utility modules containing functions for pre- and post-processing.

- `data/`
    Contains the used data. As data cannot be shared, it just holds the pseudo data used to generate simulated data, that would be stored in the same directory.

#### Other files

- `requirements.txt`
    Containing all packages with versions to build a virtual environment. Tested on python 3.9 and 3.11.

#### File tree

```
.
├── application
│   ├── analyze_application.ipynb
│   ├── fit_real_data_8_chains.ipynb
│   └── run_application.sh
├── data
│   └── pseudo_data.txt
├── LICENSE
├── README.md
├── requirements.txt
├── simulation_study
│   ├── analyze_simulation_study.ipynb
│   ├── other
│   │   ├── find_parameters_misspec.ipynb
│   │   └── simulate_data.ipynb
│   ├── run_model_b3-m5.ipynb
│   ├── run_model_b3-m5-missspec1.ipynb
│   ├── run_model_b3-m5-missspec2.ipynb
│   ├── run_model_b3-m5-naive.ipynb
│   ├── run_model_b3-m5-true.ipynb
│   ├── run_model_b6-m5.ipynb
│   ├── run_model_b6-m5-naive.ipynb
│   ├── run_model_b6-m5-true.ipynb
│   └── run_simulation_study.sh
└── wismut
    ├── analyze_chains.py
    ├── basics.py
    ├── ErrorComponent.py
    ├── __init__.py
    ├── LatentVariables.py
    ├── MCMC.py
    ├── Parameter.py
    ├── UncertainFactor.py
    ├── UnsharedError.py
    └── update.py
```

```
