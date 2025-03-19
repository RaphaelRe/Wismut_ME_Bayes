# Wismut_ME_Bayes

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15050372.svg)](https://doi.org/10.5281/zenodo.15050372)
---

*Code will not run. Data are not openly available.*
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
    Contains the code, that was used run the simulation_study. The file `simulate_data.ipynb` uses an *R kernel*. Not python.

- `wismut/`
    Contains the MCMC algorithm and utility modules containing functions for pre- and post-processing.

#### Other files

- `requirements.txt`
    Containing all packages with versions to build a virtual environment. Tested on python 3.9 and 3.11.

#### File tree

```
 .
├──  application
│   ├──  analyze_application.ipynb
│   ├──  fit_real_data_8_chains.ipynb
│   └──  run_application.sh
├──  LICENSE
├──  README.md
├── 󰌠 requirements.txt
├──  simulation_study
│   ├──  analyze_simulation_study.ipynb
│   ├──  other
│   │   ├──  find_parameters_misspec.ipynb
│   │   └──  simulate_data.ipynb
│   ├──  plots
│   ├──  run_model_b3-m5-missspec1.ipynb
│   ├──  run_model_b3-m5-missspec2.ipynb
│   ├──  run_model_b3-m5-naive.ipynb
│   ├──  run_model_b3-m5-true.ipynb
│   ├──  run_model_b3-m5.ipynb
│   ├──  run_model_b6-m5-naive.ipynb
│   ├──  run_model_b6-m5-true.ipynb
│   ├──  run_model_b6-m5.ipynb
│   └──  run_simulation_study.sh
└──  wismut
    ├──  __init__.py
    ├──  analyze_chains.py
    ├──  basics.py
    ├──  ErrorComponent.py
    ├──  LatentVariables.py
    ├──  MCMC.py
    ├──  Parameter.py
    ├──  UncertainFactor.py
    ├──  UnsharedError.py
    └──  update.py

```

```
