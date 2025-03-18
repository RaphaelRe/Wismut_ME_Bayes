import matplotlib.pyplot as plt
import pandas as pd
import arviz as az
import os
import numpy as np
import xarray as xr
import toolz
from typing import Union


def dict_map(func, tree):
    """
    Function that mimicas the treemap functionality from jax for dicts
    """
    if isinstance(tree, dict):
        return {k: dict_map(func, v) for k, v in tree.items()}
    else:
        return func(tree)


def load_traces_parameter(
    parameter: str, path: str = os.getcwd(), simulation: bool = True, verbose=False
) -> dict[int, np.ndarray]:
    if simulation:  # nested folder structure for simulated data (e.g. 100 datasets)
        subdirs = os.listdir(path)
        subdirs = list(map(int, subdirs))
        subdirs.sort()
        subdirs = list(map(str, subdirs))
        paths = [f"{path}/{d}" for d in subdirs]
    else:
        paths = [path]

    file_list = list(
        map(lambda full_path: search_files_by_name(parameter, full_path), paths)
    )
    res = {}
    for i, files in enumerate(file_list):
        # inefficient as this always makes the same conversion. However, not so important to worry about speed here
        x = [np.loadtxt(file) for file in files]
        try:
            x = np.stack(x)
            x = az.convert_to_inference_data({parameter: x})
        except ValueError:
            if verbose:
                print(
                    f"Value error in np.stack. Probably a problem with the shape. Shapes are: {list(map(lambda x: x.shape, x))}"
                )
                x = None
        res[i + 1] = x
    if not simulation:
        res = res[1]
    return res


def search_files_by_name(name: str, file_path: str):
    matching_files = []

    # Iterate through all files in the specified path
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if name in file:
                matching_files.append(os.path.join(root, file))

    matching_files.sort()
    return matching_files


def plot_chains_sim(
    samples: dict, num: tuple[int] = (1, 101), trace: bool = True, **kwargs
) -> None:
    """
    Plots the chains of the samples where samples is a dict with K results from K simulation runs.
    """
    if len(num) != 2:
        raise ValueError("num has to be a tuple of 2 integers")
    fig = plt.figure(constrained_layout=True, **kwargs)
    subfigs = fig.subfigures(num[1], 1)
    for i in range(*num):
        subfigs[i] = az.plot_trace(samples[i + 1])
    return None


def calc_rhat(
    samples: dict, num: tuple[int] = (1, 101), simplify: bool = False
) -> Union[dict[int, float], list]:
    if len(num) != 2:
        raise ValueError("num has to be a tuple of 2 integers")
    rhats = toolz.valmap(
        lambda x: calc_quanitty(az.rhat, x), {k: samples[k] for k in range(*num)}
    )
    if simplify:
        rhats = [rhats[i] for i in rhats.keys()]
    return rhats


def calc_quanitty(fun, x, **kwargs):
    try:
        fx = fun(x, **kwargs)
    except:
        fx = None
    return fx


def flat_array(x):
    try:
        return x.posterior.to_dataarray().values.flatten()
    except:
        return np.nan


def reduce_results(
    samples: dict, num: tuple[int] = (1, 101), p: float = 0.95
) -> Union[dict[str, float], np.ndarray]:
    """
    Only used for simulated data with an assumed structure
    """
    # data = {k: samples[k].posterior.to_dataarray().values.flatten() for k in range(*num)}
    data = toolz.valmap(flat_array, samples)

    mean = toolz.valmap(lambda x: calc_quanitty(np.mean, x), data)
    median = toolz.valmap(lambda x: calc_quanitty(np.median, x), data)
    hdi = toolz.valmap(lambda x: calc_quanitty(az.hdi, x, hdi_prob=p), data)

    #     median = toolz.valmap(np.mean, data)
    #     hdi = toolz.valmap(lambda x: az.hdi(x, hdi_prob=p), data)

    d = {}
    for i in data.keys():
        try:
            d[i] = {"mean": mean[i], "median": median[i], "hdi": hdi[i]}
        except:
            pass
    return d


def summarize_results(
    samples: dict, num: tuple[int] = (1, 101), p: float = 0.95
) -> pd.DataFrame:
    """
    Only used for simulated data with an assumed structure
    """
    data = {k: samples[k] for k in range(*num)}
    res = toolz.valmap(lambda d: az.summary(d, round_to=10, hdi_prob=p), data)
    return pd.concat(res)


def check_inside(x: float, q: np.ndarray) -> bool:
    try:
        ins = x > q[0] and x < q[1]
    except:
        ins = False
    return ins

