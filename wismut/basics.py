"""
This module implements some basic functions used by other modules
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix
from numba import jit


def read_data(path, sort=True, assign_groups=False) -> pd.DataFrame:
    """
    Reads and preprocesses and returns the data provided by path.

    :param path: The path to read from

    :return: The processed data
    """
    data = pd.read_csv(path)
    data = data.rename(columns={"true.exposure": "X",
                                "true.cum.exposure": 'Xcum',
                                "obs.exposure": "Z",
                                'obs.cum.exposure': 'Zcum',
                                "delta": "event",
                                "start": "truncation",
                                "stop": "t",
                                "ID": "Ident"
                                 })

    data = data.drop(columns="Unnamed: 0")
    data["I_trunc"] = data["I"]

    # divide X by 100 to get 100 wlm
    data[["X", "Xcum", "Z", "Zcum"]] = data[["X", "Xcum", "Z", "Zcum"]] / 100
    # data[["X", "Xcum", "Z", "Zcum"]] = data[["X", "Xcum", "Z", "Zcum"]]


    # set info for miners with 0-exposure to NA
    # data.loc[data.Z == 0, ['year', 'object', 'activity', 'C_Rn_obs', 'w_period',
                           # 'w_classical', 'g_period', 'g_classical', 'f_classical']] = np.nan


    if "prop" not in data.columns:
        data['prop'] = 1.  # prop is only in real data relevant. for simulated, we can set this to 1

    if "model" not in data.columns:
        data["model"] = "M2"
        data.loc[data.Z == 0, "model"] = "M0"

    data.loc[data.Z == 0, ['model']] = 'M0'


    if sort:
        data = sort_data(data)

    if assign_groups:
        print("Assinging groups...")
        idents_group = np.hstack([np.repeat(i, 10) for i in range(80)])

        # sloppy lösung, da historisch gewachsen. Generische Lösung mit zb modulo 10 wäre sinnvoller
        if data[data.X > 0].Ident.unique().shape[0] > 80*10:  # wenn es mit 80*10 nicht aufgeht muss die letzte gruppe ungleich 10 sein -> 81. Gruppe (start bei 0)
            idents_group = np.concatenate((idents_group, np.repeat(80, data[data.X > 0].Ident.unique().shape[0]-80*10)))
        if data[data.X > 0].Ident.unique().shape[0] < 80*10:  # wenn es mit 80*10 nicht aufgeht muss die letzte gruppe ungleich 10 sein -> 81. Gruppe (start bei 0)
            idents_group = idents_group[:data[data.X > 0].Ident.unique().shape[0]]

        group_assign = pd.DataFrame({'Ident': data[data.X > 0].Ident.unique(), 'group': idents_group})  # ordnet jedem worker seine gruppe zu
        data = data.merge(group_assign, 'outer')
        data.group.fillna(-1, inplace=True)  # alle die keine exposure haben, bekommen gruppe -1
        print("Done! Created 80+1 groups.")

    # return sort_data(data) if sort else data
    return data


def sort_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts the data after the ident and truncation

    :param data: An unsorted DataFrame
    """
    return data.sort_values(['Ident', 'truncation']).reset_index(drop=True)




def create_sparse_matrix(frame, lag1=1, lag2=1000):
    """
    This function creates the sparse matrix necessary to cumulate
    exposures for each miner
    in the data frame frame. This matrix has to be multiplied with
    the uncumulated exposure
    to obtain cumulated exposure. (If you want to use lags, do it BEFORE using this function. Lags are not working atm.)

    :param frame: A data pandas DataFrame
    :param lag1: currently not in use
    :param lag2: currently not in use

    :return: A sparse cumulation matrix
    """
    dd = frame['Ident'].value_counts(sort=False).to_frame()

    def get_ordered_seq(seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    ordered_seq = np.array(get_ordered_seq(frame.Ident))
    dd = dd.reindex(ordered_seq)
    # counts = dd.Ident.to_list() # change in version with pandas, this si the old version
    counts = dd.iloc[:,0].to_list() # change in version with pandas, this si the old version

    blocks = [create_sparse_block(count, lag1=lag1, lag2=lag2) for count in counts]
    A = sparse.block_diag(blocks)

    return(A)



def create_sparse_block(nb, lag1, lag2):
    """
    This function returns a sparse matrix with dimension nb
    and the lower diagonal
    filled with trues.
    """
    if (lag1 > nb):
        block = sparse.bsr_matrix((nb, nb), dtype=bool)

    else:
        if (lag2 > nb):
            lag2 = nb

        block = sparse.diags([True] * (lag2 - lag1 + 1), np.r_[-(lag2 - 1):2 - lag1],
                             shape=(nb, nb), dtype=bool)

    return(block)





def create_group_cumulation_matrices(frame, lag1=1, lag2=1000):
    '''
        The function create_group_matrices creates a dictionary with the
        different cumulation matrices for each group.
        '''
    groups = np.unique(frame['group'])
    group_cumulation_matrices = {group : create_sparse_matrix(
                                         frame[frame['group']== group],
                                         lag1 = lag1, lag2 = lag2) for group in groups}
    return(group_cumulation_matrices)







def create_mapping_matrix(mapping_identifier, data, exposed_miners) -> csr_matrix:
    """
    Creates a matrix to map the errors to the right dimension

    :param mapping_identifier: A list with strings defining the mapping identifier
    :param data: The pandas DataFrame the mapping matrix is creadted on
    :param exposed_miners: groups are determined by the exposd miners

    :return: A scipy sparse matrix
    """
    # dict with positions of each group, should handle NAs well i.e. ignoring them as group
    # exposed_miners = data.year.dropna() # not for all cases necessary but good if you have to handle year
    group_positions = data[exposed_miners].groupby(mapping_identifier).groups
    mapping_matrix = np.zeros(shape=(data.shape[0], len(group_positions)), dtype=bool)

    for j, group in enumerate(group_positions):
        mapping_matrix[group_positions[group], j] = True

    mapping_matrix_sparse = csr_matrix(mapping_matrix)

    return mapping_matrix_sparse




def create_mapping_matrix_t_o(mapping_identifier_classical, mapping_identifier_Berkson, data, exposed_miners) -> csr_matrix:
    """
    Creates a matrix to map the errors to the right dimension (used from classical to full dataset)

    :param mapping_identifier: A list with strings defining the mapping identifier
    :param data: The pandas DataFrame the mapping matrix is creadted on
    :param exposed_miners: groups are determined by the exposd miners

    :return: A scipy sparse matrix
    """
    # dict with positions of each group, should handle NAs well i.e. ignoring them as group


    if len(mapping_identifier_classical) == 1: # only the case for w_period
        group_positions_intermediate = data[exposed_miners].groupby(mapping_identifier_Berkson).agg(
                identifier1=pd.NamedAgg(column=mapping_identifier_classical[0], aggfunc='min'),
                identifier2=pd.NamedAgg(column=mapping_identifier_classical[0], aggfunc='max')
                ).reset_index()

        test = np.all(group_positions_intermediate.identifier1 == group_positions_intermediate.identifier2)
        if not test:
            raise ValueError("w_period != w_period2")
        group_positions = group_positions_intermediate.groupby("identifier1").groups

    
    elif len(mapping_identifier_classical) == 2:
        # group_positions_intermediate = data.iloc[exposed_miners.index].groupby(mapping_identifier_Berkson).agg(
        group_positions_intermediate = data[exposed_miners].groupby(mapping_identifier_Berkson).agg(
                identifier11 = pd.NamedAgg(column = mapping_identifier_classical[0], aggfunc='min'),
                identifier12 = pd.NamedAgg(column = mapping_identifier_classical[0], aggfunc='max'),
                identifier21 = pd.NamedAgg(column = mapping_identifier_classical[1], aggfunc='min'),
                identifier22 = pd.NamedAgg(column = mapping_identifier_classical[1], aggfunc='max')
                ).reset_index()

        test1 = np.all(group_positions_intermediate.identifier11== group_positions_intermediate.identifier12)
        test2 = np.all(group_positions_intermediate.identifier21== group_positions_intermediate.identifier22)
        if not test1 and test2:
            raise ValueError("something is wrong with mapping identifiers in create_mapping_matrix_t_o")
        group_positions = group_positions_intermediate.groupby(["identifier11", "identifier21"]).groups


    mapping_matrix = np.zeros(shape=(group_positions_intermediate.shape[0],len(group_positions)),dtype=bool) 
    for j, group in enumerate(group_positions):
        mapping_matrix[group_positions[group], j] = True
      
    mapping_matrix_sparse = csr_matrix(mapping_matrix)
        
    return mapping_matrix_sparse




def return_year(df: pd.DataFrame) -> int:
    """
    Helper function to detect the first year.
    This function is a wrapper since it provides some beter behavior for specific cases
    """
    years = df.year.unique()

    if years.size == 1:
        return years
    elif 1 in df.transfer_reference.values:
        years = df.year[df.transfer_reference == 1].unique()
        if years.size > 1:
            raise ValueError("There should be one year only as reference!")
        return years
    else:
        print("Circular reference detected! Fot the moment the minimum is returned")
        return years.min()




def calculate_baseline_hazard(lambda_t: np.ndarray, inter: np.ndarray):
    """
    Calculates the baseline hazard for given lambda and cutoffs
    """
    baseline_hazard = np.append(np.array(0), np.array(np.cumsum(lambda_t * inter)))
    return (baseline_hazard)




@jit(nopython=True)
def ratio_exposure_lognormal_jit(Xt, Xcand, mu_x, sigma_x):
    log_Xt = np.log(Xt)
    log_Xcand  = np.log(Xcand)
    first_term = np.sum((log_Xt**2 - log_Xcand**2))/(2*sigma_x**2)
    second_term = np.sum((mu_x/sigma_x**2 - 1)*(log_Xcand- log_Xt))
    return(first_term +second_term)
