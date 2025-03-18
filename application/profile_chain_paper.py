import os
import numpy as np
import basics
from MCMC import MCMC
import scipy.stats as stats

# set path
# path = "/home/rrehms/Wismut/Code/"
path = os.getcwd() + "/"

data = basics.read_data(path + "data/S3_M1M2M2_ExpertM3/Data_1.csv")
# data = basics.read_data(path + "data/S3_M1M2M2_ExpertM3M4/Data_1.csv")

data['tau'] = 1

data.object.unique()

data.Ident

def generate_proposal_sds():
    proposal_sd = {
            'beta': 0.00011,  # for cox
            'lambda1': 0.000211,
            'lambda2': 0.000611,
            'lambda3': 0.000611,
            'lambda4': 0.000211,
            'C_Rn_old_mu': 0.2,
            'C_Rn_old_sigma': 0.2,
            'C_Rn_ref_mu': 0.2,
            'C_Rn_ref_sigma': 0.2,
            'C_Rn_mu': 0.2,
            'C_Rn_sigma': 0.2,
            'C_RPD_mu': 0.2,
            'C_RPD_sigma': 0.2,
            'C_Exp_mu': 0.4,
            'C_Exp_sigma': 0.5,
            'gamma_alpha': 0.5,
            'gamma_beta': 0.5,
            'phi_alpha': 0.5,
            'phi_beta': 0.5,
            'omega_alpha': 0.5,
            'omega_beta': 0.5,
            }
    return proposal_sd




def generate_prior_parameters():
    prior_parameters = {'beta': {'dist': "normal", 'mean': 0, 'sd': 100},
                        'lambda1': {'dist': "gamma",'shape': 600,
                                    'scale': 1 / 10000000,
                                    'min': 0, 'max': 200
                                    },
                        'lambda2': {'dist': "gamma", 'shape': 12000,
                                    'scale': 1 / 1000000,
                                    'min': 0, 'max': 200
                                    },
                        'lambda3': {'dist': "gamma", 'shape': 46000,
                                    'scale': 1 / 1000000,
                                    'min': 0, 'max': 200
                                    },
                        'lambda4': {'dist': "gamma", 'shape': 1000,
                                    'scale': 1 / 100000,
                                    'min': 0, 'max': 200
                                    },
                        'C_Rn_mu': {'dist': "normal", 'mean': 6, 'sd': 5},
                        'C_Rn_sigma': {'dist': "normal", 'mean': 8, 'sd': 0.5},
                        'C_Exp_mu': {'dist': "normal", 'mean': 1.78, 'sd': 3},
                        'C_Exp_sigma': {'dist': "normal", 'mean': 0.79, 'sd': 2},
                        'C_RPD_mu': {'dist': "normal", 'mean': 0.15, 'sd': 0.03},
                        'C_RPD_sigma': {'dist': "normal", 'mean': 0.2, 'sd': 0.03},
                        'gamma_alpha': {'dist': "normal", 'mean': 3, 'sd': 2},
                        'gamma_beta': {'dist': "normal", 'mean': 3, 'sd': 2},
                        'phi_alpha': {'dist': "normal", 'mean': 3, 'sd': 2},
                        'phi_beta': {'dist': "normal", 'mean': 3, 'sd': 2},
                        'omega_alpha': {'dist': "normal", 'mean': 3, 'sd': 2},
                        'omega_beta': {'dist': "normal", 'mean': 3, 'sd': 2},
                        # 'gamma_alpha': {'dist': "gamma", 'mean': 3, 'sd': 2},
                        # 'gamma_beta': {'dist': "gamma", 'mean': 3, 'sd': 2},
                        # 'phi_alpha': {'dist': "gamma", 'mean': 3, 'sd': 2},
                        # 'phi_beta': {'dist': "gamma", 'mean': 3, 'sd': 2},
                        # 'omega_alpha': {'dist': "gamma", 'mean': 3, 'sd': 2},
                        # 'omega_beta': {'dist': "gamma", 'mean': 3, 'sd': 2},
                        }
    return prior_parameters




def generate_start_values(seed=123, disease_model="cox_like"):
    np.random.seed(seed)
    rnd = lambda: stats.uniform(loc=0.9, scale=0.2).rvs(1)[0]

    # beta_true = 0.003 * 100 * 0.8
    beta_true = 0.003
    l1 = 0.00006 * 1.1
    l2 = 0.00120 * 1.1
    l3 = 0.00460 * 1.1
    l4 = 0.01000 * 1.1

    start_values = {'chain1': {'beta': beta_true * rnd() + 0.0005,
                               'lambda1': l1 * rnd(),
                               'lambda2': l2 * rnd(),
                               'lambda3': l3 * rnd(),
                               'lambda4': l4 * rnd(),
                               # values for truncnorm
                               'prior_parameters': {
                                                    'M2': {'C_Rn': {'mu': 6 * rnd(),
                                                                    'sigma': 8 * rnd()
                                                                    },
                                                           },
                                                    'M2_Expert': {'C_Exp': {'mu': 1.78 * rnd(),
                                                                            'sigma': 0.79 * rnd()
                                                                            },
                                                                  },
                                                    'M3': {'C_RPD': {'mu': 0.15 * rnd(),
                                                                    'sigma': 0.2 * rnd()
                                                                     },
                                                           },
                                                    'equilibrium': {'gamma': {'alpha': 3 * rnd(),
                                                                              'beta': 3 * rnd()
                                                                              }
                                                                    },
                                                    'activity': {'phi': {'alpha': 3 * rnd(),
                                                                         'beta': 3 * rnd()
                                                                         }
                                                                 },
                                                    'working_time': {'omega': {'alpha': 3 * rnd(),
                                                                               'beta': 3 * rnd()
                                                                                }
                                                                     },
                                                    }
                               }
                    }
    if disease_model == 'ERR':
        start_values['chain1']['beta'] *= 10
    return start_values







#########
# M1a M2 M2_Expert M3
#########
uncertainty_characteristics = {
        'M1a': {'C_Rn_old': {'classical_error': {'sd': 6.56, 'structure': 'additive', 'proposal_sd': 1.5},
                             # 'classical_error': {'sd': 0, 'structure': 'additive', 'proposal_sd': 0.1},
                             'Berkson_error': {'sd': 0},
                             'exposure_model_distribution': 'norm',
                             'exposure_model_parameters': {'mu': 22.5, 'sigma': 4},  # 4 is arbitrary
                             'exposure_model_truncation': {'lower': 1e-10},
                             'mapping_identifier_classical': ['cluster_C_Rn_old'],  # empty on purpose
                             'name_obs_values': 'C_Rn_old'
                             # 'name_obs_values': 'C_Rn_old_true'
                             },
                'C_Rn_ref': {'classical_error': {'sd': 5.29, 'structure': 'additive', 'proposal_sd': 1.0},
                             # 'classical_error': {'sd': 0, 'structure': 'additive', 'proposal_sd': 0.1},
                             'Berkson_error': {'sd': 0},
                             'exposure_model_distribution': 'norm',
                             'exposure_model_parameters': {'mu': 27.3, 'sigma': 4},  # 4 is arbitrary

                             'exposure_model_truncation': {'lower': 1e-10},
                             'mapping_identifier_classical': ['cluster_C_Rn_obs_ref'],  # empty on purpose
                             'name_obs_values': 'C_Rn_obs_ref'
                             # 'name_obs_values': 'C_Rn_ref_true'
                             },
                'b': {'classical_error': {'sd': 0.33, 'structure': 'multiplicative', 'proposal_sd': 0.1},
                      # 'classical_error': {'sd': 0, 'structure': 'multiplicative', 'proposal_sd': 0.1},
                      'Berkson_error': {'sd': 0.69, 'structure': 'multiplicative', 'proposal_sd': 0.1},
                      # 'Berkson_error': {'sd': 0, 'structure': 'multiplicative', 'proposal_sd': 0.1},
                      'exposure_model_distribution': 'beta',
                      'exposure_model_parameters': {'alpha': 3, 'beta': 3},
                      'exposure_model_truncation': {'lower': 0.17, 'upper': 1},
                      'mapping_identifier_classical': ['b_period'],
                      'mapping_identifier_Berkson': ['year', 'object'],
                      'name_obs_values': 'b'
                      # 'name_obs_values': 'b_Berkson'
                      },
                'tau_e': {'classical_error': {'sd': 0.37, 'structure': 'multiplicative', 'proposal_sd': 0.9},
                          # 'classical_error': {'sd': 0, 'structure': 'multiplicative', 'proposal_sd': 0.1},
                          'Berkson_error': {'sd': 0.33, 'structure': 'multiplicative', 'proposal_sd': 0.9},
                          # 'Berkson_error': {'sd': 0, 'structure': 'multiplicative', 'proposal_sd': 0.1},
                          'exposure_model_distribution': 'beta',
                          'exposure_model_parameters': {'alpha': 3, 'beta': 3},
                          'exposure_model_truncation': {'lower': 0.46, 'upper': 1},
                          'mapping_identifier_classical': ['tau_e_period'],  # stimmat das?
                          'mapping_identifier_Berkson': ['year','object'],  # stimmt das?
                          'name_obs_values': 'tau_e'
                          # 'name_obs_values': 'tau_e_Berkson'
                          },
                'A': {'classical_error': {'sd': 0},
                      'Berkson_error': {'sd': 0},
                      'name_obs_values': 'A_calculated'  # hieß vor langer Zeit einmal A_t_o
                      },
                'A_ref': {'classical_error': {'sd': 0},
                          'Berkson_error': {'sd': 0},
                          'name_obs_values': 'A_ref'
                          },
                'r': {'classical_error': {'sd': 0},
                      'Berkson_error': {'sd': 0},
                      'name_obs_values': 'r'
                      },
                },
        'M2': {'C_Rn': {'classical_error': {'sd': 0.59, 'structure': 'additive', 'proposal_sd': 0.1},
                        # 'classical_error': {'sd': 0, 'structure': 'additive', 'proposal_sd': 0.1},
                        'Berkson_error': {'sd': 0},
                        'exposure_model_distribution': 'norm',
                        'exposure_model_parameters': {'mu': 6, 'sigma': 8},
                        'exposure_model_truncation': {'lower': 1e-10},
                        'mapping_identifier_classical': ['year', 'object'],
                        'name_obs_values': 'C_Rn_obs'
                        # 'name_obs_values': 'C_Rn_true'
                        # ???? 'vectorized_exposure': True
                        },
               },
        'M2_Expert': {'C_Exp': {'classical_error': {'sd': 0.936, 'structure': 'multiplicative', 'proposal_sd': 0.1},
                                # 'classical_error': {'sd': 0, 'structure': 'additive', 'proposal_sd': 0.1},
                                'Berkson_error': {'sd': 0},
                                'exposure_model_distribution': 'lognorm',
                                'exposure_model_parameters': {'mu': 1.78, 'sigma': 0.79},
                                'exposure_model_truncation': {'lower': 1e-10},
                                'mapping_identifier_classical': ['year', 'object'],
                                'name_obs_values': 'C_Rn_obs',
                                },
                   },
        'M3': {'C_RPD': {'classical_error': {'sd': 0.03, 'structure': 'additive', 'proposal_sd': 0.001},
                         # 'classical_error': {'sd': 0, 'structure': 'additive', 'proposal_sd': 0.1},
                         # 'Berkson_error': {'sd': 0.2, 'structure': 'multiplicative', 'proposal_sd': 0.05},
                         'Berkson_error': {'sd': 0},
                         'exposure_model_distribution': 'norm',
                         'exposure_model_parameters': {'mu': 0.15, 'sigma': 0.2},
                         'exposure_model_truncation': {'lower': 1e-10},
                         'mapping_identifier_classical': ['year', 'object'],
                         'name_obs_values': 'C_Rn_obs'
                         # 'name_obs_values': 'C_Rn_true'
                        # ????? 'vectorized_exposure': True
                         },
               'zeta': {'classical_error': {'sd': 0.33, 'structure': 'multiplicative', 'proposal_sd': 0.1},
                        # 'classical_error': {'sd': 0, 'structure': 'multiplicative', 'proposal_sd': 0.1},
                        'Berkson_error': {'sd': 1.45, 'structure': 'multiplicative', 'proposal_sd': 0.1},
                        # 'Berkson_error': {'sd': 0, 'structure': 'multiplicative', 'proposal_sd': 0.1},
                        'exposure_model_distribution': 'beta',
                        'exposure_model_parameters': {'alpha': 3, 'beta': 3},
                        'exposure_model_truncation': {'lower': 1.2, 'upper': 1.5},
                        'mapping_identifier_classical': ['object'],
                        # 'mapping_identifier_Berkson': ['object'],  # alte version
                        'mapping_identifier_Berkson': ['year','object'],  # laut code nicole
                        'name_obs_values': 'c_classical'
                        # 'name_obs_values': 'c_Berkson'
                        },
               },
        'activity': {'phi': {'classical_error': {'sd': 0.33, 'structure': 'multiplicative', 'proposal_sd': 0.01},
                             # 'classical_error': {'sd': 0, 'structure': 'multiplicative', 'proposal_sd': 0.1},
                             'Berkson_error': {'sd': 0.69, 'structure': 'multiplicative', 'proposal_sd': 0.01},
                             # 'Berkson_error': {'sd': 0, 'structure': 'multiplicative', 'proposal_sd': 0.1},
                             'exposure_model_distribution': 'beta',
                             'exposure_model_parameters': {'alpha': 3, 'beta': 3},
                             'exposure_model_truncation': {'lower': 1e-10, 'upper': 1.0},
                             'mapping_identifier_classical': ['object', 'activity'],
                             'mapping_identifier_Berkson': ['year', 'object', 'activity'],
                             'name_obs_values': 'f_classical'
                             # 'name_obs_values': 'f_Berkson'
                             },

                     },
        'working_time': {'omega': {'classical_error': {'sd': 0.04, 'structure': 'multiplicative', 'proposal_sd': 0.01},
                                   # 'classical_error': {'sd': 0, 'structure': 'multiplicative', 'proposal_sd': 0.1},
                                   'Berkson_error': {'sd': 0.12, 'structure': 'multiplicative', 'proposal_sd': 0.03},
                                   # 'Berkson_error': {'sd': 0, 'structure': 'multiplicative', 'proposal_sd': 0.1},
                                   'exposure_model_distribution': 'beta',
                                   'exposure_model_parameters': {'alpha': 3, 'beta': 3},
                                   'exposure_model_truncation': {'lower': 0.88, 'upper': 1.2},
                                   'mapping_identifier_classical': ['w_period'],
                                   'mapping_identifier_Berkson': ['year', 'object'],  # kann potentiell eine liste mit bis zu 3 strigs sein
                                   'name_obs_values': 'w_classical'
                                   # 'name_obs_values': 'w_Berkson'
                                   }
                         },
        'equilibrium': {'gamma': {'classical_error': {'sd': 0.23, 'structure': 'multiplicative', 'proposal_sd': 0.01},
                                  # 'classical_error': {'sd': 0, 'structure': 'multiplicative', 'proposal_sd': 0.01},
                                  'Berkson_error': {'sd': 0.69, 'structure': 'multiplicative', 'proposal_sd': 0.1},
                                  # 'Berkson_error': {'sd': 0, 'structure': 'multiplicative', 'proposal_sd': 0.1},
                                  'exposure_model_distribution': 'beta',
                                  'exposure_model_parameters': {'alpha': 3, 'beta': 3},
                                  'exposure_model_truncation': {'lower': 0.2, 'upper': 0.6},
                                  'mapping_identifier_classical': ['g_period', 'object'],
                                  'mapping_identifier_Berkson': ['year', 'object'], 
                                  'name_obs_values': 'g_classical'
                                  # 'name_obs_values': 'g_Berkson'
                                  },
                        },
        # 'M4': {'E_Rn': {'classical_error': {'sd': 0.936, 'structure': 'multiplicative', 'proposal_sd': 0.12},
                        # # 'classical_error': {'sd': 0, 'structure': 'additive', 'proposal_sd': 0.1},
                        # 'Berkson_error': {'sd': 0},
                        # 'exposure_model_distribution': 'lognorm',
                        # 'exposure_model_parameters': {'mu': 2, 'sigma': 0.8},
                        # 'exposure_model_truncation': {},
                        # 'mapping_identifier_classical': ['year', 'object'],
                        # 'name_obs_values': 'C_Rn_obs'
                        # # 'name_obs_values': 'C_Rn_true'
                        # },
               # },

        }



################################################################################
################ one chainData_1
################################################################################

disease_model = "cox_like"
# disease_model = "ERR"

chain = 'chain1'
start_values = generate_start_values(disease_model=disease_model)
proposal_sd = generate_proposal_sds()
prior_parameters = generate_prior_parameters()

# for measurement_model in start_values[chain]['prior_parameters']:
    # for uncertain_factor in start_values[chain]['prior_parameters'][measurement_model]:
        # print(measurement_model)
        # print(uncertain_factor)
        # # uncertainty_characteristics[measurement_model][uncertain_factor]['exposure_model_parameters'] = start_values[chain]['prior_parameters'][measurement_model][uncertain_factor]
        # start_values[chain]['prior_parameters'][measurement_model][uncertain_factor] = uncertainty_characteristics[measurement_model][uncertain_factor]['exposure_model_parameters'] 

s = np.array([0, 40, 55, 75, 104])
path_results = path + 'results/'

import pudb; pu.db
mcmc = MCMC(data=data, uncertainty_characteristics=uncertainty_characteristics,
            s=s, path_results=path_results, proposal_sd=proposal_sd,
            prior_parameters=prior_parameters, start_values=start_values,
            informative_priors=False, chain=chain,
            disease_model=disease_model, fixed_parameters=False)


iterations = 100
burnin = 1
phases = 1
thin = 1
# iterations = 5000
# burnin = 1
# phases = 10
# thin = 1


# mcmc.latent_variables.uncertain_factors['M2']['C_Rn'].prior_parameters['mu'].proposal_sd
# mcmc.latent_variables.uncertain_factors['M2']['C_Rn'].prior_parameters['mu'].acceptance[:200].mean()
# mcmc.latent_variables.uncertain_factors['M2']['C_Rn'].prior_parameters['sigma'].acceptance[:200].mean()
# mcmc.latent_variables.uncertain_factors['M2']['C_Rn'].prior_parameters['sigma'].proposal_sd
# mcmc.run_adaptive_phase(50, 3)



mcmc.run_adaptive_algorithm(iterations=iterations, burnin=burnin,
                            adaptive_phases=phases, save_chains=True,
                            thin=thin)



raise ValueError

import matplotlib.pyplot as plt
import seaborn as sns

x = np.genfromtxt(path_results + 'chain1_results_beta.txt', delimiter=',')
plt.plot(x)
plt.show()

x = np.genfromtxt(path_results + 'chain1_results_lambda1.txt', delimiter=',')
plt.plot(x)
plt.show()

x = np.genfromtxt(path_results + 'chain1_results_lambda4.txt', delimiter=',')
plt.plot(x)
plt.show()

x = np.genfromtxt(path_results + 'chain1_results_UF_C_Rn.txt', delimiter=' ')
plt.plot(x)
plt.show()

x = np.genfromtxt(path_results + 'chain1_results_UF_gamma.txt', delimiter=' ')
plt.plot(x)
plt.show()


x = np.genfromtxt(path_results + 'chain1_results_UF_omega.txt', delimiter=' ')
plt.plot(x)
plt.show()


x = np.genfromtxt(path_results + 'chain1_results_UF_C_RPD.txt', delimiter=' ')
plt.plot(x)
plt.show()

x = np.genfromtxt(path_results + 'chain1_results_UF_zeta.txt', delimiter=' ')
plt.plot(x)
plt.show()




x = np.genfromtxt(path_results + 'chain1_results_M2_C_Rn_mu.txt', delimiter=' ')
plt.plot(x)
plt.hlines(data.C_Rn_true.mean(), xmin=0, xmax=x.shape[0])

x = np.genfromtxt(path_results + 'chain1_results_M2_C_Rn_sigma.txt', delimiter=' ')
plt.plot(x)
plt.hlines(data.C_Rn_true.std(), xmin=0, xmax=x.shape[0])
plt.show()











# values
x = np.genfromtxt(path_results + 'chain1_results_beta.txt', delimiter=',')
x.mean()
x.quantile(0.025)
np.quantile(x,[0.025, 0.975])

x = np.array([np.genfromtxt(path_results + f'chain1_results_lambda{i}.txt', delimiter=',') for i in range(1,5)])
np.apply_along_axis(np.mean, 1, x)

x = np.genfromtxt(path_results + 'chain1_results_UF_omega.txt', delimiter=' ')
np.apply_along_axis(np.mean, 0, x)




################################################################################
############### test on 20 datasets
################################################################################

#################### chack more datasets
# parallel 10 datasets


def run_dataset(nb, seed, chain='chain1'):
    # data = basics.read_data(path + f"data/S2/Data_{nb}.csv")
    # data = basics.read_data(path + f"data/S3/Data_{nb}.csv")
    # data = basics.read_data(path + f"data/EHR/S3/Data_{nb}.csv")
    # data = basics.read_data(path + f"data/Scenario_Level_2_Berkson/Data_{nb}.csv", assign_groups=True)
    # data = basics.read_data(path + "data/Scenario_Level2_logNV_classical/Data_{nb}.csv", assign_groups=True)
    # data = basics.read_data(path + f"data/M2M3/Data_{nb}.csv")
    # data = basics.read_data(path + f"data/M2M3M4/Data_{nb}.csv")
    data = basics.read_data(path + f"data/M4/Data_{nb}.csv")
    data['tau'] = 1
    # data = basics.read_data(path + f"data/EHR/S3/Data_{nb}.csv")

    path_results = path + f'/results/parallel_results/{nb}/'
    if os.path.exists(path_results):
        print('Attention! Result path exists. Results may get overwritten!')
    else:
        os.mkdir(path_results)


    disease_model = 'cox_like'
    # disease_model = 'ERR'

    start_values = generate_start_values(seed, disease_model)
    proposal_sd = generate_proposal_sds()
    prior_parameters = generate_prior_parameters()

    s = np.array([0, 40, 55, 75, 104])
    # write start values for prior parameters in uncertainty characteristics (this acttually gets done when the code for one chain is called earlier, but then it gets overwritten - so its fine)
    # for measurement_model in uncertainty_characteristics:
        # for uncertain_factor in start_values[chain]['prior_parameters'][measurement_model]:
            # uncertainty_characteristics[measurement_model][uncertain_factor]['exposure_model_parameters'] = start_values[chain]['prior_parameters'][measurement_model][uncertain_factor]

    mcmc = MCMC(data=data,uncertainty_characteristics=uncertainty_characteristics, s=s, path_results=path_results, proposal_sd=proposal_sd,
            prior_parameters=prior_parameters, start_values=start_values,
            informative_priors=False, chain=chain, 
            disease_model=disease_model, fixed_parameters=False)

    iterations = 20
    burnin = 1
    phases = 0
    # iterations = 2000
    # burnin = 5
    # phases = 5

    mcmc.run_adaptive_algorithm(iterations=iterations, burnin=burnin,
            adaptive_phases=phases, save_chains=True)







if __name__ == '__main__':
    import multiprocessing
    import time
    import matplotlib.pyplot as plt

    nb_data_sets = 5
    cores = 5

    datasets = list(range(1,nb_data_sets+1))
    seeds = np.random.randint(1000, size=nb_data_sets)

    # run_dataset(1,1)

    t = time.time()
    with multiprocessing.Pool(cores) as pool:
        res = pool.starmap(run_dataset, [(nb, seeds[nb - 1]) for nb in datasets])
    print("Full calculation time: " + str(time.time() - t))


    beta = []
    lambda1 = []
    lambda4 = []
    for i in datasets:
        path_i = path + f'/results/parallel_results/{i}/'
        print(path_i + 'chain1_results_beta.txt')
        beta.append(np.genfromtxt(path_i + 'chain1_results_beta.txt', delimiter=','))
        lambda1.append(np.genfromtxt(path_i + 'chain1_results_lambda1.txt', delimiter=','))
        lambda4.append(np.genfromtxt(path_i + 'chain1_results_lambda4.txt', delimiter=','))
        # plots
        x = np.genfromtxt(path_i + 'chain1_results_beta.txt', delimiter=',')
        plt.plot(x)

    plt.show()





    list(map(lambda x: np.mean(x), beta))

    def get_infos(x):
        q = np.quantile(x, (0.025, 0.975))
        m = x.mean()
        inside = 0.3 > q[0] and 0.3 < q[1]
        return inside
        # inside = 0.00006 > q[0] and 0.00006 < q[1]
        # inside = 0.01000 > q[0] and 0.01000 < q[1]
        return {
                'mean': m,
                'qs': q,
                'inside': inside
                }


    results_beta = list(map(get_infos, beta))
    results_lambda1 = list(map(get_infos, lambda1))
    results_lambda4 = list(map(get_infos, lambda4))

    # # overall mean
    # mean = []
    # for d in results:
        # mean.append(d['mean'])
    # print("overall deviation from mean")
    # print(np.array(mean).mean() - 0.3)
    # # print((np.array(mean).mean() - 0.00006)/0.00006)

    # # nb of overlapping CIs
    # ov = []
    # for d in results:
        # ov.append(d['inside'])
    # print("\n overall coverage:")
    # print(np.array(ov).sum() / len(ov))
    
