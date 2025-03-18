import pandas as pd
import numpy as np
import scipy.stats as stats
import copy
from numba import jit

import wismut.basics as basics  


def parameter_update(parameter, current_values: dict, latent_variable) -> None:
    """
    This function is used to make an update for a Parameter.

    :param Parameter: A Parameter instance which should be updated.
    :param current_values: A dict with the values of all parameters.
    :param latent_variable: A LatentVariables instance. Is used to get the current latent exposure.
    """
    theta_t = parameter.get_current_value()
    theta_cand = parameter.propose_value()
    ratio = parameter.prior_ratio(theta_cand)

    # all values are the same except one
    candidate_values = copy.deepcopy(current_values)
    candidate_values[parameter.name] = theta_cand

    # overwrite the candidate values in the LV with the current values (equal for parameter update)
    # latent_variable.candidate_values_cum = latent_variable.values_cum
    latent_variable.candidate_values_cum = latent_variable.get_values_cum()

    ratio *= parameter.likelihood_ratio(current_values, candidate_values, latent_variable)

    accept = (np.random.uniform(0, 1) < ratio)
    if accept:
        theta_t = theta_cand

    parameter.samples[parameter.i + 1] = theta_t
    parameter.acceptance[parameter.i] = accept
    parameter.i += 1




def prior_update(prior_parameter, current_prior_values: dict) -> None:
    """
    Makes an update for a prior parameter (currently not used)

    :param prior_parameter: An instance of PriorParameter to be updated.
    :param: current_prior_values: a dict with the current prior values.
    """
    # selbe zeile mit names - vieleicht brauchen wir diese doch noch....
    # def prior_update(prior_parameter, current_values, current_prior_values, name_parent):

    # values = prior_parameter.extract_values(current_values)  # das war der alte fall wenn man in einem hierarchischen modell ist
    values = prior_parameter.uf_values
    theta_t = prior_parameter.get_current_value()
    theta_cand = prior_parameter.propose_value()

    ratio = prior_parameter.prior_ratio(theta_cand)


    # all values are the same except one
    candidate_prior_values = copy.deepcopy(current_prior_values)

    candidate_prior_values[prior_parameter.name] = theta_cand
    ratio *= prior_parameter.likelihood_ratio(current_prior_values, candidate_prior_values)

    if prior_parameter.calculate_proposal_ratio:
        ratio *= prior_parameter.proposal_ratio(theta_cand, prior_parameter.name_parent)

    accept = (np.random.uniform(0, 1) < ratio)
    if accept:
        theta_t = theta_cand

    prior_parameter.samples[prior_parameter.i + 1] = theta_t
    prior_parameter.acceptance[prior_parameter.i] = accept
    prior_parameter.i += 1



def ratio_likelihood_beta_trunc(values: np.ndarray, current_prior_values: dict, candidate_prior_values: dict, exposure_model_truncation: dict) -> float:
    """
    Calculates the the likelihood ratio for a beta distribution 
    """
    alpha_cand = candidate_prior_values['alpha']
    beta_cand = candidate_prior_values['beta']
    alpha_curr = current_prior_values['alpha']
    beta_curr = current_prior_values['beta']

    values = (values - exposure_model_truncation['lower']) / (exposure_model_truncation['upper']-exposure_model_truncation['lower'])

    ratio = np.exp(np.sum(
            stats.beta.logpdf(values, a=alpha_cand, b=beta_cand) - \
            stats.beta.logpdf(values, a=alpha_curr, b=beta_curr)
            ))

    return(ratio)



# def ratio_likelihood_normal(values, current_prior_values, candidate_prior_values):
    # """
    # Calculates the the likelihood ratio for a normal distribution.
    # """
    # ratio = np.prod(
                # stats.norm.pdf(values, loc = candidate_prior_values['mean'], scale = candidate_prior_values['sd'])/
                # stats.norm.pdf(values, loc = current_prior_values['mean'], scale = current_prior_values['sd'])
            # )
    # return(ratio)




def ratio_likelihood_normal(values: np.array, current_prior_values: dict, candidate_prior_values: dict) -> float:
    """
    Calculates the likelihood ratio where a Gaussian distibution is assumed. The ratio reduces to a smaller expression which is more efficient to calculate.
    """
    mu_curr = current_prior_values['mu']
    sd_curr = current_prior_values['sigma']
    mu_cand = candidate_prior_values['mu']
    sd_cand = candidate_prior_values['sigma']

    return np.exp((values.shape[0] * (np.log(sd_curr) - np.log(sd_cand)) +
                  1 / (2 * sd_curr**2) * np.sum((values - mu_curr)**2) -
                  1 / (2 * sd_cand**2) * np.sum((values - mu_cand)**2))
                  )


# alternative with at least 15 times speedup using jit
def normal_likelihood_simple(values: np.ndarray, mu_curr: float, sd_curr: float, mu_cand: float, sd_cand: float) -> float:
    """
    Calculates the likelihood ratio where a Gaussian distibution is assumed. The ratio reduces to a smaller expression which is more efficient to calculate.
    """
    return np.exp((values.shape[0] * (np.log(sd_curr) - np.log(sd_cand)) +
                  1 / (2 * sd_curr**2) * np.sum((values - mu_curr)**2) -
                  1 / (2 * sd_cand**2) * np.sum((values - mu_cand)**2))
                  )

normal_likelihood_simple_jit = jit(normal_likelihood_simple, nopython=True)



def ratio_likelihood_normal_fast(values: np.ndarray, current_prior_values: dict, candidate_prior_values: dict) -> float:
    """
    Calculates the likelihood ratio where a Gaussian distibution is assumed. The ratio reduces to a smaller expression which is more efficient to calculate.
    """
    mu_curr = current_prior_values['mu']
    sd_curr = current_prior_values['sigma']
    mu_cand = candidate_prior_values['mu']
    sd_cand = candidate_prior_values['sigma']
    # return foo_sub(values, mu_curr, sd_curr, mu_cand, sd_curr)
    return normal_likelihood_simple_jit(values, mu_curr, sd_curr, mu_cand, sd_cand)





def ratio_likelihood_trunc_normal(values: np.ndarray, current_prior_values: dict, candidate_prior_values: dict) -> float:
    """
    Calculates the the likelihood ratio for a normal distribution truncated at 0 (only positive values)
    """
    mean_cand = candidate_prior_values['mu']
    sd_cand = candidate_prior_values['sigma']
    mean_curr = current_prior_values['mu']
    sd_curr = current_prior_values['sigma']

    a_trunc_cand = (0 - mean_cand) / sd_cand
    a_trunc_curr = (0 - mean_curr) / sd_curr

    ratio = np.exp(np.sum(
            stats.truncnorm.logpdf(values, a=a_trunc_cand, b=np.inf, loc=mean_cand, scale=sd_cand) - \
            stats.truncnorm.logpdf(values, a=a_trunc_curr, b=np.inf, loc=mean_curr, scale=sd_curr)
            ))
    return(ratio)



def ratio_likelihood_log_normal(values: np.ndarray, current_prior_values: dict, candidate_prior_values: dict) -> float:
    """
    Calculates the the likelihood ratio for a log-normal distribution.
    """
    mean_cand = candidate_prior_values['mu']
    sd_cand = candidate_prior_values['sigma']
    mean_curr = current_prior_values['mu']
    sd_curr = current_prior_values['sigma']

    ratio = np.exp(np.sum(
                stats.lognorm.logpdf(values, s=sd_cand, scale=np.exp(mean_cand)) - \
                stats.lognorm.logpdf(values, s=sd_curr, scale=np.exp(mean_curr))
                ))
    return(ratio)


# alternative with at least 15 times speedup using jit
def normal_likelihood_simple(values: np.ndarray, mu_curr: float, sd_curr: float, mu_cand: float, sd_cand: float) -> float:
    """
    Calculates the likelihood ratio where a Gaussian distibution is assumed. The ratio reduces to a smaller expression which is more efficient to calculate.
    """
    return np.exp((values.shape[0] * (np.log(sd_curr) - np.log(sd_cand)) +
                  1 / (2 * sd_curr**2) * np.sum((values - mu_curr)**2) -
                  1 / (2 * sd_cand**2) * np.sum((values - mu_cand)**2))
                  )


normal_likelihood_simple_jit = jit(normal_likelihood_simple, nopython=True)


def ratio_likelihood_normal_fast(values: np.ndarray, current_prior_values: dict, candidate_prior_values: dict) -> float:
    """
    Calculates the likelihood ratio where a Gaussian distibution is assumed. uses jitted version
    """
    mu_curr = current_prior_values['mu']
    sd_curr = current_prior_values['sigma']
    mu_cand = candidate_prior_values['mu']
    sd_cand = candidate_prior_values['sigma']
    # return foo_sub(values, mu_curr, sd_curr, mu_cand, sd_curr)
    return normal_likelihood_simple_jit(values, mu_curr, sd_curr, mu_cand, sd_cand)



def prior_ratio_exponential(theta_t: float, theta_cand: float, prior_parameters_theta: dict, informative_priors: bool):
    """
    Calculates the the prior ratio for a exponential distribution.
    """
    ratio = stats.expon.pdf(theta_cand, loc=0, scale=1 / prior_parameters_theta['lambda'])/stats.expon.pdf(
                            theta_t, loc=0, scale=1 / prior_parameters_theta['lambda'])
    return(ratio)


def prior_ratio_gamma(theta_t: float, theta_cand: float, prior_parameters_theta: dict, informative_priors: dict):
    """
    Calculates the the likelihood ratio for a gamma distribution.
    """

    if informative_priors:
        ratio =  stats.gamma.pdf(theta_cand, a=prior_parameters_theta['shape'],
                                 scale=prior_parameters_theta['scale']) / stats.gamma.pdf(theta_t,
                                 a=prior_parameters_theta['shape'],
                                 scale=prior_parameters_theta['scale'])
    else:
        prior_min =prior_parameters_theta['min']
        prior_max =prior_parameters_theta['max']
        ratio = stats.uniform.pdf(theta_cand, loc=prior_min,
                                  scale=prior_max - prior_min) / stats.uniform.pdf(theta_t,
                                  loc=prior_min, scale=prior_max - prior_min)
    return(ratio)



# Old version, the version below is much faster using properties of the uniform distribution
# def prior_ratio_uniform(theta_t, theta_cand, prior_parameters_theta, informative_priors=False) -> float:
    # """
    # Calculates the the prior ratio for a Uniform distribution.
    # """
    # prior_min = prior_parameters_theta['min']
    # prior_max = prior_parameters_theta['max']
    # ratio = stats.uniform.pdf(theta_cand, loc=prior_min,
                                   # scale=prior_max-prior_min)/stats.uniform.pdf(theta_t,
                                   # loc=prior_min, scale=prior_max-prior_min)
    # return(ratio)

def prior_ratio_uniform(theta_t: float, theta_cand: float, prior_parameters_theta: dict, informative_priors: bool = False) -> None:
    """
    Calculates the the prior ratio for a Uniform distribution.
    """
    # prior ratio is 1 as long as theta_cand lies inside the min and max.
    # otherwise it is 0. Theta_t does not affect the ratio (see desity of
    # Uniform distribution )
    return(1.0 if ((prior_parameters_theta['min'] < theta_cand) & (theta_cand < prior_parameters_theta['max'])) else 0)




def prior_ratio_beta(theta_t: float, theta_cand: float, prior_parameters_theta: dict, informative_priors: bool) -> None:
    """
    Calculates the the prior ratio for a Beta distribution.
    """
    ratio = stats.beta.pdf(theta_cand, a=prior_parameters_theta['a'],
                                       b=prior_parameters_theta['b'],
                                       scale=prior_parameters_theta['scale']
#                                       loc = prior_parameters_theta['min'],
#                                       scale = prior_parameters_theta['max'] -
#                                       prior_parameters_theta['min'])
                                     ) /stats.beta.pdf(theta_t,
                                       a=prior_parameters_theta['a'],
                                       b=prior_parameters_theta['b'],
                                       scale=prior_parameters_theta['scale'])
#                                       loc = prior_parameters_theta['min'],
#                                       scale = prior_parameters_theta['max'] -
#                                       prior_parameters_theta['min'])
    return(ratio)


def prior_ratio_inv_gamma(theta_t: float, theta_cand: float, prior_parameters_theta: dict, informative_priors: bool) -> float:
    """
    Calculates the the prior ratio for an inverse gamma distribution.
    """

    if informative_priors:
        ratio = stats.invgamma.pdf(theta_cand, a=prior_parameters_theta['shape'],
                                   scale=prior_parameters_theta['scale']) / stats.invgamma.pdf(theta_t,
                                   a=prior_parameters_theta['shape'],
                                   scale=prior_parameters_theta['scale'])
    else:
        prior_min = prior_parameters_theta['min']
        prior_max = prior_parameters_theta['max']
        ratio = stats.uniform.pdf(theta_cand, loc=prior_min,
                                  scale=prior_max - prior_min) / stats.uniform.pdf(theta_t,
                                  loc=prior_min, scale=prior_max - prior_min)
    return(ratio)



def prior_ratio_lognormal(theta_t: float, theta_cand: float, prior_parameters_theta: dict):
    """
    Calculates the the prior ratio for a lognormal distribution.
    """
    # ratio = ((1/theta_cand)*np.exp(-((np.log(theta_cand)-prior_parameters_theta['mean'])**2)/(2*(prior_parameters_theta['sd']**2)))/
    # (1/theta_t)*np.exp(-((np.log(theta_t)-prior_parameters_theta['mean'])**2)/(2*(prior_parameters_theta['sd']**2))))
    ratio = (stats.lognorm.pdf(theta_cand, s=prior_parameters_theta['sd'], scale=np.exp(prior_parameters_theta['mean']))/
             stats.lognorm.pdf(theta_t, s=prior_parameters_theta['sd'], scale=np.exp(prior_parameters_theta['mean'])))
    return(ratio)

# def prior_ratio_normal(theta_t, theta_cand, prior_parameters_theta, informative_priors):
    # """
    # Calculates the the prior ratio for a normal distribution.
    # """
    # if informative_priors:
        # ratio = stats.norm.pdf(theta_cand,loc=prior_parameters_theta['mean'],
                            # scale=prior_parameters_theta['sd'])/stats.norm.pdf(theta_t,
                            # loc=prior_parameters_theta['mean'],
                            # scale=prior_parameters_theta['sd'])
    # else:
        # prior_min  = prior_parameters_theta['min']
        # prior_max  = prior_parameters_theta['max']
        # ratio =  stats.uniform.pdf(theta_cand, loc= prior_min,
                                   # scale = prior_max-prior_min)/stats.uniform.pdf(theta_t,
                                   # loc = prior_min, scale = prior_max-prior_min)
    # return(ratio)


def prior_ratio_normal(theta_t: float, theta_cand: float, prior_parameters_theta: dict, informative_priors: bool) -> None:
    mu = prior_parameters_theta['mean']
    var = prior_parameters_theta['sd']**2
    return np.exp(1 / (2 * var) * ((theta_t - mu)**2 - (theta_cand - mu)**2))



def prior_ratio_normal_inverse(theta_t: float, theta_cand: float, prior_parameters_theta: dict, informative_priors: dict) -> None:
    """
    Calculates the the prior ratio for a normal distribution where the parameters are getting inverted.
    """
    if informative_priors:
        inv_theta_t = 1 / np.sqrt(theta_t)
        inv_theta_cand = 1 / np.sqrt(theta_cand)

        ratio = stats.norm.pdf(inv_theta_cand, loc=prior_parameters_theta['inv_mean'],  # muss man die dann auf 0 und 1 stzen?
                               scale=prior_parameters_theta['inv_sd'])/stats.norm.pdf(inv_theta_t,
                               loc=prior_parameters_theta['inv_mean'],
                               scale=prior_parameters_theta['inv_sd'])
    else:
        prior_min = prior_parameters_theta['min']
        prior_max = prior_parameters_theta['max']
        ratio = stats.uniform.pdf(theta_cand, loc=prior_min,
                                   scale=prior_max - prior_min)/stats.uniform.pdf(theta_t,
                                   loc=prior_min, scale=prior_max - prior_min)
    return(ratio)



def prior_ratio_cauchy(theta_t: float, theta_cand: float, prior_parameters_theta: dict, informative_priors: bool) -> float:
    """
    Calculates the the prior ratio for a cauchy distribution.
    """
    ratio = (stats.cauchy.pdf(theta_cand, loc=prior_parameters_theta['loc'], scale=prior_parameters_theta['scale']) /
             stats.cauchy.pdf(theta_t, loc=prior_parameters_theta['loc'], scale=prior_parameters_theta['scale']))
    return ratio




# proposals

def proposal_normal(theta_t: float, proposal_sd: float) -> float:
    """
    Proposes a value from a normal distribution
    """
    theta_cand = theta_t + stats.norm.rvs(loc=0, scale=proposal_sd, size=theta_t.shape)
    return(theta_cand)


def proposal_trunc(theta_t: float, proposal_sd: float) -> float:
    """
    Proposes a value from a normal distribution truncated by 0.
    """
    a_trunc = (0 - theta_t) / proposal_sd
    theta_cand = stats.truncnorm.rvs(a=a_trunc, b=np.inf,loc=theta_t,
                                      scale=proposal_sd, size=theta_t.shape)
    return(theta_cand)
    
def proposal_double_trunc(theta_t: float, proposal_sd: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """
    Proposes a value from a normal distribution truncated by a_trunc and b_trunc
    """
    # a and b in truncnorm muessen als relative werte zum mittelwert und sd
    # angegeben weren!
    a_trunc = (lower - theta_t) / proposal_sd
    b_trunc = (upper - theta_t) / proposal_sd
    theta_cand = stats.truncnorm.rvs(a=a_trunc, b=b_trunc, loc=theta_t,
                                            scale=proposal_sd, size=theta_t.shape)
    return(theta_cand)




def log_proposal_ratio_trunc(theta_curr: float, theta_cand: float, proposal_sd: dict) -> float:
    """
    Calculates the log proposal ratio for a normal distribution truncated at 0
    """
    a_trunc_curr = (0 - theta_cand) / proposal_sd
    a_trunc_cand = (0 - theta_curr) / proposal_sd

    log_ratio = np.sum(stats.truncnorm.logpdf(theta_curr, a=a_trunc_curr, b=np.inf, loc=theta_cand, scale=proposal_sd) - \
                       stats.truncnorm.logpdf(theta_cand, a=a_trunc_cand, b=np.inf, loc=theta_curr, scale=proposal_sd)
                       )
    return log_ratio


def log_proposal_ratio_double_trunc(theta_curr: float, theta_cand: float, proposal_sd: float, lower: float, upper: float) -> float:
    """
    Calculates the log proposal ratio for a truncated normal distribution in a certain range (with a lower and upper bound)
    """
    a_trunc_curr = (lower - theta_cand) / proposal_sd
    a_trunc_cand = (lower - theta_curr) / proposal_sd
    b_trunc_curr = (upper - theta_cand) / proposal_sd
    b_trunc_cand = (upper - theta_curr) / proposal_sd

    log_ratio = np.sum(stats.truncnorm.logpdf(theta_curr, a = a_trunc_curr, b = b_trunc_curr,loc = theta_cand, scale = proposal_sd) - \
           stats.truncnorm.logpdf(theta_cand, a = a_trunc_cand, b =  b_trunc_cand,loc = theta_curr, scale = proposal_sd)
          )
    return log_ratio




priors = {'normal': prior_ratio_normal,
          'gamma': prior_ratio_gamma,
          }



# priors = {'beta': prior_ratio_normal,
          # 'lambda1': prior_ratio_gamma, 'lambda2': prior_ratio_gamma,
          # 'lambda3': prior_ratio_gamma, 'lambda4': prior_ratio_gamma,
          # # 'C_Rn_mu': prior_ratio_uniform,
          # # 'C_Rn_sigma': prior_ratio_uniform
          # 'C_Rn_mu': prior_ratio_normal,
          # 'C_Rn_sigma': prior_ratio_normal,
          # 'C_RPD_mu': prior_ratio_normal,
          # 'C_RPD_sigma': prior_ratio_normal,
          # 'C_Exp_mu': prior_ratio_normal,
          # 'C_Exp_sigma': prior_ratio_normal,
          # 'E_Rn_mu': prior_ratio_normal,
          # 'E_Rn_sigma': prior_ratio_normal,
          # 'zeta_alpha': prior_ratio_normal,
          # 'zeta_beta': prior_ratio_normal,
          # 'gamma_alpha': prior_ratio_normal,
          # 'gamma_beta': prior_ratio_normal,
          # 'phi_alpha': prior_ratio_normal,
          # 'phi_beta': prior_ratio_normal,
          # 'omega_alpha': prior_ratio_normal,
          # 'omega_beta': prior_ratio_normal,
          # }

proposals = {'beta': proposal_normal,
             'lambda1': proposal_trunc, 'lambda2': proposal_trunc,
             'lambda3': proposal_trunc, 'lambda4': proposal_trunc,
             'C_Rn_mu': proposal_normal,  # in priniciple, the mean of a truncated normal (trunc at 0) could be < 0 when sd is sufficiently large
             'C_Rn_sigma': proposal_trunc,
             'C_RPD_mu': proposal_normal,  # in priniciple, the mean of a truncated normal (trunc at 0) could be < 0 when sd is sufficiently large
             'C_RPD_sigma': proposal_trunc,
             'E_Rn_mu': proposal_normal,  # in priniciple, the mean of a truncated normal (trunc at 0) could be < 0 when sd is sufficiently large
             'E_Rn_sigma': proposal_trunc,
             'E_RPD_mu': proposal_normal,  # in priniciple, the mean of a truncated normal (trunc at 0) could be < 0 when sd is sufficiently large
             'E_RPD_sigma': proposal_trunc,
             'C_Exp_mu': proposal_normal,  # in priniciple, the mean of a truncated normal (trunc at 0) could be < 0 when sd is sufficiently large
             'C_Exp_sigma': proposal_trunc,
             'zeta_alpha': proposal_trunc,
             'zeta_beta': proposal_trunc,
             'gamma_alpha': proposal_trunc,
             'gamma_beta': proposal_trunc,
             'phi_alpha': proposal_trunc,
             'phi_beta': proposal_trunc,
             'omega_alpha': proposal_trunc,
             'omega_beta': proposal_trunc,
             }

prior_likelihoods = {'C_Rn_mu': {'norm': ratio_likelihood_trunc_normal, 'lognorm': ratio_likelihood_log_normal},
                     'C_Rn_sigma': {'norm': ratio_likelihood_trunc_normal, 'lognorm': ratio_likelihood_log_normal},
                     'C_RPD_mu': {'norm': ratio_likelihood_trunc_normal, 'lognorm': ratio_likelihood_log_normal},
                     'C_RPD_sigma': {'norm': ratio_likelihood_trunc_normal, 'lognorm': ratio_likelihood_log_normal},
                     'C_Exp_mu': {'norm': ratio_likelihood_trunc_normal, 'lognorm': ratio_likelihood_log_normal},
                     'C_Exp_sigma': {'norm': ratio_likelihood_trunc_normal, 'lognorm': ratio_likelihood_log_normal},
                     'E_Rn_mu': {'norm': ratio_likelihood_trunc_normal, 'lognorm': ratio_likelihood_log_normal},
                     'E_Rn_sigma': {'norm': ratio_likelihood_trunc_normal, 'lognorm': ratio_likelihood_log_normal},
                     'zeta_alpha': {'beta': ratio_likelihood_beta_trunc},
                     'zeta_beta': {'beta': ratio_likelihood_beta_trunc},
                     'gamma_alpha': {'beta': ratio_likelihood_beta_trunc},
                     'gamma_beta': {'beta': ratio_likelihood_beta_trunc},
                     'phi_alpha': {'beta': ratio_likelihood_beta_trunc},
                     'phi_beta': {'beta': ratio_likelihood_beta_trunc},
                     'omega_alpha': {'beta': ratio_likelihood_beta_trunc},
                     'omega_beta': {'beta': ratio_likelihood_beta_trunc},
                     }

# prior_likelihoods = {'C_Rn_mu': ratio_likelihood_trunc_normal,
                     # 'C_Rn_sigma': ratio_likelihood_trunc_normal,
                     # 'C_RPD_mu': ratio_likelihood_trunc_normal,
                     # 'C_RPD_sigma': ratio_likelihood_trunc_normal,
                     # # 'C_Rn_mu': ratio_likelihood_log_normal,
                     # # 'C_Rn_sigma': ratio_likelihood_log_normal,
                     # 'E_Rn_mu': ratio_likelihood_log_normal,
                     # 'E_Rn_sigma': ratio_likelihood_log_normal,
                     # 'lambda1': proposal_trunc, 'lambda2': proposal_trunc,
                     # 'lambda3': proposal_trunc, 'lambda4': proposal_trunc
                     # }



def predictor_cox(values: dict, cumulated_exposures: np.ndarray) -> float:
    """
    Calculates the predictor for a cox model
    """
    return(np.exp(values['beta'] * cumulated_exposures))


def predictor_ERR(values: dict, cumulated_exposures: np.ndarray) -> None:
    """
    Calculates the predictor for a ERR model
    """
    return(1 + values['beta'] * cumulated_exposures)


predictor = {'cox_like': predictor_cox, 'ERR': predictor_ERR}


# the same as above on log scale
def log_predictor_cox(values: dict, cumulated_exposures: np.ndarray) -> np.ndarray:
    """
    Calculates the log predictor for a cox model
    """
    return(values['beta'] * cumulated_exposures)


def log_predictor_ERR(values, cumulated_exposures) -> np.ndarray:
    """
    Calculates the predictor for a ERR model
    """
    return(np.log(1 + values['beta'] * cumulated_exposures))


log_predictor = {'cox_like': log_predictor_cox, 'ERR': log_predictor_ERR}




 ############################################################################
 # likelihood on log scale i.e. sum()
 ############################################################################
def ratio_log_likelihood(disease_model: str, data: pd.DataFrame, values_t: dict,
                         values_cand: dict, latent_variable,
                         s: np.ndarray, simplify: str = 'beta') -> float:
    '''
    Determines the  log-likelihood ratio for two different sets of parameters for all
    disease models you can wish for.

    :param disease_model: A character defining the disease model. Options are 'cox_like' and 'ERR'
    :param data: A pd.DataFrame with the used data.
    :param values_t: A dict holding the current parameter values.
    :param values_cand: A dict holding the candidate parameter values.
    :param latent_variable: a latent variable which holds current AND candidate values
    :param s: A np.ndarray with the breakpoints of the baseline hazarad
    :param simplify: A string defining if a simplification is possible (for certain parameters), options are 'Xbeta' and 'lambda'
    '''
    calculate_likelihood = True

    # constraint positivity, For ERR the hazard must be positive
    if ('ERR' in disease_model):
        test = np.any(predictor[disease_model](values_cand, latent_variable.candidate_values) <= 0)
        if (test):
            ratio = -np.inf
            calculate_likelihood = False

    if calculate_likelihood:
        lambda_t = np.array([values_t['lambda1'], values_t['lambda2'],
                    values_t['lambda3'], values_t['lambda4']])
        lambda_cand = np.array([values_cand['lambda1'], values_cand['lambda2'],
                    values_cand['lambda3'], values_cand['lambda4']])
        inter = s[1:]-s[:-1]

        dot_product_t = basics.calculate_baseline_hazard(lambda_t, inter)
        dot_product_cand = basics.calculate_baseline_hazard(lambda_cand, inter)

        # h_i(t)
        first_term = calculate_first_term_log_likelihood(disease_model, data, values_t, values_cand,
                                             lambda_t, lambda_cand,latent_variable, simplify)
        # S_i(t)/S_i(trunc) eq.5.10
        second_term = np.sum( # lambda_t[data['I']] ist lambda_j
                         (
                             -((lambda_cand[data['I'].values-1]*(data['t']-s[data['I'].values-1])+ dot_product_cand[data['I'].values-1]))+

                          (lambda_cand[data['I_trunc'].values-1]*(data['truncation']-s[data['I_trunc'].values-1])+dot_product_cand[data['I_trunc'].values-1])
                          )*
                         predictor[disease_model](values_cand, latent_variable.candidate_values_cum)

                         + ((lambda_t[data['I'].values-1]*(data['t']-s[data['I'].values-1])+ dot_product_t[data['I'].values-1])-

                            (lambda_t[data['I_trunc'].values-1]*(data['truncation']-s[data['I_trunc'].values-1])+dot_product_t[data['I_trunc'].values-1]))*
                         predictor[disease_model](values_t, latent_variable.values_cum)
                         )

        ratio = first_term+second_term

    return ratio


def calculate_first_term_log_likelihood(disease_model, data, values_t, values_cand,lambda_t, lambda_cand, latent_variable, simplify):
    if simplify== 'Xbeta':
        # Achtung! in der form nur für das einfache cox modell gedacht i.e. X*beta
        first_term = np.sum(values_cand['beta']*latent_variable.candidate_values_cum[latent_variable.case]- 
                                  values_t['beta']*latent_variable.values_cum[latent_variable.case])
    else:
        # first_term = np.prod(lambda_cand[data['I'][data['event'].values==1].values-1]/lambda_t[data['I'][data['event'].values==1].values-1])
        # alternativ evtl sum(log(lambda_cand)-log(lambda_t)) ---> schneller? numerisch stabiler? testen und evtl ändern!
        first_term = np.sum(np.log(
            lambda_cand[data['I'][data['event'].values==1].values-1] /
            lambda_t[data['I'][data['event'].values==1].values-1]
                                ))

    # for lambda obly the lambda_cand/lambda_curr on the event positions is relevant
        if not simplify == 'lambda':
            first_term = first_term + np.sum(
                    log_predictor[disease_model](values_cand, latent_variable.candidate_values_cum[latent_variable.case]) -
                    log_predictor[disease_model](values_t, latent_variable.values_cum[latent_variable.case])
                    ) 
        # old- version above sould be the same on log scale
        # first_term = np.prod(
                # (lambda_cand[data['I'][data['event'].values==1].values-1]*predictor[disease_model](values_cand, latent_variable.candidate_values_cum[latent_variable.case]))/
                # (lambda_t[data['I'][data['event'].values==1].values-1]*predictor[disease_model](values_t, latent_variable.values_cum[latent_variable.case]
                                 # )))
    return(first_term)








############################################################################
# likelihood on normal scale i.e. prods
############################################################################

def ratio_likelihood(disease_model: str, data: pd.DataFrame, values_t: dict, values_cand: dict, latent_variable, s, simplify='beta'):
    '''
    Determines the likelihood ratio for two different sets of parameters for all
    disease models you can wish for.

    :param disease_model: A character defining the disease model. Options are 'cox_like' and 'ERR'
    :param data: A pd.DataFrame with the used data.
    :param values_t: A dict holding the current parameter values.
    :param values_cand: A dict holding the candidate parameter values.
    :param latent_variable: a latent variable which holds current AND candidate values
    :param s: A np.ndarray with the breakpoints of the baseline hazarad
    :param simplify: A string defining if a simplification is possible (for certain parameters), options are 'Xbeta' and 'lambda'
    '''
    calculate_likelihood = True

    # constraint positivity, For ERR the hazard must be positive
    if ('ERR' in disease_model):
        test = np.any(predictor[disease_model](values_cand, latent_variable.candidate_values_cum) <= 0)
        if (test):
            ratio = 0
            calculate_likelihood = False

    if calculate_likelihood:
        lambda_t = np.array([values_t['lambda1'], values_t['lambda2'],
                    values_t['lambda3'], values_t['lambda4']])
        lambda_cand = np.array([values_cand['lambda1'], values_cand['lambda2'],
                    values_cand['lambda3'], values_cand['lambda4']])
        inter = s[1:]-s[:-1]

        dot_product_t = basics.calculate_baseline_hazard(lambda_t, inter)
        dot_product_cand = basics.calculate_baseline_hazard(lambda_cand, inter)

        # h_i(t)
        first_term = calculate_first_term_likelihood(disease_model, data, values_t, values_cand,
                                             lambda_t, lambda_cand,latent_variable, simplify)
        # S_i(t)/S_i(trunc) eq.5.10
        second_term = np.exp(np.sum( # lambda_t[data['I']] ist lambda_j
                         (
                             -((lambda_cand[data['I'].values-1]*(data['t']-s[data['I'].values-1])+ dot_product_cand[data['I'].values-1]))+

                          (lambda_cand[data['I_trunc'].values-1]*(data['truncation']-s[data['I_trunc'].values-1])+dot_product_cand[data['I_trunc'].values-1])
                          )*
                         predictor[disease_model](values_cand, latent_variable.candidate_values_cum)

                         + ((lambda_t[data['I'].values-1]*(data['t']-s[data['I'].values-1])+ dot_product_t[data['I'].values-1])-

                            (lambda_t[data['I_trunc'].values-1]*(data['truncation']-s[data['I_trunc'].values-1])+dot_product_t[data['I_trunc'].values-1]))*
                         predictor[disease_model](values_t, latent_variable.values_cum)
                         ))

        # S_i(t)/S_i(trunc) eq.5.10
        # second_term = np.exp(np.sum( # lambda_t[data['I']] ist lambda_j
                         # (
                             # -((lambda_cand[data['I'].values-1]*(data['t']-s[data['I'].values-1])+ dot_product_cand[data['I'].values-1]))+

                          # (lambda_cand[data['I_trunc'].values-1]*(data['truncation']-s[data['I_trunc'].values-1])+dot_product_cand[data['I_trunc'].values-1])
                          # )*
                         # predictor[disease_model](values_cand, cumulated_exposures_cand.values)

                         # + ((lambda_t[data['I'].values-1]*(data['t']-s[data['I'].values-1])+ dot_product_t[data['I'].values-1])-

                            # (lambda_t[data['I_trunc'].values-1]*(data['truncation']-s[data['I_trunc'].values-1])+dot_product_t[data['I_trunc'].values-1]))*
                         # predictor[disease_model](values_t, cumulated_exposures_t.values))
                         # )
        # print(first_term)
        # print(second_term)
        ratio = first_term*second_term

    return ratio



def calculate_first_term_likelihood(disease_model, data, values_t, values_cand,lambda_t, lambda_cand, latent_variable, simplify):
    if simplify== 'Xbeta':
        first_term = np.exp(np.sum(values_cand['beta']*latent_variable.candidate_values_cum[latent_variable.case]- 
                                  values_t['beta']*latent_variable.values_cum[latent_variable.case]))
    elif simplify == 'lambda':
        first_term = np.prod(lambda_cand[data['I'][data['event'].values==1].values-1]/lambda_t[data['I'][data['event'].values==1].values-1])
    elif simplify == 'old_version':
        first_term = np.prod(((lambda_cand[data['I'].values-1]*predictor[disease_model](values_cand, cumulated_exposures_cand.values,))**data['event'].values)/
                        ((lambda_t[data['I'].values-1]*predictor[disease_model](values_t, cumulated_exposures_t.values))**data['event'].values))
    else:
        first_term = np.prod(
                (lambda_cand[data['I'][data['event'].values==1].values-1]*predictor[disease_model](values_cand, latent_variable.candidate_values_cum[latent_variable.case]))/
                (lambda_t[data['I'][data['event'].values==1].values-1]*predictor[disease_model](values_t, latent_variable.values_cum[latent_variable.case]
                                 )))
    return(first_term)



# def likelihood(disease_model,data, values_t, cumulated_exposures, s):
    # '''
        # Calculates the likelihood for all disease models.
        # Warning: Do not delete this function. It is needed to determine the DIC. 
        # '''
    # lambda_t = np.array([values_t['lambda1'], values_t['lambda2'],
                    # values_t['lambda3'], values_t['lambda4']])
    # inter = s[1:]-s[:-1]
    # dot_product = basics.calculate_baseline_hazard(lambda_t, inter)
    # return(((lambda_t[data['I'].values-1]*predictor[disease_model](values_t, cumulated_exposures))**data['event'].values)*np.exp(
    # -((lambda_t[data['I'].values-1]*(data['t']-s[data['I'].values-1])+ 
                        # dot_product[data['I'].values-1])*predictor[disease_model](values_t, cumulated_exposures))+
    # (lambda_t[data['I_trunc'].values-1]*(data['truncation']-s[data['I_trunc'].values-1])+
                    # dot_product[data['I_trunc'].values-1])*predictor[disease_model](values_t, cumulated_exposures)))
