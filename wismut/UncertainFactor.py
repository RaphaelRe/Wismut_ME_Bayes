"""
Ths modeule implements the UncertainFactor and FixedUncertainFactor class.
"""
import numpy as np
import numpy.ma as ma
import copy
import math
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

import wismut.basics as basics  
import wismut.update as update  
from wismut.ErrorComponent import ErrorComponent, BerksonErrorComponent # , MultClassicalErrorComponent
from wismut.Parameter import Parameter, FixedParameter, PriorParameter, PriorParameterVector



class UncertainFactor:
    '''
    This class is one of various UncertainFactor instances held by the LatentVariable class.

    :param name: A strign defining the name of the factor.
    :param data: A pandas Dataframe
    :param uncertainty_characteristics: A dictionary, see LatentVariable class
    :param nb_iterations: An integer defining the initial length of the chains
    :param measurement_model: A string defining the model for this parameter
    '''
    def __init__(self, name: str, data: pd.DataFrame, uncertainty_characteristics: dict,
                 nb_iterations: int = 1000, measurement_model: str = "M99") -> None:
        # print("Initializing uncertain factor " + name)
        if measurement_model == "M99":
            raise ValueError("Error due to wrong model specification. Specify a reasonable measurement_model")
        self.name = name
        self.data = data
        self.measurement_model = measurement_model

        # self.exposed_miners = data.Z != 0  # OLD, only works for simulated data
        if measurement_model == "working_time":
            self.exposed_miners = data.model.isin(['M1a', "M2", "M3", "M6", "M2_Expert"])  # selects all lines where the measurement model is THE model
        elif measurement_model == "activity":
            # self.exposed_miners = data.model.isin(["M2", "M3", "M4"])  # selects all lines where the measurement model is THE model
            self.exposed_miners = data.model.isin(['M1a', "M2", "M2_Expert", "M3", "M4", "M6", "M4_Exploration", "M1a_Expert_WLM", "M2_Expert_WLM", "M3_Expert_WLM", "M6_Expert_WLM"])  # selects all lines where the measurement model is THE model
        elif measurement_model == "equilibrium":
            self.exposed_miners = data.model.isin(['M1a', "M2", "M6", "M2_Expert"])  # selects all lines where the measurement model is THE model
        elif measurement_model == "M1a":
            self.all_exposed_miners = data.model == measurement_model  # is ONLY used in calculate_exposure in the LV
            if self.name in ['C_Rn_old', 'b']:
                self.exposed_miners = (data.C_Rn_old > 0) & (data.model == measurement_model)
            elif self.name == 'C_Rn_ref':
                self.exposed_miners = (data.C_Rn_obs_ref > 0) & (data.model == measurement_model)
            elif self.name == 'tau_e':
                self.exposed_miners = (data.C_Rn_obs_ref > 0) & (data.tau_e_no_error != 1) & (data.model == measurement_model)
        else:
            self.exposed_miners = data.model == measurement_model  # selects all lines where the measurement model is THE model

        classical_sd = uncertainty_characteristics['classical_error']['sd']
        Berkson_sd = uncertainty_characteristics['Berkson_error']['sd']
        self.exposure_model_distribution = uncertainty_characteristics['exposure_model_distribution']
        self.exposure_model_truncation = uncertainty_characteristics['exposure_model_truncation']
        mapping_identifier_classical = uncertainty_characteristics['mapping_identifier_classical']

        name_obs_values = uncertainty_characteristics['name_obs_values']

        self.vectorized_exposure = uncertainty_characteristics.get('vectorized_exposure', False)


        self.classical_error = classical_sd > 0
        self.Berkson_error = Berkson_sd > 0
        self.i = 0
        self.target = 0.2  # target = acceptance rate


        # example:
        # uncertainty_characteristics = {
                # 'classical_error':{'sd': 0.33, 'structure': 'multiplicative', 'proposal_sd': 0.1},
                # 'Berkson_error':{'sd': 0.69, 'structure': 'multiplicative', 'proposal_sd': 0.1},
                # 'exposure_model_distribution': 'beta', ###### or lognorm
                # 'exposure_model_parameters': {'alpha': 3, 'beta': 3}, # nur für Simulation - auf echten Daten ist das kein FixedParameter und wird geschätzt
                # 'exposure_model_truncation': {'lower': 0.88, 'upper': 1.2},  #### is None in the case of lognorm
                # 'mapping_identifier' : ['w_period'], # kann potentiell eine liste mit bis zu 3 strigs sein
                # 'name_obs_values': 'w_classical'
                # 'name_true_values': 'w_Berkson' # eigentlich nur für die Simulation nötig wenn UF fixed, da bekannt - auf realen Daten None
                # }

        self.error_components = {}

        if self.classical_error:
            structure_classical = uncertainty_characteristics['classical_error']['structure']
            if structure_classical == 'multiplicative':
                self.log_LR_measurement_model = lambda proposed_values: self.log_LR_measurement_model_classical_lognormal(proposed_values, classical_sd)

            if self.name == 'phi':
                exposed_non_hewers = (data.Z != 0) & (data.activity != 1)
                self.observed_values = np.concatenate(data[exposed_non_hewers].groupby(mapping_identifier_classical).apply(lambda df: df[name_obs_values].unique()).values)
            elif self.name in ['b']:
                # raise NotImplementedError("Code der nicht getestet ist, existiert nicht! Hier b")
                exposed_non_reference = (data.C_Rn_old > 0) & (data.b_reference != 1)
                self.observed_values = np.concatenate(data[exposed_non_reference].groupby(mapping_identifier_classical).apply(lambda df: df[name_obs_values].unique()).values)
            elif self.name in ['unshared_error', 'shared_error']:
                pass
            else:
                self.observed_values = np.concatenate(data[self.exposed_miners].groupby(mapping_identifier_classical).apply(lambda df: df[name_obs_values].unique()).values)

                if self.vectorized_exposure:
                    if 'cluster' in mapping_identifier_classical:
                        self.exposure_years = np.concatenate(data[self.exposed_miners].groupby(mapping_identifier_classical).apply(basics.return_year).values)
                    else:
                        self.exposure_years = np.concatenate(data[self.exposed_miners].groupby(mapping_identifier_classical).apply(lambda df: df['year'].unique()).values)


            #### initialize exposure_model, proposal_distribution, proposal_log_ratio and true mean values (depends on the structure and type of the error)
            self.prior_parameters = {}
            self.proposal_sd_true_mean = uncertainty_characteristics['classical_error']['proposal_sd']

            if self.exposure_model_distribution == 'beta':
                dist = stats.beta
                # priors get initialized here as FiexedParameter to be able to initialize start values of the uncertain factor
                # in case that the exposure model parameters should be estimated this FixedParameter will be overwritten with
                # a PriorParameter(Vector) by the MCMC class
                self.prior_parameters['alpha'] = FixedParameter('alpha', uncertainty_characteristics['exposure_model_parameters']['alpha'])
                self.prior_parameters['beta'] = FixedParameter('beta', uncertainty_characteristics['exposure_model_parameters']['beta'])
                    
                # likelihood ratio
                self.log_LR_exposure_model = lambda proposed_values: \
                                    np.sum(
                                            dist.logpdf((proposed_values - self.exposure_model_truncation['lower'])/(self.exposure_model_truncation['upper']-self.exposure_model_truncation['lower']), a=self.prior_parameters['alpha'].get_current_value(), b=self.prior_parameters['beta'].get_current_value()) - \
                                            dist.logpdf((self.true_mean_values - self.exposure_model_truncation['lower'])/(self.exposure_model_truncation['upper']-self.exposure_model_truncation['lower']), a=self.prior_parameters['alpha'].get_current_value(), b=self.prior_parameters['beta'].get_current_value())
                                          )
                # proposal distibution
                self.propose_values = lambda: update.proposal_double_trunc(self.true_mean_values, 
                                                                           self.proposal_sd_true_mean,
                                                                           lower=self.exposure_model_truncation['lower'],
                                                                           upper=self.exposure_model_truncation['upper']
                                                                         )

                # proposal log ratio
                self.log_proposal_ratio = lambda proposed_values: \
                        update.log_proposal_ratio_double_trunc(self.true_mean_values,
                                                               proposed_values,
                                                               self.proposal_sd_true_mean, 
                                                               self.exposure_model_truncation['lower'],
                                                               self.exposure_model_truncation['upper']
                                                              )

                # init true mean values
                # values = dist.rvs(a=self.prior_parameters['alpha'].get_current_value(),
                                  # b=self.prior_parameters['beta'].get_current_value(), 
                                  # size=dimension_classical
                                 # )

                # values = exposure_model_truncation['lower'] + values*(exposure_model_truncation['upper']-exposure_model_truncation['lower'])

                # self.true_mean_values = values



            elif self.exposure_model_distribution == 'lognorm':
                dist = stats.lognorm
                # priors
                # priors get initialized here as FiexedParameter to be able to initialize start values of the uncertain factor
                # in case that the exposure model parameters should be estimated this FixedParameter will be overwritten with
                # a PriorParameter(Vector) by the MCMC class
                self.prior_parameters['mu'] = FixedParameter('mu', uncertainty_characteristics['exposure_model_parameters']['mu'])
                self.prior_parameters['sigma'] = FixedParameter('sigma', uncertainty_characteristics['exposure_model_parameters']['sigma'])
                
                # likelihood ratio
                self.log_LR_exposure_model = lambda proposed_values: \
                    np.sum(
                        dist.logpdf(proposed_values, s=self.prior_parameters['sigma'].get_current_value(), scale=np.exp(self.prior_parameters['mu'].get_current_value()))-\
                        dist.logpdf(self.true_mean_values, s=self.prior_parameters['sigma'].get_current_value(), scale=np.exp(self.prior_parameters['mu'].get_current_value()))
                        )
            
                # proposal distibution
                #### auskommentiert am 14.9. - versuch M4 zu reparieren
                # self.propose_values = lambda: update.proposal_trunc(self.true_mean_values, 
                                                                    # self.proposal_sd_true_mean
                                                                   # )
                self.propose_values = lambda: np.exp(np.random.normal(loc=0, scale=self.proposal_sd_true_mean, size=self.true_mean_values.shape)) * self.true_mean_values


                # proposal log ratio
                #### auskommentiert am 14.9. - versuch M4 zu reparieren
                # self.log_proposal_ratio = lambda proposed_values: \
                        # update.log_proposal_ratio_trunc(self.true_mean_values,
                                                        # proposed_values,
                                                        # self.proposal_sd_true_mean
                                                       # )
                self.log_proposal_ratio = lambda proposed_values: np.log(np.prod(proposed_values / self.true_mean_values))

                # init true mean values
                # self.true_mean_values = dist.rvs(s=self.prior_parameters['sigma'].get_current_value(),
                                                 # scale=np.exp(self.prior_parameters['mu'].get_current_value()), 
                                                 # size=dimension_classical
                                                # )

            elif self.exposure_model_distribution == 'norm':
                dist = stats.truncnorm
                # priors get initialized here as FiexedParameter to be able to initialize start values of the uncertain factor
                # in case that the exposure model parameters should be estimated this FixedParameter will be overwritten with
                # a PriorParameter(Vector) by the MCMC class
                self.prior_parameters['mu'] = FixedParameter('mu', uncertainty_characteristics['exposure_model_parameters']['mu'])
                self.prior_parameters['sigma'] = FixedParameter('sigma', uncertainty_characteristics['exposure_model_parameters']['sigma'])

                # likelihood ratio
                self.log_LR_exposure_model = lambda proposed_values: \
                    np.sum(
                        dist.logpdf(proposed_values, a=(0-self.prior_parameters['mu'].get_current_value())/self.prior_parameters['sigma'].get_current_value(), b=np.inf , loc=self.prior_parameters['mu'].get_current_value(), scale=self.prior_parameters['sigma'].get_current_value())-\
                        dist.logpdf(self.true_mean_values, a=(0-self.prior_parameters['mu'].get_current_value())/self.prior_parameters['sigma'].get_current_value(), b=np.inf, loc=self.prior_parameters['mu'].get_current_value(), scale=self.prior_parameters['sigma'].get_current_value())
                        )

                # proposal distibution
                self.propose_values = lambda: update.proposal_trunc(self.true_mean_values, 
                                                                    self.proposal_sd_true_mean
                                                                   )

                # proposal log ratio
                self.log_proposal_ratio = lambda proposed_values: \
                        update.log_proposal_ratio_trunc(self.true_mean_values,
                                                        proposed_values,
                                                        self.proposal_sd_true_mean
                                                       )
                # init true mean values
                # a_trunc = (0-self.prior_parameters['mu'].get_current_value())/self.prior_parameters['sigma'].get_current_value() 

                # self.true_mean_values = dist.rvs(a=a_trunc, b=np.inf, loc=self.prior_parameters['mu'].get_current_value(), 
                                                 # scale=self.prior_parameters['sigma'].get_current_value(), 
                                                 # size=dimension_classical
                                                # )


            if self.name in ['unshared_error', 'shared_error']:
                return None

            self.true_mean_values = self.observed_values.copy()
            self.true_mean_values = np.clip(self.true_mean_values,
                                            uncertainty_characteristics['exposure_model_truncation'].get('lower', -np.inf),
                                            uncertainty_characteristics['exposure_model_truncation'].get('upper', np.inf))


            #### initialize the classical errors in the error component
            dimension_classical = self.observed_values.size

            self.error_components['classical'] = ErrorComponent(structure_classical, dimension_classical, classical_sd)

            if structure_classical == 'multiplicative':
                classical_errors = self.observed_values / self.true_mean_values

            elif structure_classical == 'additive':
                classical_errors = self.observed_values - self.true_mean_values

            self.error_components['classical'].set_values(classical_errors)


            if self.name == 'phi':
                # identify positions where we have no 1
                self.selection_index_reference = data[self.exposed_miners].groupby(mapping_identifier_classical).apply(lambda df: df['activity'].unique()).values != 1
                true_mean_values = np.ones_like(self.selection_index_reference, dtype=float)
                true_mean_values[self.selection_index_reference] = self.true_mean_values
            elif self.name == 'b':
                self.selection_index_reference = data[self.exposed_miners].groupby(mapping_identifier_classical).apply(lambda df: df['b_reference'].unique()).values != 1
                true_mean_values = np.ones_like(self.selection_index_reference, dtype=float)
                true_mean_values[self.selection_index_reference] = self.true_mean_values

            else:
                true_mean_values = self.true_mean_values

            #### initialize mapping matrices and Berkson error
            if self.Berkson_error:
                mapping_identifier_Berkson = uncertainty_characteristics['mapping_identifier_Berkson']
                self.mapping_matrix_long = basics.create_mapping_matrix(mapping_identifier_Berkson, data, self.exposed_miners)  # from intermediate to long
                self.mapping_matrix_time_specific = basics.create_mapping_matrix_t_o(mapping_identifier_classical, mapping_identifier_Berkson, data, self.exposed_miners)  # from short to intermediate

                proposal_sd_Berkson = uncertainty_characteristics['Berkson_error']['proposal_sd']
                structure_Berkson = uncertainty_characteristics['Berkson_error']['structure']

                transfer_cluster = uncertainty_characteristics.get('selection_Berkson', None)
                if transfer_cluster is None:
                    self.Berkson_cluster = False
                    dimension_Berkson = self.mapping_matrix_time_specific.shape[0]
                else:
                    self.Berkson_cluster = True
                    transfer_reference = uncertainty_characteristics.get('reference_Berkson', None)
                    selection_Berkson = self.exposed_miners & (self.data[transfer_cluster] == 1) & (self.data[transfer_reference] != 1)
                    dimension_Berkson = np.concatenate(self.data[selection_Berkson].groupby(mapping_identifier_Berkson).apply(lambda df: df[name_obs_values].unique()).values).size

                self.error_components['Berkson'] = BerksonErrorComponent(structure_Berkson, dimension_Berkson, Berkson_sd, proposal_sd_Berkson)

                ##### THis is not tested
                true_mean_values_time_specific = self.mapping_matrix_time_specific * true_mean_values

                if self.Berkson_cluster:
                    selection_index_Berkson1 = self.data[self.exposed_miners].groupby(mapping_identifier_Berkson).apply(lambda df: df[transfer_cluster].unique()).values == 1
                    selection_index_Berkson2 = self.data[self.exposed_miners].groupby(mapping_identifier_Berkson).apply(lambda df: df[transfer_reference].unique()).values != 1
                    self.Berkson_not_fixed = (selection_index_Berkson1 & selection_index_Berkson2)  # selects all positions with an actual Berkson error (in a true cluster and not the reference objects)
                    Berkson_error = np.ones_like(selection_index_Berkson1, dtype=float)
                    Berkson_error[self.Berkson_not_fixed] = self.error_components['Berkson'].get_values()
                    true_values_time_specific = true_mean_values_time_specific * Berkson_error
                else:
                    true_values_time_specific = true_mean_values_time_specific * self.error_components['Berkson'].get_values()

                self.true_values_long = self.mapping_matrix_long * true_values_time_specific

                if self.name == 'tau_e':
                    self.true_values_long[self.data.tau_e_no_error == 1] = 1


            elif not self.Berkson_error:
                self.mapping_matrix_long = basics.create_mapping_matrix(mapping_identifier_classical, data, self.exposed_miners)
                self.mapping_matrix_time_specific = None
                self.true_values_long = self.mapping_matrix_long * true_mean_values


            self.samples = np.zeros((nb_iterations, self.true_mean_values.shape[0]))
            self.acceptance = np.empty(nb_iterations, bool)

        if self.Berkson_error and not self.classical_error and measurement_model not in ['ML1', 'ML2']:
            raise NotImplementedError("Pure Berkson error not implemented! \n You need to use the UnsharedError class for UnsharedBekson error")


    def get_values(self) -> np.ndarray:
        """
        Getter function

        :return: Current state of the factor
        """
        return self.true_values_long


    def get_yearly_values(self) -> dict:
        """
        Getter function for values sorted by year in dict

        :return: Current state of the factors as dict with years as keys
        """
        years = np.unique(self.exposure_years)
        yearly_values = {year: self.true_mean_values[self.exposure_years == year] for year in years}
        return yearly_values



    def adjust_values(self, accept: bool, candidate_values: np.ndarray,
                      candidate_values_long: np.ndarray, candidate_errors: dict) -> None:
        """
        This function is called by the LatentVariables class. It updates the current state of the chain depending on the arguments passed to this function.

        :param accept: A bool indiciating whether the proposed was accepted.
        :param candidate_values: An array with the cadidate values which will be set when accept is True
        :param candidate_values_long: An array with the long cadidate values which will be set when accept is True
        :param candidate_errors: An array with the cadidate errors which will be set when accept is True
        """
        if accept:
            self.true_mean_values = candidate_values
            self.true_values_long = candidate_values_long
            self.error_components['classical'].set_values(candidate_errors['classical'])

            if self.Berkson_error:
                self.error_components['Berkson'].set_values(candidate_errors['Berkson'])
            for prior_parameter in self.prior_parameters:

                if isinstance(self.prior_parameters[prior_parameter], PriorParameter):
                    self.prior_parameters[prior_parameter].set_uf_values(self.true_mean_values)

                elif isinstance(self.prior_parameters[prior_parameter], PriorParameterVector):
                    self.prior_parameters[prior_parameter].set_uf_values(self.get_yearly_values())

        self.samples[self.i + 1, :] = self.true_mean_values
        self.acceptance[self.i] = accept
        self.i += 1


    def update_prior_parameters(self) -> None:
        "Invokes an update of the prior parameters of the uncetain factor"
        if self.vectorized_exposure:
            
            current_prior_values = {key: self.prior_parameters[key].get_current_yearly_value() for key in self.prior_parameters}
            for prior_parameter in self.prior_parameters:
                self.prior_parameters[prior_parameter].update(current_prior_values)
                # current_prior_values[prior_parameter] = self.prior_parameters[prior_parameter].get_yearly_current_value()
                current_prior_values[prior_parameter] = self.prior_parameters[prior_parameter].get_current_yearly_value()


        else:
            current_prior_values = {key: self.prior_parameters[key].get_current_value() for key in self.prior_parameters}
            for prior_parameter in self.prior_parameters:
                self.prior_parameters[prior_parameter].update(current_prior_values)
                current_prior_values[prior_parameter] = self.prior_parameters[prior_parameter].get_current_value()






    def log_LR_measurement_model_classical_lognormal(self, true_mean_candidates, sigma):
        '''
            log_LR_measurement_model is the accelerated version of internal_vector and ratio_internal.
            It only contains the neccesary terms to evaluate the ratio between the current
            and the candidate value of the exposures according to the measurement
            process. See also the demonstration in the pdf file 'Accelerate
            evaluation internal process'
            to understand which terms are omitted.
            It is only adapted for classical error.
            '''
        log_Xt = np.log(self.true_mean_values)
        log_Xcand = np.log(true_mean_candidates)
        return( 1/(2*(sigma**2)) * np.sum(log_Xt**2 - log_Xcand**2 +
                                                         (2 * np.log(self.observed_values) +
                                                          (sigma**2))*(log_Xcand - log_Xt)))





    def adapt_proposal(self, nb_iterations, phase):
        """
        This function adjust the proposal sd of the error components.
        """
        acceptance_rate = np.mean(self.acceptance[phase*nb_iterations:(phase+1)*nb_iterations-1])
        diff = acceptance_rate - self.target
        change = abs(diff) > 0.02
        sign = np.sign(diff)
        self.proposal_sd_true_mean *= (1 + 0.1 * sign * change)

        if self.Berkson_error:
            self.error_components['Berkson'].adapt_proposal(sign, change)


        print(self.name)
        print("Acceptance rate: " + str(round(acceptance_rate, 4)))

        for prior_parameter in self.prior_parameters:
            self.prior_parameters[prior_parameter].adapt_proposal(nb_iterations, phase)



    def reset_values(self, iterations: int) -> None:
        """
        Resets the samples from the chain.
        """
        current_value = self.true_mean_values
        self.samples = np.zeros((iterations + 1, self.true_mean_values.shape[0]))
        self.samples[0, :] = current_value
        self.acceptance = np.empty(iterations + 1, bool)
        self.i = 0
        # reset prior parameters as well
        for prior_parameter in self.prior_parameters:
            self.prior_parameters[prior_parameter].reset_values(iterations)



    def write_samples(self, path_results: str, thin: int = 1):
        """
        Saves the chain to the path passed to the function.
        """
        file_name = path_results + 'results_UF_' + self.name + ".txt"
        index = np.arange(0, len(self.samples[:self.i + 1]), thin)
        np.savetxt(file_name, self.samples[:self.i + 1, :][index])

        
        # write prior_parameters
        for prior_parameter in self.prior_parameters:
            self.prior_parameters[prior_parameter].write_samples(path_results, thin)




class FixedUncertainFactor:
    """
    A pseudo class which can be used as an UncertainFactor by the LatentVariables class. This class can be used if one wants so specify a fixed value as UF.
    """
    def __init__(self, name, data, uncertainty_characteristics, measurement_model="M2"):
        self.name = name
        self.true_values_long = data[uncertainty_characteristics['name_true_values']].values
        self.true_values_long[data.Z == 0] = 0

        if measurement_model == "M1a":
            self.all_exposed_miners = data.model == measurement_model  # is ONLY used in calculate_exposure in the LV
            if self.name in ['C_Rn_old', 'b']:
                self.exposed_miners = data.C_Rn_old > 0
            elif self.name == 'C_Rn_ref':
                self.exposed_miners = data.C_Rn_obs_ref > 0
            elif self.name == 'tau_e':
                self.exposed_miners = (data.C_Rn_obs_ref > 0) & (data.tau_e_no_error != 1)
        else:
            self.exposed_miners = data.model == measurement_model


    def update(self):
        """
        Does nothing
        """
        pass

    def get_values(self):
        """
        Does nothing
        """
        return self.true_values_long

    def adapt_proposal(self, nb_iterations, phase):
        """
        Does nothing
        """
        pass

    def reset_values(self, iterations):
        """
        Does nothing
        """
        pass

    def write_samples(self, path, thin):
        """
        Does nothing
        """
        pass

