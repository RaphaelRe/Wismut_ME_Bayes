"""
This module implements the LatentVariables calss which holds and updates and the uncertain factors.
"""
import numpy as np
import numpy.ma as ma
from pandas import DataFrame
import scipy.stats as stats

import wismut.basics as basics  
import wismut.update as update
from wismut.UncertainFactor import UncertainFactor, FixedUncertainFactor
from wismut.UnsharedError import UnsharedError



class LatentVariables:
    '''
    This class is represents the latent exposure. It holds the uncertain factors. It aggregates them to represent the latent exposure.

    :param data: A pandas DataFrame,
    :param uncertainty_characteristics: dict of dicts - holds the infos for  each factor.
    :param nb_iterations: An integer which gets assed to the UncertainFactors
    :param disease_model: A string, currently only 'cox_like' is allowd
    :param s: A list containing the cutpoints for the baseline hazard.

    '''
    def __init__(self, data: DataFrame,
                 uncertainty_characteristics: dict,
                 nb_iterations: int,
                 disease_model: str,
                 s: np.ndarray = np.array([0, 40, 55, 75, 104])
                 ):

        # self.fix_latent_variable = fix_latent_variable
        self.data = data

        self.unexposed_miners = data.Z.values == 0

        # self.exposures_without_error = np.isin(data.model.values, np.array(["M0", "M0_NotYet"]))
        self.exposures_without_error = np.isin(data.model.values, np.array(["M0_NotYet"]))

        # if fix_latent_variable:
            # self.values = copy.deepcopy(data['X']).values
            # self.values_cum = copy.deepcopy(data['Xcum']).values
            # self.candidate_values = copy.deepcopy(data['X']).values
            # self.candidate_values_cum = copy.deepcopy(data['Xcum']).values
        # else:
        self.uncertain_factors = {}
        for measurement_model in uncertainty_characteristics:
            self.initialize_uncertain_factors(measurement_model, data, uncertainty_characteristics[measurement_model], nb_iterations)

        # self.values = np.zeros(data.shape[0]) # line required for initialization in the next line
        self.values = ma.zeros(data.shape[0])  # line required for initialization in the next line
        self.values.mask = self.unexposed_miners

        self.values = self.calculate_exposure()

        self.cumulation_matrix = basics.create_sparse_matrix(data) # lag=1 on default
        self.values_cum = self.cumulate_values(self.values)

        if 'ML1' in uncertainty_characteristics.keys() or 'ML2' in uncertainty_characteristics.keys(): 
            self.group_cumulation_matrix = basics.create_group_cumulation_matrices(data)
            self.group_likelihood_helper = GroupLatentVariable()  # THis calss is only a helper which holds the cumulated values for a specific group. ---> Likelihood assumes a long vector, therefore this makes evaluation faster
            self.log_LR_disease_model = lambda parameter_values, group: np.log(update.ratio_likelihood(disease_model,
                                               self.group_view[group], parameter_values, parameter_values, self.group_likelihood_helper, s))

        else:
            self.log_LR_disease_model = lambda parameter_values: np.log(update.ratio_likelihood(disease_model,
                                               data, parameter_values, parameter_values, self, s))
        # self.unexposed_miners = data.Z == 0
        self.case = data['event'].values == 1


        # self.log_LR_disease_model = lambda parameter_values: update.ratio_log_likelihood(disease_model,
                                # data, parameter_values, parameter_values, self, s)


    def get_values(self) -> np.ndarray:
        """
        Getter method.

        :return: Returns the current latent exposure.
        """
        return self.values

    def get_values_cum(self) -> np.ndarray:
        """
        Getter method.

        :return: Returns the cumulated current latent exposure.
        """
        return self.values_cum



    def initialize_uncertain_factors(self, measurement_model: str,
                                     data: DataFrame,
                                     uncertainty_characteristics: dict,
                                     nb_iterations: int) -> None:
        """
        This function is used in the initialization
        """
        self.uncertain_factors[measurement_model] = {}
        for parameter in uncertainty_characteristics:
            # print(parameter)

            if uncertainty_characteristics[parameter].get('fixed', False):
                self.uncertain_factors[measurement_model][parameter] = FixedUncertainFactor(parameter, data, uncertainty_characteristics[parameter], measurement_model)

            elif not uncertainty_characteristics[parameter].get('fixed', False):  # harakiri, UF should be either True or False
                classical_sd = uncertainty_characteristics[parameter]['classical_error']['sd']
                Berkson_sd = uncertainty_characteristics[parameter]['Berkson_error']['sd']
                if (classical_sd == 0) and (Berkson_sd == 0):
                    uncertainty_characteristics[parameter]['name_true_values'] = uncertainty_characteristics[parameter]['name_obs_values']
                    self.uncertain_factors[measurement_model][parameter] = FixedUncertainFactor(parameter, data, uncertainty_characteristics[parameter], measurement_model)
                elif parameter == "unshared_error":
                    self.initialize_selection_arrays(measurement_model)
                    self.uncertain_factors[measurement_model][parameter] = UnsharedError(parameter, data, uncertainty_characteristics[parameter], nb_iterations, measurement_model)
                else:
                    self.uncertain_factors[measurement_model][parameter] = UncertainFactor(parameter, data, uncertainty_characteristics[parameter], nb_iterations, measurement_model)


    def initialize_selection_arrays(self, measurement_model):
        """
        Only reuqired for Unshared errors! Not used for nomrmal UncertainFactors.
        """
        
        groups = self.data.group.unique()
        self.group_selection = {group: self.data.group == group for group in groups}  # returns an dict with the selection of each group

        self.group_view = {int(group): self.data[self.data['group'].values == group] for group in groups}
        self.group_update_selection = {group: (self.group_view[group]['model'].values == measurement_model) for group in groups} # info which values should be updated in each group

        # möglicher code falls wir doch noch mehrere periods bekommen (aus franz. cohort)
        # for period in self.group_periods[group]:
            # self.group_period_selection[group][period] =  (self.group_view[group]['period'].values == period)
        


    def get_uncertain_factor_values(self) -> dict:
        """
        Extracts the current values of the uncertain factors holded by itself.

        :return: A dictionary containg the values of the different uncertain factors.
        """
        return {measurement_model: {key: self.uncertain_factors[measurement_model][key].get_values() for key in self.uncertain_factors[measurement_model]} for measurement_model in self.uncertain_factors}


    def set_candidate_exposure(self, model: str, name_factor: str, candidate_values_long: np.ndarray) -> None:
        """
        For model M1a we can not directly set the new exposure in update_uncertain_factor() like in the other models. So we encapsulated it in an extra method.
        """
        self.candidate_values = np.copy(self.values)
        UF_values = self.get_uncertain_factor_values()
        UF_values[model][name_factor] = candidate_values_long

        if model == 'M1a':
            i = self.uncertain_factors["M1a"]['C_Rn_old'].all_exposed_miners
            i_old_mining = self.uncertain_factors["M1a"]['C_Rn_old'].exposed_miners
            i_active_mining = self.uncertain_factors["M1a"]['C_Rn_ref'].exposed_miners

            # make test to ensure corectness
            test1 = not np.all(UF_values['M1a']['tau_e'][self.data.tau_e_no_error == 1] == 1)
            test2 = not np.all(UF_values['M1a']['tau_e'][i & (~i_active_mining)] == 0)
            test3 = not np.all(UF_values['M1a']['C_Rn_ref'][i & (~i_active_mining)] == 0)
            test4 = not np.all(UF_values['M1a']['b'][i & (~i_old_mining)] == 0)
            test5 = not np.all(UF_values['M1a']['C_Rn_old'][i & (~i_old_mining)] == 0)

            if test1 or test2 or test3 or test4 or test5:
                raise ValueError("calculate exposure for M1a failed! One of the tests is False")

            self.candidate_values[i] = (UF_values['M1a']['C_Rn_old'][i] * UF_values['M1a']['b'][i] + UF_values['M1a']['r'][i] * \
                        (UF_values['M1a']['C_Rn_ref'][i] / UF_values['M1a']['A_ref'][i]) * \
                        (UF_values['M1a']['tau_e'][i] * UF_values['M1a']['A'][i])
                        ) * \
                        UF_values["working_time"]['omega'][i] * UF_values['equilibrium']['gamma'][i] * UF_values["activity"]['phi'][i] * \
                        self.data.prop.values[i] * self.data.tau.values[i] * 12 / 100
                        # self.data.prop.values[i] * self.data.tau.values[i] * 12

        elif model == 'M6':
            i = self.uncertain_factors["M6"]['C_Rn_ref_0'].exposed_miners
            self.candidate_values[i] = (UF_values['M6']['C_Rn_ref_0'][i] + \
                         (UF_values['M6']['C_Rn_ref_130'][i] - UF_values['M6']['C_Rn_ref_0'][i]) * UF_values['M6']['d'][i] / 130 * UF_values['M6']['epsilon'][i] * UF_values['M6']['epsilon2'][i]
                        ) * \
                        UF_values["working_time"]['omega'][i] * UF_values['equilibrium']['gamma'][i] * UF_values["activity"]['phi'][i] * \
                        self.data.prop.values[i] * self.data.tau.values[i] * 12 / (37 * 100)
                        # self.data.prop.values[i] * self.data.tau.values[i] * 12 / 37



    def calculate_exposure(self) -> np.ndarray:
        """
        Aggregates the different uncertain factors.

        :return: The resulting latent exposure
        """
        UF_values = self.get_uncertain_factor_values()
        values = np.zeros_like(self.values)  # # # Vorschlag Raphi: self.values ist nun ein masked array, somit gibt np.zeros_like auch ein masked array zurück im += werden alle operatoren überladen und UF-values korrekt gecastet...dynamic typing ist toll!
        for measurement_model in self.uncertain_factors:
            if measurement_model == "M1a":
                i = self.uncertain_factors["M1a"]['C_Rn_old'].all_exposed_miners
                i_old_mining = self.uncertain_factors["M1a"]['C_Rn_old'].exposed_miners
                i_active_mining = self.uncertain_factors["M1a"]['C_Rn_ref'].exposed_miners

                test1 = not np.all(UF_values['M1a']['tau_e'][self.data.tau_e_no_error == 1] == 1)
                test2 = not np.all(UF_values['M1a']['tau_e'][i & (~i_active_mining)] == 0)
                test3 = not np.all(UF_values['M1a']['C_Rn_ref'][i & (~i_active_mining)] == 0)
                test4 = not np.all(UF_values['M1a']['b'][i & (~i_old_mining)] == 0)
                test5 = not np.all(UF_values['M1a']['C_Rn_old'][i & (~i_old_mining)] == 0)
                
                if test1 or test2 or test3 or test4 or test5:
                    raise ValueError("calculate exposure for M1a failed! One of the tests is False")

                values[i] = (UF_values['M1a']['C_Rn_old'][i] * UF_values['M1a']['b'][i] + UF_values['M1a']['r'][i] * \
                            (UF_values['M1a']['C_Rn_ref'][i] / UF_values['M1a']['A_ref'][i]) * \
                            (UF_values['M1a']['tau_e'][i] * UF_values['M1a']['A'][i])
                            ) * \
                            UF_values["working_time"]['omega'][i] * UF_values['equilibrium']['gamma'][i] * UF_values["activity"]['phi'][i] * \
                            self.data.prop.values[i] * self.data.tau.values[i] * 12 / 100
                            # self.data.prop.values[i] * self.data.tau.values[i] * 12

            if measurement_model == "M1b":

                i = self.uncertain_factors["M1b"]['C_Rn'].exposed_miners

                values[i] = (UF_values['M1b']['C_Rn'][i] / UF_values['M1b']['??A_ref??'][i]) * \
                            UF_values['M1b']['???tau_e???'][i] * UF_values['M1b']['A'][i] * UF_values['M1b']['tau_E'][i] * \
                            UF_values["working_time"]['omega'][i] * UF_values['equilibrium']['gamma'][i] * UF_values["activity"]['phi'][i] * \
                            self.data.prop.values[i] * self.data.tau.values[i] * 12 / 100
                            # self.data.prop.values[i] * self.data.tau.values[i] * 12



            if measurement_model == "M2":
                i = self.uncertain_factors["M2"]['C_Rn'].exposed_miners  # take gamma since it only present in M2 to select all rows for M2
                values[i] += UF_values["working_time"]['omega'][i] * UF_values['equilibrium']['gamma'][i] * UF_values[measurement_model]['C_Rn'][i] * UF_values["activity"]['phi'][i] * self.data.prop.values[i] * self.data.tau.values[i] * 12 / 100
                # values[i] += UF_values["working_time"]['omega'][i] * UF_values['equilibrium']['gamma'][i] * UF_values[measurement_model]['C_Rn'][i] * UF_values["activity"]['phi'][i] * self.data.prop.values[i] * self.data.tau.values[i] * 12

            if measurement_model == "M2_Expert":
                i = self.uncertain_factors[measurement_model]['C_Exp'].exposed_miners  # take gamma since it only present in M2 to select all rows for M2
                # values[i] += UF_values[measurement_model]['omega'][i] * UF_values[measurement_model]['gamma'][i] * UF_values[measurement_model]['C_Rn'][i] * UF_values[measurement_model]['phi'][i] * self.data.prop.values[i] * 12 / 100
                values[i] += UF_values["working_time"]['omega'][i] * UF_values['equilibrium']['gamma'][i] * UF_values[measurement_model]['C_Exp'][i] * UF_values["activity"]['phi'][i] * self.data.prop.values[i] * self.data.tau.values[i] * 12 / 100
                # values[i] += UF_values["working_time"]['omega'][i] * UF_values['equilibrium']['gamma'][i] * UF_values[measurement_model]['C_Exp'][i] * UF_values["activity"]['phi'][i] * self.data.prop.values[i] * self.data.tau.values[i] * 12

            if measurement_model == "M3":
                i = self.uncertain_factors["M3"]['zeta'].exposed_miners  # take zetasince it only present in M2 to select all rows for M2
                # values[i] += UF_values[measurement_model]['omega'][i] * UF_values[measurement_model]['zeta'][i] * UF_values[measurement_model]['C_Rn'][i] * UF_values[measurement_model]['phi'][i] * self.data.prop.values[i] * 12 / 100
                values[i] += UF_values["working_time"]['omega'][i] * UF_values[measurement_model]['zeta'][i] * UF_values[measurement_model]['C_RPD'][i] * UF_values["activity"]['phi'][i] * self.data.prop.values[i] * self.data.tau.values[i] * 12 / 100
                # values[i] += UF_values["working_time"]['omega'][i] * UF_values[measurement_model]['zeta'][i] * UF_values[measurement_model]['C_RPD'][i] * UF_values["activity"]['phi'][i] * self.data.prop.values[i] * self.data.tau.values[i] * 12

            if measurement_model in ["M4", "M4_Exploration", "M1a_Expert_WLM", "M2_Expert_WLM", "M3_Expert_WLM", "M6_Expert_WLM"]:
                i = self.uncertain_factors[measurement_model]['E_Rn'].exposed_miners  # take E_Rn since it only present in M2 to select all rows for M2
                values[i] += UF_values[measurement_model]['E_Rn'][i] * UF_values["activity"]['phi'][i] * self.data.prop.values[i] * self.data.tau.values[i] / 100
                # values[i] += UF_values[measurement_model]['E_Rn'][i] * UF_values["activity"]['phi'][i] * self.data.prop.values[i] * self.data.tau.values[i]

            if measurement_model == "M6":
                i = self.uncertain_factors[measurement_model]['C_Rn_ref_0'].exposed_miners

                values[i] = (UF_values[measurement_model]['C_Rn_ref_0'][i] + \
                             (UF_values[measurement_model]['C_Rn_ref_130'][i] - UF_values[measurement_model]['C_Rn_ref_0'][i]) * UF_values[measurement_model]['d'][i] / 130 * UF_values[measurement_model]['epsilon'][i] * UF_values[measurement_model]['epsilon2'][i]
                            ) * \
                            UF_values["working_time"]['omega'][i] * UF_values['equilibrium']['gamma'][i] * UF_values["activity"]['phi'][i] * \
                            self.data.prop.values[i] * self.data.tau.values[i] * 12 / (37 * 100)
                            # self.data.prop.values[i] * self.data.tau.values[i] * 12 / 37

            if measurement_model in ["ML1", "ML2"]:
                # Achtung! UF values sind hier die error werte. In allen anderen Fällen sind es tatsächliche UF werte
                errors = UF_values[measurement_model]['unshared_error']
                for group in errors.keys():
                    Z_group = self.group_view[group].Z.values

                    if group in errors.keys():
                        group_selection = self.group_selection[group]
                        X_group = values[group_selection]
                        U_group = errors[group]

                        if self.uncertain_factors[measurement_model]['unshared_error'].Berkson_error:
                            X_group[self.group_update_selection[group]] = Z_group[self.group_update_selection[group]] * U_group
                        elif self.uncertain_factors[measurement_model]['unshared_error'].classical_error:
                            X_group[self.group_update_selection[group]] = Z_group[self.group_update_selection[group]] / U_group
                    else:
                        X_group = Z_group
                    values[group_selection] = X_group


            if measurement_model in ["working_time", "activity", "equilibrium"]:
                pass

        values[self.exposures_without_error] = self.data.Z.values[self.exposures_without_error] * self.data.prop.values[self.exposures_without_error] / 100
        # values[self.exposures_without_error] = self.data.Z.values[self.exposures_without_error] * self.data.prop.values[self.exposures_without_error]

        return values



    def cumulate_values(self, values: np.ndarray) -> np.ndarray:
        """
        Cumulates the latent exposure

        :retun: A numpy array with the cumulated latent exposure
        """
        return self.cumulation_matrix * values  # values is masked, but gets casted to normal array with sparse matrix




    def update(self, parameter_values: dict) -> None:
        """
        Invokes an update of all uncetain factors hold by the class.
        """
        for measurement_model in self.uncertain_factors:
            for factor in self.uncertain_factors[measurement_model]:
                self.update_uncertain_factor(measurement_model, factor, parameter_values)



    def adapt_proposal(self, nb_iterations: int, phase: int) -> None:
        """
        Adapts the proposal sd of all uncertain factors hold by the class.
        """
        for measurement_model in self.uncertain_factors:
            for factor in self.uncertain_factors[measurement_model]:
                self.uncertain_factors[measurement_model][factor].adapt_proposal(nb_iterations, phase)


    def write_samples(self, path: str, thin: int = 1) -> None:
        """
        Writing the samples of all uncertain_factors.

        :param path: A string defining the path where the values should be written
        :param thin: Only the n-th value will be written where n = thin
        """
        for measurement_model in self.uncertain_factors:
            for factor in self.uncertain_factors[measurement_model]:
                self.uncertain_factors[measurement_model][factor].write_samples(path, thin)


    def update_uncertain_factor(self, measurement_model: str, factor_name: str,
                                parameter_values: dict) -> None:
        '''
        Updates the uncertain factor defined by "factor_name". Update assumes always classical error and checks for Berkson

        :param measurement_model: A string specifying the measurement model. E.g. 'M2' for the second model
        :param factor_name: A string specifying the name of the factor to be updated. E.g. 'omega'
        :param parameter_values: A dictionary with parameter values, i.e. the  values of lambda etc. which will be used to make the update
        '''
        factor = self.uncertain_factors[measurement_model][factor_name]

        if isinstance(factor, UnsharedError):
            self.update_unshared_error(measurement_model, factor_name, parameter_values)
        else:
            if isinstance(factor, FixedUncertainFactor):
                return None

            candidate_errors = {}  #  is used in factor.adjust_values

            true_mean_candidates = factor.propose_values()

            if factor.error_components['classical'].structure == 'multiplicative':

                classical_candidate_errors = factor.observed_values / true_mean_candidates  ## nicht mehr nötig
                classical_log_measurement_ratio = factor.log_LR_measurement_model(true_mean_candidates) 

            elif factor.error_components['classical'].structure == 'additive':
                classical_candidate_errors = factor.observed_values - true_mean_candidates
                classical_log_measurement_ratio = factor.error_components['classical'].log_LR_measurement_model(classical_candidate_errors)

            classical_log_proposal_ratio = factor.log_proposal_ratio(true_mean_candidates)

            candidate_errors['classical'] = classical_candidate_errors

            log_exposure_ratio = factor.log_LR_exposure_model(true_mean_candidates)


            if factor.name == 'phi':
                true_mean_cand_complete = np.ones_like(factor.selection_index_reference, dtype=np.float64)
                true_mean_cand_complete[factor.selection_index_reference] = true_mean_candidates
            elif factor.name == 'b':
                true_mean_cand_complete = np.ones_like(factor.selection_index_reference, dtype=np.float64)
                true_mean_cand_complete[factor.selection_index_reference] = true_mean_candidates
            else:
                true_mean_cand_complete = true_mean_candidates

            if not factor.Berkson_error:
                true_values_long_candidate = factor.mapping_matrix_long * true_mean_cand_complete
                Berkson_log_measurement_ratio = 0
                Berkson_log_proposal_ratio = 0

            else:
                true_mean_cand_t_o = factor.mapping_matrix_time_specific * true_mean_cand_complete
                Berkson_candidate_errors = factor.error_components['Berkson'].propose_values()

                if factor.Berkson_cluster:
                    Berkson_errors = np.ones_like(factor.Berkson_not_fixed, dtype=float)
                    Berkson_errors[factor.Berkson_not_fixed] = Berkson_candidate_errors
                    true_values_cand_t_o = true_mean_cand_t_o * Berkson_errors
                else:
                    true_values_cand_t_o = true_mean_cand_t_o * Berkson_candidate_errors

                true_values_long_candidate = factor.mapping_matrix_long * true_values_cand_t_o

                Berkson_log_measurement_ratio = factor.error_components['Berkson'].log_LR_measurement_model(Berkson_candidate_errors)
                Berkson_log_proposal_ratio = factor.error_components['Berkson'].log_proposal_ratio(Berkson_candidate_errors)

                candidate_errors['Berkson'] = Berkson_candidate_errors

            if factor.name == 'tau_e':
                true_values_long_candidate[self.data.tau_e_no_error == 1] = 1


            if factor.measurement_model in ['M1a', 'M6']:
                self.set_candidate_exposure(factor.measurement_model, factor.name, true_values_long_candidate)
            else:
                self.candidate_values = np.copy(self.values)
                i = factor.exposed_miners
                self.candidate_values[i] = self.values[i] * true_values_long_candidate[i] / factor.true_values_long[i]


            self.candidate_values_cum = self.cumulate_values(self.candidate_values)
            log_disease_ratio = self.log_LR_disease_model(parameter_values)



            ratio = np.exp(log_disease_ratio +
                           log_exposure_ratio +
                           classical_log_measurement_ratio +
                           classical_log_proposal_ratio +
                           Berkson_log_measurement_ratio +
                           Berkson_log_proposal_ratio
                           )

            # print(factor.i)
            # print("log_disease_ratio: " + str(np.exp(log_disease_ratio)))
            # print("log_exposure_ratio: " + str(np.exp(log_exposure_ratio)))
            # print("classical_log_measurement_ratio: " + str(np.exp(classical_log_measurement_ratio)))
            # print("classical_log_proposal_ratio: " + str(np.exp(classical_log_proposal_ratio)))
            # print("Berkson_log_measurement_ratio: " + str(np.exp(Berkson_log_measurement_ratio)))
            # print("Berkson_log_proposal_ratio: " + str(np.exp(Berkson_log_proposal_ratio)))

            accept = (np.random.uniform(0, 1) < ratio)
            # print(accept)
            factor.adjust_values(accept, true_mean_candidates, true_values_long_candidate, candidate_errors)
            factor.update_prior_parameters()

            if accept:
                self.values = self.candidate_values
                self.values_cum = self.candidate_values_cum



    def update_unshared_error(self, measurement_model, factor_name, parameter_values):
        error = self.uncertain_factors[measurement_model][factor_name]
        for group in error.groups:
            self.update_unshared_error_group(group, parameter_values, error)


    def update_unshared_error_group(self, group: int, parameter_values: dict, error: UnsharedError):  # error ist ein pointer auf den error der upgedated werden soll (siehe update_unshared_error)
        group_selection = self.group_selection[group]
        group_update_selection = self.group_update_selection[group]

        Xt = self.values[group_selection]
        Xcand =  np.array(Xt, copy = True)
        
        if error.Berkson_error:
            if error.structure == "multiplicative":
                Ucand = error.propose_values(group)
                Xcand[group_update_selection] = Ucand  * error.observed_exposure[group]
            if error.structure == "additive":
                raise NotImplementedError("Additive Error not implemented yet")
                # idea: Xcand[group_update_selection] = Ucand  + error.observed_exposure[group]
                
            classical_log_measurement_ratio = 0
            classical_log_proposal_ratio = 0 
            log_exposure_ratio = 0
            
            Berkson_log_measurement_ratio = error.log_LR_measurement_model_Berkson(Ucand, group)
            Berkson_log_proposal_ratio = error.log_proposal_ratio_Berkson(Ucand, group)

        elif error.classical_error:
            if error.structure == "multiplicative":
                Xcand[group_update_selection] = error.propose_values_classical(Xt[group_update_selection], group)
                Ucand = error.observed_exposure[group].values / Xcand[group_update_selection]
            if error.structure == "additive":
                raise NotImplementedError("Additive Error not implemented yet")
                # idea: Xcand[group_update_selection] = error.observed_exposure[group] - Ucand

            Berkson_log_measurement_ratio = 0
            Berkson_log_proposal_ratio = 0

            log_exposure_ratio = error.log_LR_exposure_model(Xt[group_update_selection], Xcand[group_update_selection])
            classical_log_measurement_ratio = error.log_LR_measurement_model_classical_lognormal(Xt[group_update_selection], Xcand[group_update_selection], group)
            classical_log_proposal_ratio = np.log(np.prod(Xcand[group_update_selection] / Xt[group_update_selection]))

            # print(f"log_exposure_ratio: {np.exp(log_exposure_ratio)}")
            # print(f"classical_log_measurement_ratio: {np.exp(classical_log_measurement_ratio)}")
            # print(f"classical_log_proposal_ratio: {np.exp(classical_log_proposal_ratio)}")

        Xcum_cand = self.group_cumulation_matrix[group] * Xcand 
        Xcum_t = self.group_cumulation_matrix[group] * Xt


        # ratios bauen und evaluieren
        self.group_likelihood_helper.candidate_values_cum = Xcum_cand
        self.group_likelihood_helper.values_cum = Xcum_t
        self.group_likelihood_helper.case = self.group_view[group]['event'].values == 1

        log_disease_ratio = self.log_LR_disease_model(parameter_values, group)

        ratio = np.exp(log_disease_ratio +
                       log_exposure_ratio +
                       classical_log_measurement_ratio +
                       classical_log_proposal_ratio +
                       Berkson_log_measurement_ratio +
                       Berkson_log_proposal_ratio
                       )
        # print(ratio)


        accept = (np.random.uniform(0,1)<ratio)
        # print(accept)

        if accept:
            # set values of the lv, UnsahredError
            self.values[group_selection] = Xcand
            self.values_cum[group_selection] = Xcum_cand
            error.error_values[group] = Ucand

            error.group_acceptance[group] += 1
        error.group_count[group] += 1



    def reset_values(self, iterations: int) -> None:
        """
        Function to reset the values of the chain. Deletes the old and initializes a new one.

        :param interations: int defining on how many array should be initialized
        """
        for measurement_model in self.uncertain_factors:
            for factor in self.uncertain_factors[measurement_model]:
                self.uncertain_factors[measurement_model][factor].reset_values(iterations)




class GroupLatentVariable:
    """
    This is only a helper to save calculation time for Unshared error
    """
    def _init__(self):
        self.group = -999
        self.values_cum = 'not initialized'
        self.candidate_values_cum = 'not initialized'
        self.case = 'not initialized'


