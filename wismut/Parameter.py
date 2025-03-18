import numpy as np
import scipy.stats as stats
import copy
from pandas import DataFrame

import wismut.update as update


class Parameter:
    """
    The Parameter class represents the parameters of interest. Therefore the an instance can be one of [beta, lambda1, lambda2, lambda3, lambda4]

    :param name: A string defining the name of the parameter
    :param start_values: A dictionary containing as key the name of the parameter and as value the start value as float
    :param data: A pandas DataFrame containing the data
    :param precision: A dictionary with the name as key and as value a integer of digits to be printed.
    :param proposal_sd: A dictionary with the name as key and as value a float difining the initaial proposal sd.
    :param prior_parameters: A dict containing the as keys the hyper priors and as values the value of them.
    :param calculate_proposal_ratio: Boolean defining whether this ratio should be calculated.
    :param s: A list containing the cutpoints for the baseline hazard.
    :param disease_model: A string, currently only 'cox_like' possible.
    :param informative_priors: A boolean defining whethe rinformative priors  shoud be used.
    """

    def __init__(self, name: str, start_values: dict, data: DataFrame,
                 precision: dict, proposal_sd: dict, prior_parameters: dict,
                 calculate_proposal_ratio: bool, s: np.ndarray,
                 disease_model: str, informative_priors: bool = False):

        self.name = name
        self.target = 0.4  # target = acceptance rate
        self.precision = precision[name]  # on how many digits to round in output
        self.samples = np.empty(30000)  # only big enough for the adaptive phase, it will be reinitialized after the burnin phase
        self.samples[0] = start_values[name]  # realisation des k-ten parameters
        self.i = 0  # iteration step
        self.acceptance = np.empty(30000, dtype=bool)  # # only big enough for the adaptive phase, it will be reinitialized after the burnin phase
        self.proposal_sd = proposal_sd[name]
        self.calculate_proposal_ratio = calculate_proposal_ratio

        self.prior_ratio = lambda theta_cand: update.priors[prior_parameters[name]["dist"]](self.samples[self.i],
                                    theta_cand, prior_parameters[name], informative_priors)

        self.propose_value = lambda: update.proposals[name](self.samples[self.i], self.proposal_sd)

        if disease_model == 'exposure':
            self.likelihood_ratio = lambda values_t, values_cand, cumulated_exposure: latent_variables.ratio_exposure_model[name](values_t, values_cand)

        else:
            if ((disease_model == 'cox_like') & (name == 'beta')):
                self.likelihood_ratio = lambda values_t, values_cand, cumulated_exposure: update.ratio_likelihood(disease_model,
                                        data, values_t, values_cand, cumulated_exposure, s, simplify='Xbeta')
            elif 'lambda' in name:
                self.likelihood_ratio = lambda values_t, values_cand, cumulated_exposure: update.ratio_likelihood(disease_model,
                                        data, values_t, values_cand, cumulated_exposure, s, simplify='lambda')
            else:
                self.likelihood_ratio = lambda values_t, values_cand, cumulated_exposure: update.ratio_likelihood(disease_model,
                                        data, values_t, values_cand, cumulated_exposure, s)

        self.update = lambda current_values, exposures: update.parameter_update(self, current_values, exposures)



    def adapt_proposal(self, nb_iterations: int, phase: int) -> None:
        """
        Function to adapt the proposal_sd.

        :param nb_iterations: Integer defining the start of the sequence of  interest, the acceptance rate is getting calculated on.
        :param phase: Integer specifying the current phase

        """
        acceptance_rate = np.mean(self.acceptance[phase * nb_iterations:(phase + 1) * nb_iterations - 1])
        diff = acceptance_rate - self.target
        change = abs(diff) > 0.02
        sign = np.sign(diff)
        self.proposal_sd *= (1 + 0.1 * sign * change)
        print(self.name)
        print("Acceptance rate: " + str(round(acceptance_rate, 4)))

    def write_samples(self, path_results: str, thin: int = 1) -> None:
        """
        Writes the sampled chain
        """
        file_name = path_results + 'results_' + self.name + ".txt"
        index = np.arange(0, len(self.samples[:self.i + 1]), thin)
        np.savetxt(file_name, self.samples[:self.i + 1][index])

    def get_current_value(self):
        """
        Getter method of the class

        :return: Returns the curretn state of the parameter
        """
        current_value = self.samples[self.i]
        return(current_value)


    def proposal_ratio(self, theta_cand: float, name_parent: str) -> None:
        """
        Function to calculate the proposal ratio of the parameter.

        :return: Returns the ratio for the submitted state
        """
        theta_curr = self.get_current_value()
        proposal_sd = self.proposal_sd

        a_trunc_curr = (0 - theta_cand) / proposal_sd
        a_trunc_cand = (0 - theta_curr) / proposal_sd
        if update.proposals[name_parent].__name__ == 'proposal_trunc':
            ratio = stats.truncnorm.pdf(theta_curr, a=a_trunc_curr, b=np.inf, loc=theta_cand, scale=proposal_sd) / \
                    stats.truncnorm.pdf(theta_cand, a=a_trunc_cand, b=np.inf, loc=theta_curr, scale=proposal_sd)
        elif update.proposals[name_parent].__name__ == 'proposal_double_trunc':
            b_trunc_curr = (1 - theta_cand) / proposal_sd
            b_trunc_cand = (1 - theta_curr) / proposal_sd
            ratio = stats.truncnorm.pdf(theta_curr, a=a_trunc_curr, b=b_trunc_curr, loc=theta_cand, scale=proposal_sd) / \
                    stats.truncnorm.pdf(theta_cand, a=a_trunc_cand, b=b_trunc_cand, loc=theta_curr, scale=proposal_sd)
        return(ratio)


    def reset_values(self, iterations: int) -> None:
        """
        Resets the values of the chain.

        :param iterations: An integer defining on how many value sthe new array should be re-initialized.
        """
        current_value = self.get_current_value()
        self.samples = np.empty(iterations + 2)
        self.samples[0] = current_value
        self.acceptance = np.empty(iterations + 2, dtype=bool)
        self.i = 0


    def get_statistics(self) -> dict:
        """
        Returns current summary statistics of the chain.
        """
        statistics = {}
        i = self.i
        statistics['median'] = round(np.median(self.samples[:i]), self.precision)
        statistics['mean'] = round(np.mean(self.samples[:i]), self.precision)
        statistics['IC_2.5'] = round(np.percentile(self.samples[:i], 2.5), self.precision)
        statistics['IC_97.5'] = round(np.percentile(self.samples[:i], 97.5), self.precision)
        statistics['acceptance'] = round(np.mean(self.acceptance[self.i]), 4)
        return(statistics)




    def update_priors(self, current_prior_values: dict) -> None:
        """
        Writes the values from the current_prior_values to the current values of Parameters prior parameters

        :param current_prior_values: A dict with current prior values.
        """
        for prior_parm in self.prior_parameters:
            self.prior_parameters[prior_parm] = current_prior_values[prior_parm]



class FixedParameter:
    """
    The FixedParameter class is used to represent fixed vlues as Parameter. It implements basic methods allowing the MCMC class to treat them as standard parameters (Without changing their value). It can also be used as a fixed prior parameter
    """
    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value
        self.i = 0
        self.proposal_sd = 0

    def adapt_proposal(self, nb_iterations, phase):
        """
        Does nothing.
        """
        pass

    # def update(self, current_values, latent_variable):
    def update(self, *args):
        """
        Does nothing.
        """
        pass

    def update_priors(self, current_prior_values):
        """
        Does nothing.
        """
        pass

    def write_samples(self, path_results, thin):
        """
        Does nothing.
        """
        pass

    def reset_values(self, iterations):
        """
        Does nothing.
        """
        pass

    def get_current_value(self):
        """
        Getter method.

        :return: Value of the Parameter
        """
        current_value = self.value
        return(current_value)

    def get_statistics(self):
        """
        Passes that this parameter is fixed
        """
        statistics = {}
        statistics['median'] = 'fixed'
        statistics['mean'] = 'fixed'
        statistics['IC_2.5'] = 'fixed'
        statistics['IC_97.5'] = 'fixed'
        statistics['acceptance'] = 'fixed'
        return(statistics)



class PriorParameter(Parameter):
    """
    Inherits from Parameter. A PriorParameter is used for the measurements of radon gas etc.. i.e. C_Rn
    For initial arguments see Parameter documentation.
    """

    def __init__(self, name, start_values, precision, proposal_sd,
                 prior_parameters, uncertain_factor, calculate_proposal_ratio,
                 measurement_model, exposure_model_distribution, year="", exposure_model_truncation=None):

        prior_start_values = copy.deepcopy(start_values)
        prior_start_values[name] = start_values[uncertain_factor][name]
        proposal_sd[name] = proposal_sd[uncertain_factor + '_' + name]
        data = None
        s = None
        disease_model = 'harakiriiiii'
        precision = {name: precision}

        super().__init__(name, prior_start_values, data, precision, proposal_sd,
                         prior_parameters, calculate_proposal_ratio, s,
                         disease_model, informative_priors=False)


        self.name_parent = uncertain_factor + '_' + name
        self.year = year  # vor allem für prior parameter vector relevant

        self.measurement_model = measurement_model
        self.prior_parameters = prior_parameters[self.name_parent]

        self.prior_ratio = lambda theta_cand: update.priors[self.prior_parameters["dist"]](self.samples[self.i], theta_cand, self.prior_parameters, informative_priors=True)

        if exposure_model_distribution == "beta":
            self.likelihood_ratio = lambda current_prior_values, candidate_prior_values: update.prior_likelihoods[self.name_parent][exposure_model_distribution](self.uf_values, current_prior_values, candidate_prior_values, exposure_model_truncation=exposure_model_truncation)
        else:
            self.likelihood_ratio = lambda current_prior_values, candidate_prior_values: update.prior_likelihoods[self.name_parent][exposure_model_distribution](self.uf_values, current_prior_values, candidate_prior_values)

        self.update = lambda current_prior_values: update.prior_update(self, current_prior_values)

        self.propose_value = lambda: update.proposals[self.name_parent](self.samples[self.i], self.proposal_sd)

        self.calculate_proposal_ratio = update.proposals[self.name_parent].__name__ in ['proposal_trunc', 'proposal_double_trunc']



    def get_proposal_sd(self) -> None:
        return self.proposal_sd



    def set_state(self, path: str, chain: str, thin: int) -> None:
        """
        Basically the same funct as for Parameter. However, here we need a custom
        method since self.name is just the name. To load it we also need the
        self.name_parent.
        For detials see doc of this method for parameter.
        """
        name_parameter = self.name_parent + '_' + self.name

        x = np.loadtxt(f'{path}/{chain}_results_{name_parameter}.txt')
        # recycle values thin times since we only have the thinned values (exception is the last one)
        x_long = np.append(np.repeat(x[:-1], thin), x[-1])
        self.i = x_long.shape[0] - 1
        self.samples[:self.i + 1] = x_long


    def set_uf_values(self, values: np.ndarray) -> None:
        """
        Is used in PriorParameterVector
        """
        self.uf_values = values


    def write_samples(self, path_results: str, thin: int = 1) -> None:
        "See Parameter documentation"
        file_name = path_results + 'results_' + self.measurement_model + '_' + self.name_parent + "_" + str(self.year) + '.txt'
        index = np.arange(0, len(self.samples[:self.i + 1]), thin)
        np.savetxt(file_name, self.samples[:self.i + 1][index])


    def adapt_proposal(self, nb_iterations: int, phase: int) -> None:
        """
        Function to adapt the proposal_sd.

        :param nb_iterations: Integer defining the start of the sequence of  interest, the acceptance rate is getting calculated on.
        :param phase: Integer specifying the current phase

        """
        acceptance_rate = np.mean(self.acceptance[phase * nb_iterations:(phase + 1) * nb_iterations - 1])
        diff = acceptance_rate - self.target
        change = abs(diff) > 0.02
        sign = np.sign(diff)
        self.proposal_sd *= (1 + 0.1 * sign * change)
        print(f"    {self.name}_{self.year}")
        print("     Acceptance rate: " + str(round(acceptance_rate, 4)))




class PriorParameterVector:
    """
    A wrapper around PriorParameter. When we want to have an exposure model for each year, we use this class as a wrapper. It behaves like a PriorParameter from outside. In general. all function calls from this class just iterates over its PriorParameters held in a dict.
    """

    def __init__(self, name_parent, start_values, exposure_years, precision, proposal_sd,
                 prior_parameters, uncertain_factor, calculate_proposal_ratio,
                 measurement_model, exposure_model_distribution):
        self.name_parent = name_parent
        self.parameters = {}
        self.exposure_years = exposure_years

        parameter_names = np.unique(exposure_years)
        for param in parameter_names:
            self.parameters[param] = PriorParameter(name=name_parent,
                                                    start_values=start_values,
                                                    precision=precision,
                                                    proposal_sd=proposal_sd,
                                                    prior_parameters=prior_parameters,
                                                    uncertain_factor=uncertain_factor,
                                                    calculate_proposal_ratio=True,
                                                    measurement_model=measurement_model,
                                                    exposure_model_distribution=exposure_model_distribution,
                                                    year=param)



    def update(self, current_values: dict) -> None:
        """
        Overloaded Method from PriorParameter: Calls iteratively the update from PriorParameter
        """
        for year in self.parameters.keys():
            current_yearly_values = {key: current_values[key][year] for key in current_values}
            self.parameters[year].update(current_yearly_values)



    def set_uf_values(self, values: dict) -> None:
        """
        Overloaded Method from PriorParameter: Calls iteratively the update from PriorParameter
        """
        for year in self.parameters:
            self.parameters[year].set_uf_values(values[year])



    def adapt_proposal(self, nb_iterations: int, phase: int) -> None:
        """
        Overloaded Method from PriorParameter: Calls iteratively the update from PriorParameter
        """
        for p in self.parameters.keys():
            self.parameters[p].adapt_proposal(nb_iterations, phase)



    def write_samples(self, path_results: str, thin: int = 1) -> None:
        """
        Overloaded Method from PriorParameter: Calls iteratively the update from PriorParameter
        """
        for parameter in self.parameters.keys():
            self.parameters[parameter].write_samples(path_results=path_results, thin=thin)



    def reset_values(self, iterations: int) -> None:
        """
        Overloaded Method from PriorParameter: Calls iteratively the update from PriorParameter
        """
        for parameter in self.parameters.keys():
            self.parameters[parameter].reset_values(iterations)


    def get_current_value(self) -> np.ndarray:
        """
        Overloaded Method from PriorParameter: Calls iteratively the update from PriorParameter
        """
        current_values = np.array([self.parameters[year].get_current_value() for year in self.exposure_years])
        return(current_values)


    def get_current_yearly_value(self) -> np.ndarray:
        """
        Overloaded Method from PriorParameter: Calls iteratively the update from PriorParameter
        """
        current_yearly_values = {parm: self.parameters[parm].get_current_value() for parm in self.parameters.keys()}
        return(current_yearly_values)


    def get_statistics(self) -> None:
        """
        Overloaded Method from PriorParameter: Calls iteratively the update from PriorParameter
        """
        statistics = {}
        for parameter in self.parameters.keys():
            statistics[parameter] = self.parameters[parameter].get_statistics()
