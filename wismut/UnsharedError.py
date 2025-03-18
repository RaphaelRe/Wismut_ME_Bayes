import numpy as np
import pandas as pd
import scipy.stats as stats

import wismut.basics as basics
from wismut.UncertainFactor import UncertainFactor


class UnsharedError(UncertainFactor):
    '''
    This class is a class

    :param data: data
    ....

    '''
    def __init__(self, name: str, data: pd.DataFrame, uncertainty_characteristics: dict,
                 nb_iterations: int = 1000, measurement_model: str = "M99") -> None:

        super().__init__(name, data, uncertainty_characteristics,
                 nb_iterations, measurement_model)
    # sollte von UF erben und sich auch so verhalten!!!!!!
    

    # weiß der UnsharedError ob er eine Berkson oder clasicla ist? Oder hat er subkomponenten, die das eben sind (schlechte ideee)
    # benutzt er ErrorComponents? (auch eher schlechte Idee?)




    ##### Aus franz. cohort
                      
        # self.period = period                                                                                      
        # period_data =   data[data['period'].values == period]      

        self.groups = np.unique(data[data['Z']>0]['group']).astype(int)
        self.group_view = {int(group): data[data['group'].values == group] for group in self.groups}

        if self.classical_error:
            error_sd = uncertainty_characteristics['classical_error']['sd']
            proposal_sd = uncertainty_characteristics['classical_error']['proposal_sd']
            structure = uncertainty_characteristics['classical_error']['structure']
        elif self.Berkson_error:
            error_sd = uncertainty_characteristics['Berkson_error']['sd']
            proposal_sd = uncertainty_characteristics['Berkson_error']['proposal_sd']
            structure = uncertainty_characteristics['Berkson_error']['structure']

        self.structure= structure

        self.n = {group: len(self.group_view[group][(self.group_view[group]['model'].values == measurement_model) &
                                                                (self.group_view[group]['Z'] > 0)]) for group in self.groups}


        ### RR ### kein vorschlag, aber mit SH klären das stadn schon, aber das hier impliziert IMMER Multiplikative Fehler, außerdem: warum zur hölle hier über stats norm und nicht np.random? np ist bestimmt 80% schneller
        self.error_values = {group: np.exp(stats.norm.rvs(loc = -(error_sd**2)/2, scale = error_sd, 
                                          size = self.n[group])) for group in self.groups}
        ### RR ###
        ### RR ###  Vorschlag neu:
        # Zeile drüber umschreiben und dist nuttzen wie in error component - impliziert aich immer multiplikativen error
        dist = stats.lognorm(scale=np.exp(-(error_sd**2) / 2), s=error_sd)   # hier ist np.random nicht wichtig, da hier nur initalisiert wird
        self.error_values = {group:  dist.rvs(size = self.n[group]) for group in self.groups}
        self.log_LR_measurement_model_Berkson = lambda proposed_values, group: np.sum(dist.logpdf(proposed_values)-dist.logpdf(self.error_values[group]))  # das ist im grunde aus der ErrorComponent geklaut
        ### RR ### 


        # only needed for shared error
        if name == "shared_error":
            self.n = {group: len(np.unique(self.group_view[group][(self.group_view[group]['model'].values == measurement_model) &
                                                                (self.group_view[group]['Z'] > 0)]['Ident'])) for group in self.groups}
            individuals = {group: np.unique(self.group_view[group][(self.group_view[group]['model'].values == measurement_model) &
                                                                    (self.group_view[group]['Z'] > 0)]['Ident'])
                                                        for group in self.groups}
            self.group_error_matrices = latent_variables.create_group_individual_matrices(period_data, individuals)
            self.long_values = {group: self.values[group].transpose()*self.group_error_matrices[group] for group in self.groups}   # ist so lang wie eine Gruppe. d.h. an den stellen die nicht upgedated werden stehen hier 0 oder einer 1 (je nach additiv oder multiplicativ)

        self.observed_exposure = {group: self.group_view[group][self.group_view[group]['model'].values == measurement_model]['Z'] for group in self.groups} 


        self.proposal_sd = {group: proposal_sd for group in self.groups} 
        self.group_acceptance  = {group: 0 for group in self.groups} 
        self.group_count = {group: 0 for group in self.groups} 
        ## group_period_selection selects the period values in the group data frame
       
        self.propose_values = lambda group: self.propose_values_group(group) # for namespace reasons we have this function as a lambda function wrapper around the actual function

        self.log_LR_measurement_model_classical_lognormal = lambda Xt, Xcand, group: self.log_LR_measurement_model_classical_lognormal_group(Xt, Xcand, group, sigma=error_sd)

        # self.log_LR_exposure_model = lambda Xt, Xcand: self.log_LR_exposure_model_lognormal(Xt, Xcand, uncertainty_characteristics['exposure_model_parameters'])
        # das hier überschreibt das exposure model vom UncertainFactor
        self.log_LR_exposure_model = lambda Xt, Xcand: self.log_LR_exposure_model_lognormal_fast(Xt, Xcand, uncertainty_characteristics['exposure_model_parameters'])

    def get_values(self) -> dict:
        return self.error_values


    def propose_values_group(self, group) -> np.ndarray:
        # proposed_values = np.exp(stats.norm.rvs(loc = 0, 
                                                # scale=self.proposal_sd[group],
                                                # size=self.n[group]))
        proposed_values = np.exp(np.random.normal(loc=0, scale=self.proposal_sd[group], size=self.n[group]))
        return proposed_values * self.error_values[group]


    def propose_values_classical(self, Xt, group) -> np.ndarray:
        proposed_values = np.exp(np.random.normal(loc=0, scale=self.proposal_sd[group], size=self.n[group]))
        return proposed_values * Xt



    def log_proposal_ratio_Berkson(self, cand_values, group):
        """
        Calculates the the log proposal ratio for a given value passed to the function

        :return: The Ratio as float64
        """
        if self.structure == 'multiplicative':
            # log proposal simpiefies to cand / curr
            log_proposal_ratio = np.log(np.prod(cand_values / self.error_values[group]))

        elif self.structure == 'additive':
            raise ValueError("Es darf keinen additiven Berkson Fehler geben!!!")
            log_proposal_ratio = 0
            # falls wir hier bei 0 trunkieren müssen wir aus dem covid- proposal_ratio von trunc_norm benutzen
        
        return log_proposal_ratio


    def log_proposal_ratio_classical(self, cand_values, group):
        raise NotImplementedError("Classical Error for Unshared is hard coded in the update (see LatentVariable)")
        # könnte sein, dass es wie Berkson ist



    def log_LR_measurement_model_classical_lognormal_group(self, Xt, Xcand, group, sigma):
        '''
            log_LR_measurement_model_classical_lognormal is the accelerated version of internal_vector and ratio_internal.
             It only contains the neccesary terms to evaluate the ratio between the current
            and the candidate value of the exposures according to the measurement
            process. See also the demonstration in the pdf file 'Accelerate
            evaluation internal process'
            to understand which terms are omitted.
            It is only adapted for classical error.
            '''
        log_Xt = np.log(Xt)
        log_Xcand = np.log(Xcand)
        return( 1/(2*(sigma**2)) * np.sum(log_Xt**2 - log_Xcand**2 +
                                                         (2 * np.log(self.observed_exposure[group].values) +
                                                          (sigma**2))*(log_Xcand - log_Xt)))




    def log_LR_exposure_model_lognormal(self, Xt, Xcand, parameters):
        '''
        Calculates the ratio for the exposure model for two potential values of X.
        '''
        mu_x = parameters['mu']
        sigma_x = parameters['sigma']
        log_Xt = np.log(Xt)
        log_Xcand  = np.log(Xcand)
        first_term = np.sum((log_Xt**2 - log_Xcand**2))/(2*sigma_x**2)
        second_term = np.sum((mu_x/sigma_x**2 - 1)*(log_Xcand- log_Xt))
        return(first_term +second_term)

    
    def log_LR_exposure_model_lognormal_fast(self, Xt, Xcand, parameters):
        '''
        Calculates the ratio for the exposure model for two potential values of X.
        '''
        mu_x = parameters['mu']
        sigma_x = parameters['sigma']
        return basics.ratio_exposure_lognormal_jit(Xt.astype(float), Xcand.astype(float), float(mu_x), float(sigma_x))






    def adapt_proposal(self, nb_iterations, phase):
        """
        This function adjust the proposal sd of the error components.
        """
        # this is not the most memory efficent way, but it is clean and readable
        # acceptance_rate = {group: self.group_acceptance[group]/self.group_count[group] for group in self.group_acceptance}
        # diff = {group: acceptance_rate[group] - self.target}
        # change = {group: abs(diff[group]) > 0.02}
        # sign = {group: np.sign(diff[group])}
        # self.proposal_sd = {group: self.proposal_sd[group] * (1  + 0.1* sign[group] * change[group])}
        # self.group_acceptance  = {group: 0 for group in self.groups} 
        # self.group_count = {group: 0 for group in self.groups} 

        mean_acceptance_rate = []
        for group in self.groups:
            acceptance_rate = self.group_acceptance[group]/self.group_count[group]
            mean_acceptance_rate.append(acceptance_rate)
            diff = acceptance_rate - self.target
            change = abs(diff) > 0.02
            sign = np.sign(diff)
            self.proposal_sd[group] *= (1  + 0.1* sign * change)
            self.group_acceptance[group] = 0
            self.group_count[group] = 0

        if len(mean_acceptance_rate) >0:
            print('Unshared Error mean acceptance rate '+ str(np.round(np.mean(mean_acceptance_rate), 3))+
                ' with interval [' + str(np.round(np.percentile(mean_acceptance_rate, 2.5),3))+ 
                ',' + str(np.round(np.percentile(mean_acceptance_rate, 97.5), 3))+']')



    def reset_values(self, iterations: int) -> None:
        """
        Resets the samples from the chain.
        """
        pass



    def write_samples(self, path_results: str, thin: int = 1):
        """
        Saves the chain to the path passed to the function.
        """
        pass
