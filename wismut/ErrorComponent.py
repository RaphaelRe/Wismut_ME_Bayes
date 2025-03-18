"""
This module implements the ErrorComponents (classical and Berkson)
"""
import numpy as np
import scipy.stats as stats


class ErrorComponent:
    """
    This is a classical error component which provides basic structures.

    :param structure: A string which sould be either 'additive' or 'multiplicative'
    :param dimension: The size of the error. Serves only as info, is not actually used
    :paran sigma: A float specifying the variance of the error

    """
    ####
    #### Remark;  Diese Klasse is eigentlich komplett sinnlos und kann theoretisch in den Uncertain factor eingegliedert werden. Sie ist aufgrund von historischen Begebenheiten entstanden und inzwischen obsolet.
    def __init__(self, structure: str, dimension: int, sigma: float) -> None:
        self.structure = structure
        self.dim = dimension

        if self.structure == 'additive':
            dist = stats.norm(loc=0, scale=sigma)
        else:
            dist = "harakiki"  # for multiplicative error you should use the log_LR_measurement_model of the UncertainFactor
            # dist = stats.lognorm(loc=0, scale=np.exp(-(sigma**2) / 2), s=sigma)

        self.log_LR_measurement_model = lambda proposed_values: np.sum(dist.logpdf(proposed_values)-dist.logpdf(self.values))




    def get_values(self):
        """
        Getter method
        :return: current values
        """
        return self.values

    def set_values(self, values):
        """
        Setter method
        """
        self.values = values





class BerksonErrorComponent(ErrorComponent):
    '''
    The Berkson class inherits from the ErrorComponent class and implements a Berkson error.
    '''
    def __init__(self, structure, dimension, sigma, proposal_sd):
        super().__init__(structure, dimension, sigma)

        self.proposal_sd = proposal_sd
        # self.initialize_values(sigma)
        self.initialize_values(proposal_sd)

        if self.structure == 'additive':
            dist = stats.norm(loc=0, scale=sigma)
        else:
            dist = stats.lognorm(loc=0, scale=np.exp(-(sigma**2) / 2), s=sigma)

        self.log_LR_measurement_model = lambda proposed_values: np.sum(dist.logpdf(proposed_values)-dist.logpdf(self.values))


    def log_proposal_ratio(self, cand_values):
        """
        Calculates the the log proposal ratio for a given value passed to the function

        :return: The Ratio as float64
        """
        if self.structure == 'multiplicative':
            # full ratio
            # log_proposal_ratio = np.sum(stats.lognorm.logpdf(self.values, loc=0, scale=cand_values, s=self.proposal_sd) -
                    # stats.lognorm.logpdf(cand_values, loc=0, scale=self.values, s=self.proposal_sd))

            # log proposal simpiefies to cand / curr
            log_proposal_ratio = np.log(np.prod(cand_values / self.values))

        elif self.structure == 'additive':
            raise ValueError("Es darf keinen additiven Berkson Fehler geben!!!")
            log_proposal_ratio = 0
            # falls wir hier bei 0 trunkieren müssen wir aus dem covid- proposal_ratio von trunc_norm benutzen
        return log_proposal_ratio

    # def initialize_values(self, sigma):
    def initialize_values(self, proposal_sd):
        """
        Initialiizes the start values
        """
        # assumes always multiplicative

        # values = np.random.normal(0,sigma,self.dim)
        # self.values = np.exp(values)

        # vermutlich böööööse self.values = stats.lognorm.rvs(loc=0, scale=np.exp(-(sigma**2)/2), s=sigma, size=self.dim)
        self.values = stats.lognorm.rvs(loc=0, scale=np.exp(-(proposal_sd**2)/2), s=proposal_sd, size=self.dim)


    def propose_values(self):
        """
        Proposes a value given the current state.

        :return: numpy array with proposed errors
        """
        # errors_new = np.random.normal(loc=0, scale=self.proposal_sd, size=self.dim)
        # log_proposed_values =  np.log(self.values) + errors_new
        # proposed_values = np.exp(log_proposed_values)
        # print("Achtung zwei versionen beim vorschlag für multiplikative berkson/clasical! esten ob gleich")
        # return stats.lognorm.rvs(s=self.proposal_sd, loc=0, scale=self.values)
        return np.exp(stats.norm.rvs(loc=0, scale=self.proposal_sd, size=(self.values.shape[0]))) * self.values  # wenn diese Version auf np.random.normal wechseln


    def adapt_proposal(self, sign, change):
        """
        Adapt the current proposal sd
        """
        self.proposal_sd *= (1 + 0.1 * sign * change)
        # change ist entweder 0 oder 1, sign ist entweder -1 oder +1,  Wird vom uncertain_factor bestimmt und übergeben






# class MultClassicalErrorComponent(BerksonErrorComponent):
    # '''
    # A multiplicative classical error which inherits from the BerksonErrorComponent.
    # Only used for lognormally distirbuted exposure. Maybe also normally.
    # '''

    # def __init__(self, dimension, sigma, proposal_sd, observed_values):
        # structure = 'multiplicative'
        # super().__init__(structure, dimension, sigma, proposal_sd)

        # self.observed_values = observed_values

        # self.log_LR_measurement_model = lambda current_values, proposed_values: self.log_LR_measurement_model_classical_lognormal(current_values, proposed_values, sigma)


    # def log_LR_measurement_model_classical_lognormal(self, Xt, Xcand, sigma):
        # '''
        # log_LR_measurement_model_classical_lognormal is the accelerated version of internal_vector and ratio_internal.
        # It only contains the neccesary terms to evaluate the ratio between the current
            # and the candidate value of the exposures according to the measurement
        # process.
        # '''
        # log_Xt = np.log(Xt)
        # log_Xcand = np.log(Xcand)
        # return( 1/(2*(sigma**2)) * np.sum(log_Xt**2 - log_Xcand**2 +
                                                         # (2 * np.log(self.observed_values) +
                                                          # (sigma**2))*(log_Xcand - log_Xt)))

