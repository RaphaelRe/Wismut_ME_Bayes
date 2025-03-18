"""
Main MCMC class

This module implements the master class MCMC which controls the whole algortihm.
"""

import time
from pandas import DataFrame
from IPython import display
# import progressbar as pb

from wismut.LatentVariables import LatentVariables
from wismut.Parameter import Parameter, PriorParameter, PriorParameterVector


class MCMC:
    """
    Main class of the algorithm.

    It holds all parameters and latent variables. It invokes the updates calls and the saving of the results.

    :param data: A pandas DataFrame object holding the required data.
    :param uncertainty_characteristics: A dictionaray specifying the error structure.
    :param s: a 1-dim np.array defining the break points for lambdas.
    :param path_results: A string defining the path for writing the chains.
    :param proposal_sd: A dictionary defining the proposal sd for each of the parameters.
    :param prior_parameters: A dictionary defining hyper parameters for the priors distributions of the parameters.
    :param start_values: A dictionary defining the start values of the parameters.
    :param informative_priors: A Boolean definign whether informative priors should be used. Should be False at the moment.
    :param chain: A string defining the chain which will be initialized.
    :param disease_model: A string defining the disease model. Possible options are 'ERR' and 'cox_like'.
    :param fixed_parameters: A boolean value. An option to fix the parameters. Not in use at the moment.
    :param write_predictions: A boolean defining if posterior predictions should be written. Currently not in use.
    """

    def __init__(
        self,
        data,
        uncertainty_characteristics,
        s,
        path_results,
        proposal_sd,
        prior_parameters,
        start_values,
        informative_priors,
        chain,
        disease_model,
        fixed_parameters=False,
        write_predictions=False,
        display=True,
    ):
        self.data = data
        self.path = path_results + chain + "_"

        # self.fix_latent_variable = fix_latent_variable
        self.fixed_parameters = fixed_parameters

        self.filename = self.path + "statistics" + ".txt"
        self.write_predictions = write_predictions

        self.disease_model = disease_model

        self.latent_variables = LatentVariables(
            data,
            uncertainty_characteristics,
            nb_iterations=10000,
            disease_model=self.disease_model,
            s=s,
        )

        self.initialize_parameters(
            data,
            proposal_sd,
            prior_parameters,
            start_values,
            s,
            chain,
            informative_priors,
        )

    def get_current_values(self) -> dict:
        """
        Getter on higher level. Returns all current parameter values as dictionary.

        :return: A dictionary with current state of the parameters.
        """
        current_values = {
            key: self.parameters[key].get_current_value() for key in self.parameters
        }
        return current_values

    def reset_values(self, iterations: int) -> None:
        """
        Helper function to reset the chains.

        :param iterations: Chains get re-initialized with an empty array of this length.
        """
        for key in self.parameters.keys():
            self.parameters[key].reset_values(iterations)
        self.latent_variables.reset_values(iterations)

    def update_chains(self) -> None:
        """
        Invokes an update for all parameters and latent variables.
        """
        for key in self.parameters.keys():
            # print('updating variable:' + str(key))
            self.parameters[key].update(
                self.get_current_values(), self.latent_variables
            )
        self.update_latentVariables()

    def update_latentVariables(self) -> None:
        """
        Invokes an update of the latent variables and therfore all uncertain factors.
        """
        current_parameter_values = self.get_current_values()
        self.latent_variables.update(current_parameter_values)

    def adapt_proposals(self, nb_iterations: int, phase: int) -> None:
        """
        Invokes the adapt_proposal functions of the parameers and the LatentVariables

        :param nb_iterations: Integer defining the start of the sequence of  interest, the acceptance rate is getting calculated on.
        :param phase: Integer specifying the current phase
        """
        for key in self.parameters:
            self.parameters[key].adapt_proposal(nb_iterations, phase)
        self.latent_variables.adapt_proposal(nb_iterations, phase)

    def run_adaptive_phase(
        self, nb_iterations: int, nb_phases: int, clear_display: bool = False
    ) -> None:
        """
        Runs the adaptive phase of the algorithm.

        :param nb_iterations: Indicates the start the sequence the acceptance  rate gets calculated on.
        :param nb_phases: The number of adaptive phases to do
        :param clear_display: A boolean specifying whether output should be cleard during adaptive phase.
        """
        print("starting adaptive phase......")
        # pbar = pb.ProgressBar(maxval=nb_phases).start()
        for phase in range(nb_phases):
            if clear_display:
                display.clear_output()

            print("Adaptive phase " + str(phase))
            for iterations in range(nb_iterations):
                self.update_chains()
            # pbar.update(phase)
            print("\n")
            self.adapt_proposals(nb_iterations, phase)
            # write proposal sd:
            # with open('proposal_sds' ,'a') as ff:
            # ff.write( '\n')
            # ff.write('Adaptive_phase: '+ str(phase) + '\n')
            # with open('proposal_sds_lv' ,'a') as ff:
            # ff.write( '\n')
            # ff.write('Adaptive_phase: '+ str(phase) + '\n')
        # pbar.finish()

    def run_burnin(self, nb_burnin: int) -> None:
        """
        Runs the burinin of the algorithm

        :param nb_burnin: An integer defining the number of burnin iterations
        """
        print("Start burnin-phase")
        self.reset_values(nb_burnin)
        # pbar = pb.ProgressBar(maxval=nb_burnin).start()
        for i in range(nb_burnin):
            self.update_chains()
            # pbar.update(i)
            if i % 1000 == 0:
                print(f"Iteration {i}/{nb_burnin} [BURNIN] finished")
        # pbar.finish()
        print("End Burnin-phase")

    def run_algorithm(
        self, nb_iterations: int, save_chains: bool = True, thin: int = 1
    ) -> None:
        """
        Runs the MCMC algorithm.

        :param nb_iterations: An integer defining the number of samples drawn from the posterior,i.e. the number of updates.
        :save_chains: Boolean indicating whether the chains should be written.
        """
        print("Start algorithm")
        self.reset_values(nb_iterations)
        # pbar = pb.ProgressBar(maxval=nb_iterations).start()
        for i in range(nb_iterations):
            self.update_chains()
            if i > 0 and i % 1000 == 0:
                print(f"Iteration {i}/{nb_iterations} [SAMPLING] finished")
                self.write_statistics()
                if save_chains:
                    self.write_chains(thin)
        self.write_chains(thin)

    def run_adaptive_algorithm(
        self,
        iterations: int,
        burnin: int,
        adaptive_phases: int = 100,
        save_chains: bool = True,
        thin: int = 1,
        clear_display: bool = False,
    ) -> None:
        """
        Run the full algorithm with adaptive phase, burnin and actual sampling.

        :param iterations: Integer defining the number of sampling iterations.
        :param burnin: Integer defining the the number of burnin iterations.
        :param adaptive_phases: Integer defining the number of adaptive phases.
        :param save_chains: Boolean indicating whether the chains should be written.
        :param thin: Integer defining the thinning of the chain. 10 means for example to keep only each 10th sample.

        The algorithm saves iterations/thin samples. The values of the adaptive phase and burn-in will be discarded
        """
        start = time.perf_counter()
        self.run_adaptive_phase(80, adaptive_phases, clear_display=clear_display)
        if clear_display:
            display.clear_output()
        print(f"Calculation time adaptive phase: {time.perf_counter() - start}")

        start = time.perf_counter()
        self.run_burnin(burnin)
        # print(time.perf_counter() - start)
        print(f"Calculation time burnin: {time.perf_counter() - start}")

        start = time.perf_counter()
        self.run_algorithm(iterations, save_chains, thin)
        # print(time.perf_counter() - start)
        print(f"Calculation time sampling: {time.perf_counter() - start}")
        if clear_display:
            display.clear_output()

    def write_chains(self, thin: int = 1) -> None:
        """
        This function saves the samples from the chain to the path given in the initialization of the MCMC object.

        :paran thin: Integer defining how many samples should be kept. A thin of 10 means that only each 10th sample will be written.
        """
        for key in self.parameters:
            self.parameters[key].write_samples(self.path, thin)
        self.latent_variables.write_samples(self.path, thin)
        # TODO: Posterior predictives

    def write_statistics(self) -> None:
        """
        This function writes summary statistics for the samples.
        """
        with open(self.filename, "w") as result:
            result.write("The final results of the algorithm are:")
            parameters = self.parameters
            for key in parameters:
                stats = parameters[key].get_statistics()
                result.write(key + ":")
                result.write(str(stats))
                result.write("\n \n")

            result.write("\n \n")

    def initialize_parameters(
        self,
        data: DataFrame,
        proposal_sd: dict,
        prior_parameters: dict,
        start_values: dict,
        s: list,
        chain: str,
        informative_priors: bool,
        precision: dict = {
            "beta": 3,
            "lambda1": 4,
            "lambda2": 4,
            "lambda3": 4,
            "lambda4": 4,
        },
    ):
        """
        Function initialize the parameters, i.e. beta, lambda1,.., lambda4. Gets called in the initialization of the MCMC object

        :param data: see MCMC class
        :param proposal_sd: see MCMC class
        :param prior_parameters: see MCMC class
        :param start_values: see MCMC class
        :param s: see MCMC class
        :param chain: see MCMC class
        :param informative_priors: see MCMC class
        :param precision: The number of digits printed out. Default should sufficient
        """

        parameter_names = start_values[chain].keys()
        self.parameters = {}
        for parameter in parameter_names:
            if parameter != "prior_parameters":
                if parameter in ["lambda1", "lambda2", "lambda3", "lambda4"]:
                    calculate_proposal_ratio = True
                else:
                    calculate_proposal_ratio = False
                if parameter == "beta":
                    informative_priors = True
                else:
                    informative_priors = False
                self.parameters[parameter] = Parameter(
                    name=parameter,
                    start_values=start_values[chain],
                    data=data,
                    precision=precision,
                    proposal_sd=proposal_sd,
                    prior_parameters=prior_parameters,
                    calculate_proposal_ratio=calculate_proposal_ratio,
                    s=s,
                    disease_model=self.disease_model,
                    informative_priors=informative_priors,
                )
            elif parameter == "prior_parameters":
                for measurement_model in start_values[chain][parameter]:
                    for uncertain_factor in start_values[chain][parameter][
                        measurement_model
                    ]:  # parameter is always 'prior_parameter' here
                        for prior_parameter in start_values[chain][parameter][
                            measurement_model
                        ][uncertain_factor]:
                            uf = self.latent_variables.uncertain_factors[
                                measurement_model
                            ][uncertain_factor]
                            exposure_model_distribution = uf.exposure_model_distribution
                            exposure_model_truncation = uf.exposure_model_truncation
                            if uf.vectorized_exposure:
                                exposure_years = uf.exposure_years
                                uf.prior_parameters[prior_parameter] = (
                                    PriorParameterVector(
                                        name_parent=prior_parameter,
                                        start_values=start_values[chain][parameter][
                                            measurement_model
                                        ],
                                        exposure_years=exposure_years,
                                        precision=3,
                                        proposal_sd=proposal_sd,
                                        prior_parameters=prior_parameters,
                                        uncertain_factor=uncertain_factor,
                                        calculate_proposal_ratio=True,
                                        measurement_model=measurement_model,
                                        exposure_model_distribution=exposure_model_distribution,
                                    )
                                )

                                true_mean_values = uf.get_yearly_values()

                            else:
                                uf.prior_parameters[prior_parameter] = PriorParameter(
                                    name=prior_parameter,
                                    start_values=start_values[chain][parameter][
                                        measurement_model
                                    ],
                                    precision=3,
                                    proposal_sd=proposal_sd,
                                    prior_parameters=prior_parameters,
                                    uncertain_factor=uncertain_factor,
                                    calculate_proposal_ratio=True,
                                    measurement_model=measurement_model,
                                    exposure_model_distribution=exposure_model_distribution,
                                    exposure_model_truncation=exposure_model_truncation,
                                )

                                true_mean_values = uf.true_mean_values

                            uf.prior_parameters[prior_parameter].set_uf_values(
                                true_mean_values
                            )
