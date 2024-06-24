#  Copyright 2021 The QUARK Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import TypedDict
import logging
from cma import CMAEvolutionStrategy
from tensorboardX import SummaryWriter
from matplotlib import figure, axes
import matplotlib.pyplot as plt
from qiskit.quantum_info import Pauli
from qiskit.opflow.primitive_ops import PauliOp
from qiskit.opflow import PauliExpectation, CircuitSampler, StateFn
from qiskit.quantum_info import Statevector
from qiskit import Aer, transpile, assemble
from qiskit_aer import AerSimulator
from qiskit.providers import Backend
from qiskit import execute
import torch
from qiskit.exceptions import QiskitError
import qiskit_aer 
import math


from modules.training.Training import *
from utils_mpi import is_running_mpi, get_comm

MPI = is_running_mpi()
comm = get_comm()


class Classical(Training):
    """
    This module optmizes the paramters of quantum circuit using CMA-ES. 
    This training method is referred to as quantum circuit born machine (QCBM).
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__("Classical")

        self.n_states_range: list
        self.target: np.array
        self.study_generalization: bool
        self.generalization_metrics: dict
        self.writer: SummaryWriter
        self.loss_func: callable
        self.fig: figure
        self.ax: axes.Axes

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module

        :return: list of dict with requirements of this module
        :rtype: list[dict]
        """
        return [
            {
                "name": "numpy",
                "version": "1.23.5"
            },
            {
                "name": "cma",
                "version": "3.3.0"
            },
            {
                "name": "tensorboard",
                "version": "2.13.0"
            },
            {
                "name": "tensorboardX",
                "version": "2.6.2"
            }
        ]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for the quantum circuit born machine

        :return:
            .. code-block:: python

                return {

                    "population_size": {
                        "values": [5, 10, 100, 200, 10000],
                        "description": "What population size do you want?"
                    },

                    "max_evaluations": {
                        "values": [100, 1000, 20000, 100000],
                        "description": "What should be the maximum number of evaluations?"
                    },

                    "sigma": {
                        "values": [0.01, 0.5, 1, 2],
                        "description": "Which sigma would you like to use?"
                    },

                    "pretrained": {
                        "values": [False],
                        "custom_input": True,
                        "postproc": str,
                        "description": "Please provide the parameters of a pretrained model?"
                    },

                    "loss": {
                        "values": ["KL", "NLL"],
                        "description": "Which loss function do you want to use?"
                    }
                }
        """
        return {
            "population_size": {
                "values": [5, 10, 100, 200, 10000],
                "description": "What population size do you want?"
            },

            "max_evaluations": {
                "values": [100, 1000, 20000, 100000],
                "description": "What should be the maximum number of evaluations?"
            },

            "sigma": {
                "values": [0.01, 0.5, 1, 2],
                "description": "Which sigma would you like to use?"
            },

            "pretrained": {
                "values": [False],
                "custom_input": True,
                "postproc": str,
                "description": "Please provide the parameters of a pretrained model?"
            },

            "loss": {
                "values": ["KL", "NLL"],
                "description": "Which loss function do you want to use?"
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python

            population_size: int
            max_evaluations: int
            sigma: float
            pretrained: str
            loss: str

        """
        population_size: int
        max_evaluations: int
        sigma: float
        pretrained: str
        loss: str

    def get_default_submodule(self, option: str) -> Core:
        raise ValueError("This module has no submodules.")

    def setup_training(self, input_data, config) -> tuple:
        """
        Method to configure the training setup including CMA-ES and tensorboard.

        :param input_data: A representation of the quntum machine learning model that will be trained
        :type input_data: dict
        :param config: Config specifying the parameters of the training
        :type config: dict
        :return: Updated input_data 
        :rtype: dict
        """

        logging.info(
            f"Running config: [backend={input_data['backend']}] [n_qubits={input_data['n_qubits']}] "\
            f"[population_size={config['population_size']}]")

        self.adjacency_matrix = input_data["adjacency_matrix"]
        self.pauli_strings = input_data["pauli_strings"]
        self.n_qubits = input_data["n_qubits"]
        self.execute_circuit = input_data["execute_circuit"]
        self.circuit = input_data["circuit"]
        self.backend = input_data["backend"]
        self.n_shots =input_data["n_shots"]
        self.k = input_data["compression_degree"]

        
        if config['loss'] == "KL":
            self.loss_func = self.kl_divergence
        elif config['loss'] == "NLL":
            self.loss_func = self.nll
        elif config['loss'] == "nonlinear_loss":
            self.loss_func = self.nonlinear_loss
        else:
            raise NotImplementedError("Loss function not implemented")

        self.n_params = len(self.circuit.parameters)
        x0 = torch.randn(self.n_params, requires_grad=True)  # Initial parameters as a PyTorch tensor

        # Setup the optimizer
        optimizer = torch.optim.Adam([x0], lr=config.get('learning_rate', 0.01))
        
        
        return x0, optimizer

    def start_training(self, input_data: dict, config: Config, **kwargs: dict) -> (dict, float):
        """
        This function finds the best parameters of the circuit on a transformed problem instance and returns a solution.

        :param input_data: A representation of the quntum machine learning model that will be trained
        :type input_data: dict
        :param config: Config specifying the paramters of the training
        :type config: dict
        :param kwargs: optional additional settings
        :type kwargs: dict
        :return: Dictionary including the information of previous modules as well as of the training
        :rtype: dict
        """

        params, optimizer = self.setup_training(input_data, config)
        W = torch.tensor(self.adjacency_matrix)
        alpha = self.n_qubits ** math.floor(self.k / 2)
        
        self.beta = 0.5
        num_edges = np.count_nonzero(self.adjacency_matrix) / 2
        self.m = 3 * self.n_qubits * (self.n_qubits - 1) / 2
        self.nu =  (num_edges / 2) + (self.m - 1) / 4

        num_epochs = 1000  # Placeholder for the number of training epochs
        for epoch in range(num_epochs):
            
            optimizer.zero_grad()
            params = torch.nn.Parameter(torch.randn(self.n_params, requires_grad=True))
            params_list = params.detach().numpy().tolist()
            solutions = self.executed_circuit(params_list,self.circuit)


            expectation_values = torch.tensor(self.get_expectation_value(self.pauli_strings, solutions),dtype=torch.float32, requires_grad=True)
            loss = self.nonlinear_loss(W, alpha, expectation_values)

            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")

        
        





        # After training, params holds the optimized parameters
        optimized_params = params.detach().numpy()

        # Update input_data with the best parameters found
        input_data['optimized_params'] = optimized_params
        return input_data, loss



    def data_visualization(self, loss_epoch, pmfs_model, samples, epoch):
        index = loss_epoch.argmin()
        best_pmf = pmfs_model[index] / pmfs_model[index].sum()
        if self.study_generalization:

            if samples is None:
                counts = self.sample_from_pmf(
                    n_states_range=self.n_states_range,
                    pmf=best_pmf.get() if GPU else best_pmf,
                    n_shots=self.generalization_metrics.n_shots)
            else:
                counts = samples[int(index)]

            metrics = self.generalization_metrics.get_metrics(counts.get() if GPU else counts)
            for (key, value) in metrics.items():
                self.writer.add_scalar(f"metrics/{key}", value, epoch)

        nll = self.nll(best_pmf.reshape([1, -1]), self.target)
        kl = self.kl_divergence(best_pmf.reshape([1, -1]), self.target)
        self.writer.add_scalar("metrics/NLL", nll.get() if GPU else nll, epoch)
        self.writer.add_scalar("metrics/KL", kl.get() if GPU else kl, epoch)

        self.ax.clear()
        self.ax.imshow(
            best_pmf.reshape(int(np.sqrt(best_pmf.size)), int(np.sqrt(best_pmf.size))).get() if GPU
            else best_pmf.reshape(int(np.sqrt(best_pmf.size)),
                                    int(np.sqrt(best_pmf.size))),
            cmap='binary',
            interpolation='none')
        self.ax.set_title(f'Iteration {epoch}')
        self.writer.add_figure('grid_figure', self.fig, global_step=epoch)

        return best_pmf

    def nonlinear_loss(self, W, alpha, expectation_values):
        tanh_values = torch.tanh(alpha * expectation_values)
        product_matrix = torch.outer(tanh_values, tanh_values)
        loss_original = torch.sum(W * product_matrix)
        regularization_term = self.beta * self.nu * (torch.sum(tanh_values ** 2) / self.m) ** 2
        
        loss = loss_original + regularization_term
        return loss

    

    def get_expectation_value(self, pauli, statevector):

        operator_matrices = [p.to_matrix() for p in pauli]
        expectation_value = [np.vdot(statevector, op_matrix @ statevector.data) for op_matrix in operator_matrices]

        return np.real(expectation_value)


    def executed_circuit(self,params_list, circuit):
        backend = qiskit_aer.backends.statevector_simulator.StatevectorSimulator()

        # Ensure that there are no measurements in the circuit
        circuit.remove_final_measurements(inplace=True)
        # Transpile the circuit for the backend
        param_dict = {param: value for param, value in zip(circuit.parameters, params_list)}
        bound_circuit = circuit.bind_parameters(param_dict)
        transpiled_circuit = transpile(bound_circuit, backend=backend)

        # Execute the transpiled circuits
        job = backend.run(transpiled_circuit)

        # Retrieve the statevectors
        result = job.result()
        data = result.data()
        statevector = result.get_statevector()
        return statevector
