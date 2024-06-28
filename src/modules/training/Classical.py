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
from torch.utils.tensorboard import SummaryWriter
from qiskit.exceptions import QiskitError
import qiskit_aer 
import math

from qiskit.quantum_info import Operator

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

        self.writer: SummaryWriter 
        self.loss_func: callable = None
        self.fig = None
        self.ax = None
        self.n_params = 0
        self.circuit = None
        self.adjacency_matrix = None
        self.pauli_strings = None
        self.n_qubits = 0
        self.beta = 0.5
        self.nu = 0
        self.m = 0
        self.k = 0

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
        self.circuit = input_data["circuit"]
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
        params = torch.randn(self.n_params, requires_grad=True)  # Initial parameters as a PyTorch tensor

        # Setup the optimizer
        optimizer = torch.optim.Adam([params], lr=config.get('learning_rate', 0.01))
        self.writer = SummaryWriter("/Users/admin/Documents/marwamarso/QUARK-1/results")
        
        return params, optimizer

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
        num_epochs = config.get("num_epochs", 1000)
        min_loss = float('inf')
        best_cut_value = float('-inf')
        V_best = kwargs.get("V_best", float('inf'))
        
        self.backend = qiskit_aer.backends.statevector_simulator.StatevectorSimulator()
        W = torch.tensor(self.adjacency_matrix)
        alpha = self.n_qubits ** math.floor(self.k / 2)
        self.fig, self.ax = plt.subplots()
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            params_list = params.detach().numpy().tolist()
            statevector = self.executed_circuit(params_list, self.circuit)
            expectations = self.get_expectation_values(self.pauli_strings, statevector)
            loss = self.loss_func(W, alpha, expectations)
            loss.backward()
            optimizer.step()
            min_loss = min(min_loss, loss.item())
            print(f"Epoch {epoch}, Loss: {min_loss}")

            # Calculate the objective function value (cut value)
            cut_value = self.calculate_cut_value(W, expectations)
            # Update the best cut value seen so far
            best_cut_value = max(best_cut_value, cut_value.item())

            r = cut_value.item() / V_best if V_best != 0 else 0

            if epoch % 10 == 0:  # Change 10 to the desired frequency of logging
                self.data_visualization(loss.item(), epoch)

        self.writer.close()

        input_data['optimized_params'] = params.detach().numpy()
        return input_data, min_loss



    def data_visualization(self, loss_epoch, epoch):

        non_linearl_loss = loss_epoch

        self.writer.add_scalar("metrics/loss", non_linearl_loss, epoch)


        self.ax.clear()

        self.ax.set_title(f'Iteration {epoch}')
        self.writer.add_figure('grid_figure', self.fig, global_step=epoch)

        return non_linearl_loss

    def nonlinear_loss(self, W, alpha, expectation_values):
        num_edges = np.count_nonzero(self.adjacency_matrix) / 2
        self.nu =  (num_edges / 2) + (self.m - 1) / 4
        self.m = 3 * self.n_qubits * (self.n_qubits - 1) / 2
        self.nu =  (num_edges / 2) + (self.m - 1) / 4

        tanh_values = torch.tanh(alpha * expectation_values)
        product_matrix = torch.outer(tanh_values, tanh_values)
        loss_original = torch.sum(W * product_matrix)
        regularization_term = self.beta * self.nu * (torch.sum(tanh_values ** 2) / self.m) ** 2
        
        loss = loss_original + regularization_term
        return loss

    

    def get_expectation_value(self, pauli_strings, statevector):

        expectation_values = torch.empty(len(pauli_strings), dtype=torch.float32, device=statevector.device)

        # Calculate the expectation value for each Pauli string.
        for idx, pauli in enumerate(pauli_strings):
            # Convert the Pauli string to a matrix.
            pauli_matrix_np = Operator(pauli).data
            pauli_matrix = torch.tensor(pauli_matrix_np, dtype=torch.cfloat, device=statevector.device, requires_grad=True)

            
            # Calculate the expectation value.
            statevector_adjoint = torch.conj(statevector)  # Conjugate transpose (adjoint)
            expectation_value = torch.real(torch.dot(statevector_adjoint, torch.mv(pauli_matrix, statevector)))
            expectation_values[idx] = expectation_value

        return expectation_values

    def get_expectation_values(self, pauli_objects,statevector):
        sampler = CircuitSampler(self.backend)
        expectation_values = []

        for pauli in pauli_objects:
            # Wrap the quantum state in a StateFn
            state_fn = StateFn(statevector)

            # Wrap the Pauli object in a PauliOp and convert to an expectation
            pauli_expectation = PauliExpectation().convert(StateFn(pauli, is_measurement=True) @ state_fn)

            # Use the sampler to compute the expectation value
            sampled_expect_op = sampler.convert(pauli_expectation)
            expectation_value = sampled_expect_op.eval().real

            expectation_values.append(expectation_value)
        
        return expectation_values

    def executed_circuit(self,params, circuit):
        

        # Make sure 'params' is a tensor with gradient tracking enabled.
        if not isinstance(params, torch.Tensor):
            params = torch.tensor(params, dtype=torch.float32, requires_grad=True)

        # Prepare the parameter dictionary for binding.
        param_dict = {param: value.item() for param, value in zip(circuit.parameters, params)}

        # Bind the parameters and transpile the quantum circuit for the backend.
        bound_circuit = circuit.bind_parameters(param_dict)
        transpiled_circuit = transpile(bound_circuit, backend=self.backend)

        # Execute the transpiled circuit.
        job = self.backend.run(transpiled_circuit)

        # Retrieve the statevector as a PyTorch tensor with 'requires_grad' enabled.
        result = job.result()
        statevector_np = result.get_statevector()
        statevector_torch = torch.tensor(statevector_np, dtype=torch.cfloat, requires_grad=True)

        return statevector_np

    def calculate_cut_value(self, W, expectations):
        """
        Calculate the objective function (cut value) using the adjacency matrix and the expectation values.
        """
        product_matrix = torch.outer(expectations, expectations)
        cut_value = torch.sum(W * (1 - product_matrix))
        return cut_value