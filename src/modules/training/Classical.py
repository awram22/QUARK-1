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
from tensorboardX import SummaryWriter
from matplotlib import figure, axes
import matplotlib.pyplot as plt
from qiskit.quantum_info import Pauli, commutator, Statevector, PauliList, SparsePauliOp
from qiskit.circuit.library import EfficientSU2
from psfam.pauli_organizer import PauliOrganizer
from qiskit.primitives import Estimator
from qiskit_aer import AerSimulator
from qiskit.providers import Backend
from qiskit.quantum_info import Pauli
#from qiskit import execute
import torch
#from torch.utils.tensorboard import SummaryWriter
from qiskit.exceptions import QiskitError
import math
from qiskit import transpile


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

        #self.writer: SummaryWriter 
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
            f"Running config: [backend={input_data['backend']}] [n_qubits={input_data['n_qubits']}] ")

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
        optimizer = torch.optim.Adam([params], lr=config.get('learning_rate', 0.001))
        #self.writer = SummaryWriter("/Users/admin/Documents/marwamarso/QUARK-1/results")
        
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
        
        self.backend = AerSimulator()
        W = torch.tensor(self.adjacency_matrix, dtype=torch.int64)
        alpha = self.n_qubits ** math.floor(self.k / 2)
        self.fig, self.ax = plt.subplots()
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            params_list = params.detach().numpy().tolist()
            expectation_values = self.calculate_expectation_values(self.circuit, params_list, self.pauli_strings, self.backend)
            loss = self.loss_func(W, alpha, torch.tensor(expectation_values, requires_grad=True))
            loss.backward()
            optimizer.step()
            min_loss = min(min_loss, loss.item())
            

            # Calculate the objective function value (cut value)
            cut_value = self.calculate_cut_value(W, torch.tensor(expectation_values, requires_grad=True))
            # Update the best cut value seen so far
            best_cut_value = max(best_cut_value, cut_value.item())

            r = cut_value.item() / V_best if V_best != 0 else 0
            print(f"Epoch {epoch}, Loss: {min_loss}, Cut Valie: {best_cut_value}, r: {r}")
            
            if epoch % 10 == 0:  # Change 10 to the desired frequency of logging
                self.data_visualization(loss.item(), epoch)

        #self.writer.close()

        input_data['optimized_params'] = params.detach().numpy()
        return input_data, min_loss



    def data_visualization(self, loss_epoch, epoch):

        non_linearl_loss = loss_epoch

        #self.writer.add_scalar("metrics/loss", non_linearl_loss, epoch)


        self.ax.clear()

        self.ax.set_title(f'Iteration {epoch}')
        #self.writer.add_figure('grid_figure', self.fig, global_step=epoch)

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

    def add_measurements(self, circuit, pauli_list):
        meas_circuit = circuit.copy()
        for pauli_string in pauli_list.to_labels():
            for i, pauli in enumerate(pauli_string):
                if pauli == 'X':
                    meas_circuit.h(i)
                elif pauli == 'Y':
                    meas_circuit.sdg(i)
                    meas_circuit.h(i)
                # For 'Z' or 'I' we do nothing (default Z-basis measurement)
        meas_circuit.measure_all()
        return meas_circuit

    def calculate_expectation_values(self, circuit, params, pauli_strings, backend):
        pauli_list = PauliList(pauli_strings)
        commute_groups = pauli_list.group_commuting(qubit_wise=True)
        
        expectation_values = {pauli: None for pauli in pauli_strings}
        estimator = Estimator()

        for group in commute_groups:
            # Create circuit with measurements for the combined Pauli string
            meas_circuit = self.add_measurements(circuit, group)

            # Bind parameters
            bound_circuit = meas_circuit.assign_parameters({p: v for p, v in zip(circuit.parameters, params)})

            # Transpile and get statevector
            transpiled_circuit = transpile(bound_circuit, backend)
            sim_result = self.backend.run(transpiled_circuit, shots=5000).result()
            counts = sim_result.get_counts()
            for pauli in group:
                total_counts = sum(counts.values())
                exp_val = 0
                pauli_str = pauli.to_label()
                for bitstring, count in counts.items():
                    parity = self.calculate_parity(bitstring, pauli_str)
                    exp_val += parity * count / total_counts
                expectation_values[pauli_str] = exp_val
        return [expectation_values[pauli.to_label()] for pauli in pauli_strings]

    def calculate_parity(self,bitstring, pauli_str):
        parity = 1
        for i, p in enumerate(pauli_str):
            if p in {'X', 'Y'}:
                parity *= (-1) ** (int(bitstring[i]) == 1)
        return parity


    def calculate_cut_value(self, W, expectations):
        """
        Calculate the objective function (cut value) using the adjacency matrix and the expectation values.
        """
        product_matrix = torch.outer(expectations, expectations)
        cut_value = torch.sum(W * (1 - product_matrix))
        return cut_value