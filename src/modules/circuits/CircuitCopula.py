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
from itertools import combinations

from scipy.special import binom

from modules.circuits.Circuit import Circuit
from modules.applications.QML.generative_modeling.mappings.LibraryQiskit import LibraryQiskit
from modules.applications.QML.generative_modeling.mappings.PresetQiskitNoisyBackend import PresetQiskitNoisyBackend
from modules.applications.QML.generative_modeling.mappings.CustomQiskitNoisyBackend import CustomQiskitNoisyBackend


class CircuitCopula(Circuit):
    """
    This class generates a library-agnostic gate sequence, i.e. a list containing information
    about the gates and the wires they act on. The marginal ditribtions generated by the copula 
    are uniformaly distributed.
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__("DiscreteCopula")
        self.submodule_options = ["LibraryQiskit", "CustomQiskitNoisyBackend", "PresetQiskitNoisyBackend"]

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module

        :return: list of dict with requirements of this module
        :rtype: list[dict]
        """
        return [
            {
                "name": "scipy",
                "version": "1.11.1"
            }
        ]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this Copula Circuit.

        :return:

        .. code-block:: python

            return {
                "depth": {
                    "values": [1, 2, 3, 4, 5],
                    "description": "What depth do you want?"
                },
            }

        """
        return {
            "depth": {
                "values": [1, 2, 3, 4, 5],
                "description": "What depth do you want?"
            },
        }

    def get_default_submodule(self, option: str) -> LibraryQiskit:
        if option == "LibraryQiskit":
            return LibraryQiskit()
        elif option == "PresetQiskitNoisyBackend":
            return PresetQiskitNoisyBackend()
        elif option == "CustomQiskitNoisyBackend":
            return CustomQiskitNoisyBackend()
        else:
            raise NotImplementedError(f"Option {option} not implemented")

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python

             depth: int

        """
        depth: int

    def generate_gate_sequence(self, input_data: dict, config: Config) -> dict:
        """
        Returns gate sequence of copula architecture
    
        :param input_data: Collection of information from the previous modules
        :type input_data: dict
        :param config: Config specifying the number of qubits of the circuit
        :type config: Config
        :return: Dictionary including the gate sequence of the Copula Circuit
        :rtype: dict
        """
        n_registers = input_data["n_registers"]
        n_qubits = input_data["n_qubits"]
        depth = config["depth"]
        n = n_qubits // n_registers

        gate_sequence = []
        for k in range(n):
            gate_sequence.append(["Hadamard", [k]])

        for j in range(n_registers - 1):
            for k in range(n):
                gate_sequence.append(["CNOT", [k, k + n * (j + 1)]])

        gate_sequence.append(["Barrier", None])

        shift = 0
        for _ in range(depth):
            for k in range(n):
                for j in range(n_registers):
                    gate_sequence.append(["RZ", [j * n + k]])
                    gate_sequence.append(["RX", [j * n + k]])
                    gate_sequence.append(["RZ", [j * n + k]])

            k = 3 * n + shift
            for i, j in combinations(range(n), 2):
                for l in range(n_registers):
                    gate_sequence.append(["RXX", [l * n + i, l * n + j]])

                k += 1
            shift += 3 * n + int(binom(n, 2))

        gate_sequence.append(["Barrier", None])

        for k in range(n_qubits):
            gate_sequence.append(["Measure", [k, k]])

        output_dict = {
            "gate_sequence": gate_sequence,
            "circuit_name": "Copula",
            "n_qubits": n_qubits,
            "n_registers": n_registers,
            "depth": depth,
            "histogram_train": input_data["histogram_train"],
            "store_dir_iter": input_data["store_dir_iter"],
            "dataset_name": input_data["dataset_name"]
        }

        return output_dict
