from typing import TypedDict
from typing import Union
from itertools import product, combinations

import dwave_networkx as dnx
import networkx

from modules.applications.Mapping import *
from modules.circuits.CircuitHardwareEfficient import CircuitHardwareEfficient
from modules.circuits.CircuitBrickwork import CircuitBrickwork
from utils import start_time_measurement, end_time_measurement
from qiskit.quantum_info import Pauli


class PauliCorrelationsEncoding(Mapping):
    """
    QUBO formulation for the TSP.

    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.submodule_options = ["CircuitHardwareEfficient", "CircuitBrickwork"]

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module

        :return: list of dict with requirements of this module
        :rtype: list[dict]
        """
        return [
            {
                "name": "networkx",
                "version": "2.8.8"
            },
            {
                "name": "dwave_networkx",
                "version": "0.8.13"
            }
        ]
    
    def get_default_submodule(self, option: str) -> Union[CircuitHardwareEfficient, CircuitBrickwork]:

        if option == "CircuitHardwareEfficient":
            return CircuitHardwareEfficient()
        elif option == "CircuitBrickwork":
            return CircuitBrickwork()
        else:
            raise NotImplementedError(
                f"Circuit Option {option} not implemented")
        
    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this mapping

        :return:
                 .. code-block:: python

                     return {
                                "lagrange_factor": {
                                                    "values": [0.75, 1.0, 1.25],
                                                    "description": "By which factor would you like to multiply your "
                                                                    "lagrange?",
                                                    "custom_input": True,
                                                    "postproc": float
                                }
                            }

        """
        return {
            "compression_degree": {
                "values": [1,2,3],
                "description": "How many traceless single-qubit strings do you want to use? (compression_degree)"
                 }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config
 
        .. code-block:: python

             lagrange_factor: float

        """
        compression_degree: int

    def map(self, input_data: dict, config: dict) -> (dict, float):
        start = start_time_measurement()

        n_qubits = input_data["n_qubits"]
        k = config["compression_degree"]
        pauli_strings = []
        max_pauli_strings = len(input_data["adjacency_matrix"])

        # Generate all unique k-combinations of qubit positions
        for qubit_indices in combinations(range(n_qubits), k):
            for pauli_op in 'XYZ':
            # Create a list of 'I' for all qubits
                label = ['I'] * n_qubits
                # Replace 'I' with the actual Pauli operator at the specified qubit positions
                for index in qubit_indices:
                    label[index] = pauli_op
                    # Create a Pauli object from the label
                pauli_object = Pauli(''.join(label))
                # Append the Pauli object to the list
                pauli_strings.append(pauli_object)
                if len(pauli_strings) >= max_pauli_strings:
                    break
            if len(pauli_strings) >= max_pauli_strings:
                break
        
        encoding_config = {"n_qubits": n_qubits, "pauli_strings": pauli_strings, "adjaceny_matrix": input_data["adjacency_matrix"], "compression_degree":k}
        
        end = end_time_measurement(start)
        return encoding_config, end
