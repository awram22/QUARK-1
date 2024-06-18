from typing import TypedDict
from typing import Union
from itertools import product, combinations

import dwave_networkx as dnx
import networkx

from modules.applications.Mapping import *
from modules.circuits.CircuitHardwareEfficient import CircuitHardwareEfficient
from modules.circuits.CircuitBrickwork import CircuitBrickwork
from utils import start_time_measurement, end_time_measurement


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
        """

        """
        start = start_time_measurement()
        pauli_operators = ['X', 'Y', 'Z']
        n_qubits = input_data["n_qubits"]
        adjacency_matrix = input_data["adjacency_matrix"]
        k = config["compression_degree"]
        pauli_strings = []

        # Iterate over each Pauli operator type
        for op in pauli_operators:
            # Generate all unique placements for k non-identity operators among n qubits
            unique_placements = combinations(range(n_qubits), k)
            
            for placement in unique_placements:
                # Start with an identity string
                pauli_string = ['I'] * n_qubits
                # Place the identical non-identity Pauli operators in the string
                for index in placement:
                    pauli_string[index] = op
                # Convert to a string and add to the list
                pauli_strings.append(''.join(pauli_string))
        
        encoding_config = {"n_qubits": n_qubits, "pauli_strings": pauli_strings, "adjacency_matrix": adjacency_matrix}
        
        return encoding_config, end_time_measurement(start)

