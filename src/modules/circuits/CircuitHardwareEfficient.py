
from typing import TypedDict

from modules.circuits.Circuit import Circuit
from modules.applications.QML.generative_modeling.mappings.LibraryQiskit import LibraryQiskit


class CircuitHardwareEfficient(Circuit):
    """
    This class generates a library-agnostic gate sequence, i.e. a list containing information
    about the gates and the wires they act on.
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__("DiscreteStandard")
        self.submodule_options = ["LibraryQiskit"]

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
        Returns the configurable settings for this standard circuit.

        :return:
                 .. code-block:: python

                     return {
                                "depth": {
                                    "values": [1, 2, 3, 4, 5],
                                    "description": "What depth do you want?"
                                }
                            }

        """
        return {

            "depth": {
                "values": [1, 2, 3],
                "description": "What depth do you want?"
            }
        }

    def get_default_submodule(self, option: str) -> LibraryQiskit:
        if option == "LibraryQiskit":
            return LibraryQiskit()
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
        Returns gate sequence of a hardware-efficient circuit, including measurement in different bases.

        :param input_data: Collection of information from the previous modules
        :type input_data: dict
        :param config: Config specifying the number of qubits of the circuit
        :type config: Config
        :param pauli_strings: List of Pauli strings representing the measurement bases for each qubit
        :type pauli_strings: list
        :return: Dictionary including the gate sequence of the Hardware Efficient Circuit
        :rtype: dict
        """
        n_qubits = input_data["n_qubits"]
        depth = config["depth"]
        pauli_strings = input_data["pauli_strings"]
        adjacency_matrix = input_data['adjaceny_matrix']

        gate_sequence = []

        # Layer of Hadamard gates
        gate_sequence.extend([("Hadamard", [k]) for k in range(n_qubits)])

        # L repetitions of CNOT gates and single qubit Y (θi) rotations
        for _ in range(depth):
            for k in range(n_qubits - 1):
                gate_sequence.append(("CNOT", [k, k + 1]))
            for k in range(n_qubits):
                gate_sequence.append(("RY", [k]))
            gate_sequence.append(("Barrier", None))

        for k in range(n_qubits):
            gate_sequence.append(["Measure", [k, k]])

        output_dict = {
            "gate_sequence": gate_sequence,
            "circuit_name": "HardwareEfficient",
            "n_qubits": n_qubits,
            "depth": depth,
            "pauli_strings": pauli_strings,
            "adjacency_matrix":adjacency_matrix,
            "compression_degree":input_data["compression_degree"]

        }

        return output_dict
