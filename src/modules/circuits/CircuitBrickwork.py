
from typing import TypedDict

from modules.circuits.Circuit import Circuit
from modules.applications.QML.generative_modeling.mappings.LibraryQiskit import LibraryQiskit


class CircuitBrickwork(Circuit):
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
        Returns gate sequence of a hardware-efficient circuit in brickwork architecture.

        :param input_data: Collection of information from the previous modules
        :type input_data: dict
        :param config: Config specifying the number of qubits and depth of the circuit
        :type config: Config
        :return: Dictionary including the gate sequence of the Hardware Efficient Circuit
        :rtype: dict
        """
        n_qubits = input_data["n_qubits"]
        depth = config["depth"]

        gate_sequence = []

        # Initialize rotation sequence (X, Y, Z)
        rotation_sequence = ["RX", "RY", "RZ"]

        # Brickwork architecture with single-qubit rotations followed by MS gates
        for d in range(depth):
            # Single-qubit rotations layer
            for k in range(n_qubits):
                rotation_type = rotation_sequence[k % 3]  # Cycle through X, Y, Z
                gate_sequence.append((rotation_type, [k]))

            # MS two-qubit gates layer
            for k in range(0, n_qubits - 1, 2):
                gate_sequence.append(("RXX", [k, k + 1]))  # Assuming MS gate with 3 parameters
            gate_sequence.append(("Barrier", None))  # Barrier after each layer

        # Measurement layer
        for k in range(n_qubits):
            gate_sequence.append(("Measure", [k, k]))

        output_dict = {
            "gate_sequence": gate_sequence,
            "circuit_name": "BrickworkArchitecture",
            "n_qubits": n_qubits,
            "depth": depth,
            "pauli_strings": [],  # Assuming no Pauli strings for now
            "adjacency_matrix": None  # Assuming no adjacency matrix for now
        }

        return output_dict
