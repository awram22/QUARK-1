
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

    def generate_gate_sequence(self, input_data: dict, config: dict) -> dict:
        """
        Returns gate sequence of a hardware-efficient circuit in brickwork architecture.
        """
        n_qubits = input_data["n_qubits"]
        adjaceny_matrix = input_data['adjaceny_matrix']
        depth = config["depth"]

        gate_sequence = []
        # Cycle through the rotation axes X, Y, Z for each layer
        rotation_sequence = ["RX", "RY", "RZ"]

        # Brickwork architecture with single-qubit rotations followed by MS gates
        for d in range(depth):
            # Single-qubit rotations layer, all qubits rotate around the same axis
            rotation_type = rotation_sequence[d % 3]
            for q in range(n_qubits):
                gate_sequence.append((rotation_type, [q]))

            # Add a barrier after each single-qubit gate layer
            gate_sequence.append(("Barrier", []))

            # MS two-qubit gates layer
            # For even number of qubits, pair them up
            for q in range(0, n_qubits - 1, 2):
                gate_sequence.append(("RXX", [q, q + 1]))
            # For odd number of qubits, you'll need to define the entanglement pattern

            # Add a barrier after each two-qubit gate layer
            gate_sequence.append(("Barrier", []))

        # Measurement layer
        for q in range(n_qubits):
            gate_sequence.append(("Measure", [q]))

        output_dict = {
            "gate_sequence": gate_sequence,
            "circuit_name": "BrickworkArchitecture",
            "n_qubits": n_qubits,
            "depth": depth,
            "adjacency_matrix": adjaceny_matrix,
            "pauli_strings": input_data["pauli_strings"],
            'compression_degree':input_data["compression_degree"]
        }

        return output_dict
