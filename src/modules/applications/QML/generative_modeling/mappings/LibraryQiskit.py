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

from typing import Union
import logging
 
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import Aer
from qiskit.compiler import transpile, assemble
from qiskit.providers import Backend
import numpy as np

from modules.training.QCBM import QCBM
from modules.training.Inference import Inference
from modules.training.Classical import Classical
from modules.applications.QML.generative_modeling.mappings.Library import Library

logging.getLogger("qiskit").setLevel(logging.WARNING)


class LibraryQiskit(Library):
    """
    This module maps a library-agnostic gate sequence to a qiskit circuit
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__("LibraryQiskit")
        self.submodule_options = ["QCBM", "Inference", "Classical"]

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module

        :return: list of dict with requirements of this module
        :rtype: list[dict]
        """
        return [
            {
                "name": "qiskit",
                "version": "0.45.0"
            },
            # {
            #    "name": "qiskit_ibm_runtime",
            #      "version": "0.10.0"
            # },
            {
                "name": "numpy",
                "version": "1.23.5"
            },
            {
                "name": "qiskit-ibmq-provider",
                "version": "0.19.2"
            }
        ]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for the Qiskit Library.

        :return:
                 .. code-block:: python

                        return {
            "backend": {
                "values": ["aer_statevector_simulator_gpu", "aer_statevector_simulator_cpu",
                           "cusvaer_simulator (only available in cuQuantum applicance)", "aer_simulator_gpu",
                           "aer_simulator_cpu", "ionQ_Harmony", "Amazon_SV1",
                           "simulator_statevector IBM Quantum Platform", "ibm_brisbane IBM Quantum Platform"],
                "description": "Which backend do you want to use? (aer_statevector_simulator\
                             uses the measurement probability vector, the others are shot based)"
            },

            "n_shots": {
                "values": [100, 1000, 10000, 1000000],
                "description": "How many shots do you want use for estimating the PMF of the model?\
                 (If the aer_statevector_simulator selected, only relevant for studying generalization)"
            }
        }

        """
        return {
            "backend": {
                "values": ["aer_statevector_simulator_gpu", "aer_statevector_simulator_cpu",
                           "cusvaer_simulator (only available in cuQuantum applicance)", "aer_simulator_gpu",
                           "aer_simulator_cpu", "ionQ_Harmony", "Amazon_SV1", "ibm_brisbane IBM Quantum Platform"],
                "description": "Which backend do you want to use? (aer_statevector_simulator\
                             uses the measurement probability vector, the others are shot based)"
            },
            #TODO Discuss: Use IBM Eagle (so 1 of 3) or IBM Brisbane so one specific?

            "n_shots": {
                "values": [100, 1000, 10000, 1000000],
                "description": "How many shots do you want use for estimating the PMF of the model?\
                 (If the aer_statevector_simulator selected, only relevant for studying generalization)"
            }
        }

    def get_default_submodule(self, option: str) -> Union[QCBM, Inference, Classical]:

        if option == "QCBM":
            return QCBM()
        if option == "Classical":
            return Classical()
        elif option == "Inference":
            return Inference()
        else:
            raise NotImplementedError(f"Option {option} not implemented")

    def sequence_to_circuit(self, input_data: dict) -> dict:
        """
        Maps the gate sequence, that specifies the architecture of a quantum circuit
        to its Qiskit implementation. 

        :param input_data: Collected information of the benchmarking process
        :type input_data: dict
        :return: Same dictionary but the gate sequence is replaced by it Qiskit implementation
        :rtype: dict
        """
        n_qubits = input_data["n_qubits"]
        gate_sequence = input_data["gate_sequence"]

        circuit = QuantumCircuit(n_qubits, n_qubits)
        param_counter = 0
        for gate, wires in gate_sequence:

            if gate == "Hadamard":
                circuit.h(wires[0])

            elif gate == "CNOT":
                circuit.cx(wires[0], wires[1])
            
            elif gate == "Sdg":
                circuit.sdg(wires[0])

            elif gate == "RZ":
                circuit.rz(Parameter(f"x_{param_counter:03d}"), wires[0])
                param_counter += 1

            elif gate == "RX":
                circuit.rx(Parameter(f"x_{param_counter:03d}"), wires[0])
                param_counter += 1

            elif gate == "RY":
                circuit.ry(Parameter(f"x_{param_counter:03d}"), wires[0])
                param_counter += 1

            elif gate == "RXX":
                circuit.rxx(Parameter(f"x_{param_counter:03d}"), wires[0], wires[1])
                param_counter += 1

            elif gate == "RYY":
                circuit.ryy(Parameter(f"x_{param_counter:03d}"), wires[0], wires[1])
                param_counter += 1

            elif gate == "RZZ":
                circuit.rzz(Parameter(f"x_{param_counter:03d}"), wires[0], wires[1])
                param_counter += 1

            elif gate == "CRY":
                circuit.cry(Parameter(f"x_{param_counter:03d}"), wires[0], wires[1])
                param_counter += 1

            elif gate == "Barrier":
                circuit.barrier()

            elif gate == "Measure":
                circuit.measure(wires[0], wires[0])
            
            else:
                raise NotImplementedError(f"Gate {gate} not implemented")

        input_data["circuit"] = circuit
        input_data.pop("gate_sequence")
        return input_data

    @staticmethod
    def select_backend(config: str) -> dict:
        """
        This method configures the backend

        :param config: Name of a backend
        :type config: str
        :return: Configured qiskit backend
        :rtype: qiskit.providers.Backend
        """
        if config == "cusvaer_simulator (only available in cuQuantum applicance)":
            import cusvaer # pylint: disable=C0415
            from qiskit.providers.aer import AerSimulator # pylint: disable=C0415
            backend = AerSimulator(
                method="statevector",
                device="GPU",
                cusvaer_enable=True,
                noise_model=None,
                cusvaer_p2p_device_bits=3,
                cusvaer_comm_plugin_type=cusvaer.CommPluginType.MPI_AUTO,
                cusvaer_comm_plugin_soname="libmpi.so"
            )

        elif config == "aer_simulator_gpu":
            from qiskit_aer import AerSimulator
            backend = AerSimulator()
            backend.set_options(device="GPU")

        elif config == "aer_simulator_cpu":
            from qiskit_aer import AerSimulator
            backend = AerSimulator.get_backend("aer_simulator")
            backend.set_options(device="CPU")

        elif config == "aer_statevector_simulator_gpu":
            from qiskit_aer import AerSimulator
            backend = AerSimulator.get_backend('statevector_simulator')
            backend.set_options(device="GPU")

        elif config == "aer_statevector_simulator_cpu":
            from qiskit_aer import AerSimulator
            backend = AerSimulator(method='statevector')
            # Set the backend options to use the CPU
            backend.set_options(device='CPU')
            Aer.get_backend('aer_simulator')
            #backend = AerSimulator.get_backend('statevector_simulator')
            #backend.set_options(device="CPU")

        # elif config == "simulator_statevector IBM Quantum Platform":
            # define token here once
            # service = QiskitRuntimeService(token='TOKEN')
            # service = QiskitRuntimeService()
            # backend = service.get_backend('simulator_statevector')

        # elif config == "ibm_brisbane IBM Quantum Platform":
            #define token here once
            #service = QiskitRuntimeService(token='TOKEN')
            # service = QiskitRuntimeService()
            # backend = service.get_backend('ibm_brisbane')

        elif config == "ionQ_Harmony":
            from modules.devices.braket.Ionq import Ionq # pylint: disable=C0415
            from qiskit_braket_provider import AWSBraketBackend, AWSBraketProvider # pylint: disable=C0415
            device_wrapper = Ionq("ionQ", "arn:aws:braket:::device/qpu/ionq/ionQdevice")
            backend = AWSBraketBackend(
                device=device_wrapper.device,
                provider=AWSBraketProvider(),
                name=device_wrapper.device.name,
                description=f"AWS Device: {device_wrapper.device.provider_name} {device_wrapper.device.name}.",
                online_date=device_wrapper.device.properties.service.updatedAt,
                backend_version="2",
            )

        elif config == "Amazon_SV1":
            from modules.devices.braket.SV1 import SV1 # pylint: disable=C0415
            from qiskit_braket_provider import AWSBraketBackend, AWSBraketProvider # pylint: disable=C0415
            device_wrapper = SV1("SV1", "arn:aws:braket:::device/quantum-simulator/amazon/sv1")
            backend = AWSBraketBackend(
                device=device_wrapper.device,
                provider=AWSBraketProvider(),
                name=device_wrapper.device.name,
                description=f"AWS Device: {device_wrapper.device.provider_name} {device_wrapper.device.name}.",
                online_date=device_wrapper.device.properties.service.updatedAt,
                backend_version="2",
            )

        else:
            raise NotImplementedError(f"Device Configuration {config} not implemented")

        return backend

    @staticmethod
    def get_execute_circuit(circuit: QuantumCircuit, backend: Backend, config: str, config_dict: dict) \
            -> callable:  # pylint: disable=W0221,R0915
        """
        This method combines the qiskit circuit implementation and the selected backend and returns a function,
        that will be called during training.

        :param circuit: Qiskit implementation of the quantum circuit
        :type circuit: qiskit.circuit.QuantumCircuit
        :param backend: Configured qiskit backend
        :type backend: qiskit.providers.Backend
        :param config: Name of a backend
        :type config: str
        :param config_dict: Contains information about config
        :type config_dict: dict
        :return: Method that executes the quantum circuit for a given set of parameters
        :rtype: callable
        """
        n_shots = config_dict["n_shots"]
        n_qubits = circuit.num_qubits
        circuit_transpiled = transpile(circuit, backend=backend)

        if config in ["aer_statevector_simulator_gpu", "aer_statevector_simulator_cpu"]:
            circuit_transpiled.remove_final_measurements()

            def execute_circuit(solutions):
                all_circuits = [circuit_transpiled.bind_parameters(solution) for solution in solutions]
                qobj = assemble(all_circuits, backend=backend)
                jobs = backend.run(qobj)
                job_result = [jobs.result().get_statevector(circuit) for circuit in all_circuits]
                return job_result, None

        elif config in ["ionQ_Harmony", "Amazon_SV1"]:
            import time as timetest # pylint: disable=C0415
            def execute_circuit(solutions):
                all_circuits = [circuit_transpiled.bind_parameters(solution) for solution in solutions]
                jobs = backend.run(all_circuits, shots=n_shots)
                while not jobs.in_final_state():
                    logging.info("Waiting 10 seconds for task to finish")
                    timetest.sleep(10)

                samples_dictionary = [jobs.result().get_counts(circuit).int_outcomes() for circuit in all_circuits]

                samples = []
                for result in samples_dictionary:
                    target_iter = np.zeros(2 ** n_qubits)
                    result_keys = list(result.keys())
                    result_vals = list(result.values())
                    target_iter[result_keys] = result_vals
                    target_iter = np.asarray(target_iter)
                    samples.append(target_iter)

                samples = np.asarray(samples)
                pmfs = samples / n_shots

                return pmfs, samples

        elif config in ["cusvaer_simulator (only available in cuQuantum applicance)", "aer_simulator_cpu",
                                "aer_simulator_gpu"]:
            def execute_circuit(solutions):
                all_circuits = [circuit_transpiled.bind_parameters(solution) for solution in solutions]
                qobjs = assemble(all_circuits, backend=backend)
                jobs = backend.run(qobjs, shots=n_shots)
                samples_dictionary = [jobs.result().get_counts(circuit).int_outcomes() for circuit in all_circuits]

                samples = []
                for result in samples_dictionary:
                    target_iter = np.zeros(2 ** n_qubits)
                    result_keys = list(result.keys())
                    result_vals = list(result.values())
                    target_iter[result_keys] = result_vals
                    target_iter = np.asarray(target_iter)
                    samples.append(target_iter)

                samples = np.asarray(samples)
                pmfs = samples / n_shots
                return pmfs, samples

        return execute_circuit

    @staticmethod
    def Y_measurement(qc,qubit,cbit):
        qc.sdg(qubit)
        qc.h(qubit)
        qc.measure(qubit,cbit)
        return qc
    
    
    @staticmethod
    def X_measurement(qc,qubit,cbit):
        qc.h(qubit)
        qc.measure(qubit,cbit)
        return qc
