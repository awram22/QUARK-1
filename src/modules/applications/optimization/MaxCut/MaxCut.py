
from typing import TypedDict
import pickle
from itertools import product, combinations

import networkx as nx
import numpy as np

from modules.applications.Application import *
from modules.applications.optimization.MaxCut.encodings.PauliCorrelationsEncoding import PauliCorrelationsEncoding
from modules.applications.optimization.Optimization import Optimization
from utils import start_time_measurement, end_time_measurement


class MaxCut(Optimization):


    def __init__(self):
        """
        Constructor method
        """
        super().__init__("MaxCut")
        self.submodule_options = ["PauliCorrelationsEncoding"]

    @staticmethod
    def get_requirements() -> list:
        return [
            {
                "name": "networkx",
                "version": "2.8.8"
            },
            {
                "name": "numpy",
                "version": "1.23.5"
            }
        ]

    def get_solution_quality_unit(self) -> str:
        return "Tour cost"
    def get_default_submodule(self, option: str) -> PauliCorrelationsEncoding:

        if option == "PauliCorrelationsEncoding":
            return PauliCorrelationsEncoding()
        else:
            raise NotImplementedError(
                f"Circuit Option {option} not implemented")
        
    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this application

        :return:
                 .. code-block:: python

                      return {
                                "n_qubits": {
                                "values": [4, 6, 8, 10, 12],
                                "description": "How many qubits do you want to use?"
                                }
                            }

        """
        return {
            "n_qubits": {
                "values": [13, 17, 19, 24],
                "description": "How many qubits do you want to use?"
            }
        }



    def generate_problem(self, config) -> dict:
        """
        Uses the reference graph to generate a problem for a given config.

        :param config:
        :type config: dict
        :return: n_qubits
        :rtype: dict
        """
        adjacency_matrix = MaxCut.load_graph_to_numpy_array("/Users/admin/Documents/marwamarso/QUARK-1/src/modules/applications/optimization/MaxCut/data/G9.txt")
        n_qubits = config["n_qubits"]

        application_config = {"n_qubits": n_qubits, "adjacency_matrix": adjacency_matrix}
        
        return application_config
    
    
    def load_graph_to_numpy_array(file_path):
        # Read the file and store the edges
        with open(file_path, 'r') as file:
            first_line = file.readline()
            num_vertices, num_edges = map(int, first_line.split())

            # Initialize the adjacency matrix with zeros
            # Assuming the graph is 1-indexed, we add 1 to the size so that indices match
            adjacency_matrix = np.zeros((num_vertices + 1, num_vertices + 1), dtype=int)

            # Read the rest of the lines which contain the edges
            for line in file:
                # Split the line into vertex1, vertex2, and weight
                vertex1, vertex2, weight = map(int, line.split())
                # Since it's an undirected graph, set the value for both [vertex1][vertex2] and [vertex2][vertex1]
                adjacency_matrix[vertex1][vertex2] = weight
                adjacency_matrix[vertex2][vertex1] = weight

            return adjacency_matrix



    def validate(self, solution: list) -> (bool, float):
        """
        Checks if it is a valid TSP tour.

        :param solution: list containing the nodes of the solution
        :type solution: list
        :return: Boolean whether the solution is valid, time it took to validate
        :rtype: tuple(bool, float)
        """
        start = start_time_measurement()
        nodes = self.application.nodes()

        if solution is None:
            return False, end_time_measurement(start)
        elif len([node for node in list(nodes) if node not in solution]) == 0:
            logging.info(f"All {len(solution)} nodes got visited")
            return True, end_time_measurement(start)
        else:
            logging.error(f"{len([node for node in list(nodes) if node not in solution])} nodes were NOT visited")
            return False, end_time_measurement(start)

    def evaluate(self, solution: list) -> (float, float):
        """
        Find distance for given route e.g. [0, 4, 3, 1, 2] and original data.

        :param solution:
        :type solution: list
        :return: Tour cost and the time it took to calculate it
        :rtype: tuple(float, float)
        """
        start = start_time_measurement()
        # get the total distance without return
        total_dist = 0
        for idx, _ in enumerate(solution[:-1]):
            dist = self.application[solution[idx + 1]][solution[idx]]
            total_dist += dist['weight']

        logging.info(f"Total distance (without return): {total_dist}")

        # add distance between start and end point to complete cycle
        return_distance = self.application[solution[0]][solution[-1]]['weight']
        # logging.info('Distance between start and end: ' + return_distance)

        # get distance for full cycle
        distance_with_return = total_dist + return_distance
        logging.info(f"Total distance (including return): {distance_with_return}")

        return distance_with_return, end_time_measurement(start)

    def save(self, path: str, iter_count: int) -> None:
        with open(f"{path}/graph_iter_{iter_count}.gpickle", "wb") as file:
            pickle.dump(self.application, file, pickle.HIGHEST_PROTOCOL)
