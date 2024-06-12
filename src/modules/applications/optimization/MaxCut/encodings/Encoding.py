from itertools import product

import numpy as np

from modules.Core import *
from utils import start_time_measurement, end_time_measurement


class Mapping(Core, ABC):
    """
    The task of the transformation module is to translate data and problem specification of the application into
    preprocessed format.
    """

    def __init__(self, name):
        """
        Constructor method
        """
        super().__init__()
        self.transformation_name = name

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
            }
        ]

    def preprocess(self, input_data: dict, config: dict, **kwargs):
        """
        In this module, the preprocessing step is tansforming the data to the correct target format.

        :param input_data: Collected information of the benchmarking process
        :type input_data: dict
        :param config: Config specifying the parameters of the transormation
        :type config: dict
        :param kwargs:
        :type kwargs: dict
        :return: tuple with transformed problem and the time it took to map it
        :rtype: (dict, float)
        """

        start = start_time_measurement()
        output = self.encode(input_data, config)

        return output, end_time_measurement(start)

    def postprocess(self, input_data: dict, config: dict, **kwargs):
        """
        Does the reverse transformation

        :param input_data:
        :type input_data: dict
        :param config:
        :type config: dict
        :param kwargs:
        :type kwargs: dict
        :return:
        """
        start = start_time_measurement()

        output = self.reverse_transform(input_data)
        output["Transformation"] = True
        if "inference" in input_data:
            output["inference"] = input_data["inference"]
        return output, end_time_measurement(start)

    @abstractmethod
    def encode(self, input_data: dict, config: dict) -> dict:
        """
        Helps to ensure that the model can effectively learn the underlying 
        patterns and structure of the data, and produce high-quality outputs.

        :param input_data: Input data for transformation.
        :type input_data: dict
        :param config: Configuration parameters for the transformation.
        :type config: dict
        :return: Transformed data.
        :rtype: dict
        """
        pass