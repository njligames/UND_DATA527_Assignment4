# project/test.py

import unittest

# from NNCarPricePrediction import Tester
import json
import os
import random

def loadRValues(path):
    r_values = None
    if os.path.isfile(path):
        with open(path, 'r') as openfile:
            r_values = json.load(openfile)
    if None == r_values:
        r_values = []


    return r_values

def appendRValue(path, r_values, val):
    r_values.append(val)
    with open(path, "w") as outfile:
        json.dump(r_values, outfile)

class TestCalculations(unittest.TestCase):

    def test_json_functions(self):
        path = "data/data.json"
        r_values = loadRValues(path)
        val = {
            "index":len(r_values),
            "r_value":random.random(),
            "useMilage":False,
            "useFuelType":False,
            "useTransmission":False,
            "useOwnership":False,
            "useManufacture":False,
            "useEngine":False,
            "useSeats":False
        }
        appendRValue(path, r_values, val)


if __name__ == '__main__':
    unittest.main()

