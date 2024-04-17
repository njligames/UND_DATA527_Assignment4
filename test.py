# project/test.py

import unittest

from NNCarPricePrediction import Tester

class TestCalculations(unittest.TestCase):

    def test_calculation(self):
        tester = Tester()
        tester.test()

if __name__ == '__main__':
    unittest.main()

