# James Folk
# DATA 527 – Predictive Modeling
# Assignment 4
# DEADLINE: April 11, 2024
# Spring 2024

import numpy as np
from scipy.stats import zscore
import csv
import datetime
from enum import Enum 

class CarPricePredictor:
    def __init__(self):
        self.price = {}
        self.fuelType = {}
        self.transmission = {}
        self.ownership = {}
        self.manufacture = {}
        self.engine = {}
        self.seats = {}

    def open(self, filename):
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            self.data = []
            for row in reader:
                self.data.append(row)

    def predictPrice(self, milage, fuelType, transmission, ownership, manufactureYear, engine, seats):
        def validateMilage(var):
            if var < self.kms_min:
                return False
            if var > self.kms_max:
                return False
            return True
        def validateFuelType(var):
            if var < self.fuelType_min:
                return False
            if var > self.fuelType_max:
                return False
            return True
        def validateTransmission(var):
            if var < self.transmission_min:
                return False
            if var > self.transmission_max:
                return False
            return True
        def validateOwnership(var):
            if var < self.ownership_min:
                return False
            if var > self.ownership_max:
                return False
            return True
        def validateManufactureYear(var):
            if var < self.manufacture_min:
                return False
            if var > self.manufacture_max:
                return False
            return True
        def validateEngine(var):
            if var < self.engine_min:
                return False
            if var > self.engine_max:
                return False
            return True
        def validateSeats(var):
            if var < self.seats_min:
                return False
            if var > self.seats_max:
                return False
            return True

        valid = validateMilage(milage)
        valid = valid and validateFuelType(fuelType)
        valid = valid and validateTransmission(fuelType)
        valid = valid and validateOwnership(fuelType)
        valid = valid and validateManufactureYear(fuelType)
        valid = valid and validateEngine(fuelType)
        valid = valid and validateSeats(fuelType)

        if valid:
            return 0.0
        return None 

    def __processCarName(self, var):
        return var

    def __processCarPrice(self, var):
        if "car prices (in rupee)" == var.strip().lower():
            return None
        words = str(var).split(' ')
        if len(words) != 2:
            return None

        var = float(words[0])
        scale = words[1]
        if "Lakh" == scale:
            var = var * 100_000
        if "Crore" == scale:
            var = var * 10_000_000

        # 0.012 United States Dollar
        conversion = 0.012
        var = var * conversion

        if scale in self.price.keys():
            self.price[scale] += 1
        else:
            self.price[scale] = 1

        return var

    def __processMilage(self, var):
        if "kms driven" == var.strip().lower():
            return None

        words = var.split(' ')
        var = int(words[0].replace(',', ''))

        return var

    def __processFuelType(self, var):
        if "fuel type" == var.strip().lower():
            return None

        if var in self.fuelType.keys():
            self.fuelType[var] += 1
        else:
            self.fuelType[var] = 1

        return self.FuelType[var.lower()].value
        # {'Fuel Type': 1, 'Diesel': 2423, 'Petrol': 2967, 'Cng': 80, 'Electric': 14, 'Lpg': 28}

    class FuelType(Enum):
        diesel = 1
        petrol = 2
        cng = 3
        electric = 4
        lpg = 5

    def __processTransmission(self, var):
        if "transmission" == var.strip().lower():
            return None

        if var in self.transmission.keys():
            self.transmission[var] += 1
        else:
            self.transmission[var] = 1

        return self.Transmission[var.lower()].value
        # {'Transmission': 1, 'Manual': 3962, 'Automatic': 1550}

    class Transmission(Enum):
        manual = 1
        automatic = 2

    def __processOwnership(self, var):
        if "ownership" == var.strip().lower():
            return None

        if var in self.ownership.keys():
            self.ownership[var] += 1
        else:
            self.ownership[var] = 1
        idx = "_" + var.replace(" ", "")
        if "_0thOwner" == idx:
            return None
        # {'Ownership': 1, '1st Owner': 3736, '2nd Owner': 1314, '3rd Owner': 359, '4th Owner': 84, '5th Owner': 12, '0th Owner': 7}
        return self.Ownership[idx.lower()].value

    class Ownership(Enum):
        _1stowner = 1
        _2ndowner = 2
        _3rdowner = 3
        _4thowner = 4
        _5thowner = 5

    def __processManufacture(self, var):
        if "manufacture" == var.strip().lower():
            return None

        if var in self.manufacture.keys():
            self.manufacture[var] += 1
        else:
            self.manufacture[var] = 1

        today = datetime.date.today()

        return today.year - int(var)

    def __processEngine(self, var):
        if "engine" == var.strip().lower():
            return None

        words = var.split(' ')
        var = int(words[0])

        if var in self.engine.keys():
            self.engine[var] += 1
        else:
            self.engine[var] = 1

        return var

    def __processSeats(self, var):
        if "seats" == var.strip().lower():
            return None

        if var in self.seats.keys():
            self.seats[var] += 1
        else:
            self.seats[var] = 1

        # {'Seats': 1, '5 Seats': 4673, '6 Seats': 61, '7 Seats': 631, '4 Seats': 88, '8 Seats': 54, '2 Seats': 5}
        idx = "_" + var.replace(" ", "")
        return self.Seats[idx.lower()].value

    class Seats(Enum):
        _2seats = 2
        _4seats = 4
        _5seats = 5
        _6seats = 6
        _7seats = 7
        _8seats = 8


























    def process(self):

        preprocessed_data = []
        self.Y = []
        # Car Name,Car Prices (In rupee),kms Driven,Fuel Type,Transmission,Ownership, Manufacture,Engine,Seats
        for i in range(0, len(self.data)):
            # self.data[i][0] = processCarName(self.data[i][0])
            self.data[i][1] = self.__processCarPrice(self.data[i][1])
            # print(self.data[i][1])
            self.data[i][2] = self.__processMilage(self.data[i][2])
            # print(self.data[i][2])
            self.data[i][3] = self.__processFuelType(self.data[i][3])
            # print(self.data[i][3])
            self.data[i][4] = self.__processTransmission(self.data[i][4])
            self.data[i][5] = self.__processOwnership(self.data[i][5])
            self.data[i][6] = self.__processManufacture(self.data[i][6])
            self.data[i][7] = self.__processEngine(self.data[i][7])
            self.data[i][8] = self.__processSeats(self.data[i][8])

            if None != self.data[i][1] and None != self.data[i][2] and None != self.data[i][3] and None != self.data[i][4] and None != self.data[i][5] and None != self.data[i][6] and None != self.data[i][7] and None != self.data[i][8]:
                self.Y.append(self.data[i][1])
                preprocessed_data.append(
                    [
                        # self.data[i][0], # Car Name
                        # self.data[i][1], # Car Prices (In rupee)
                        self.data[i][2], # kms Driven
                        self.data[i][3], # Fuel Type
                        self.data[i][4], # Transmission
                        self.data[i][5], # Ownership
                        self.data[i][6], # Manufacture
                        self.data[i][7], # Engine
                        self.data[i][8], # Seats
                    ])

        """
        Normalize the array using min-max scaling..

        Parameters:
        - ary: List of values.

        Returns:
        - list: Normalized Array.
        """
        def normalizeMinMaxScaling(ary):
            if len(ary) > 0:
                _max = max(ary)
                _min = min(ary)
                new_ary = []
                for item in ary:
                    new_ary.append((item-_min)/(_max-_min))
                return new_ary, _min, _max
            return ary, 0, 0

        kms = []
        fuelType = []
        transmission = []
        ownership = []
        manufacture = []
        engine = []
        seats = []
        for d in preprocessed_data:
            kms.append(d[0])
            fuelType.append(d[1])
            transmission.append(d[2])
            ownership.append(d[3])
            manufacture.append(d[4])
            engine.append(d[5])
            seats.append(d[6])

        kms, self.kms_min, self.kms_max                            = normalizeMinMaxScaling(kms)
        fuelType, self.fuelType_min, self.fuelType_max             = normalizeMinMaxScaling(fuelType)
        transmission, self.transmission_min, self.transmission_max = normalizeMinMaxScaling(transmission)
        ownership, self.ownership_min, self.ownership_max          = normalizeMinMaxScaling(ownership)
        manufacture, self.manufacture_min, self.manufacture_max    = normalizeMinMaxScaling(manufacture)
        engine, self.engine_min, self.engine_max                   = normalizeMinMaxScaling(engine)
        seats, self.seats_min, self.seats_max                      = normalizeMinMaxScaling(seats)

        self.X = []
        if len(kms) == len(fuelType) and len(kms) == len(transmission) and len(kms) == len(ownership) and len(kms) == len(engine) and len(kms) == len(seats):
            for idx in range(0, len(kms)):
                self.X.append(
                    [
                        kms[idx],
                        fuelType[idx],
                        transmission[idx],
                        ownership[idx],
                        manufacture[idx],
                        engine[idx],
                        seats[idx],
                    ])


        # print(self.price)
        # print(self.fuelType)
        # print(self.transmission)
        # print(self.ownership)
        # print(self.manufacture)
        # print(self.engine)
        # print(self.seats)




def main():
    loader = CarPricePredictor()
    loader.open("Data.csv")
    loader.process()
    print(loader.X)
    print(loader.Y)

if __name__=="__main__":
    main()


