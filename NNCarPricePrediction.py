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

import os.path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import json

class CarPricePredictor:
    def __init__(self):
        self.mil = {}
        self.price = {}
        self.fuelType = {}
        self.transmission = {}
        self.ownership = {}
        self.manufacture = {}
        self.engine = {}
        self.seats = {}

    """
    Normalize the array using min-max scaling..

    Parameters:
    - ary: List of values.

    Returns:
    - list: Normalized Array.
    """
    def normalizeMinMaxScaling(self, ary):
        if len(ary) > 0:
            _max = max(ary)
            _min = min(ary)
            new_ary = []
            for item in ary:
                new_ary.append(self.normalizeMinMaxScalingValue(item, _min, _max))
            return new_ary, _min, _max
        return ary, 0, 0

    def normalizeMinMaxScalingValue(self, val, minimum, maximum):
        return (val-minimum)/(maximum-minimum)

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

        valid = True
        item = []
        try:
            if self.useMilage:
                __milage = self.__processMilage(milage)
                valid = valid and validateMilage(__milage)
                item.append(self.normalizeMinMaxScalingValue(__milage, self.kms_min, self.kms_max))
        except Exception as e:
            print(e)

        try:
            if self.useFuelType:
                __fuelType = self.__processFuelType(fuelType)
                item.append(self.normalizeMinMaxScalingValue(__fuelType, self.fuelType_min, self.fuelType_max))
        except Exception as e:
            print(e)

        try:
            if self.useTransmission:
                __transmission = self.__processTransmission(transmission)
                item.append(self.normalizeMinMaxScalingValue(__transmission, self.transmission_min, self.transmission_max))
        except Exception as e:
            print(e)

        try:
            if self.useOwnership:
                __ownership = self.__processOwnership(ownership)
                item.append(self.normalizeMinMaxScalingValue(__ownership, self.ownership_min, self.ownership_max))
        except Exception as e:
            print(e)

        try:
            if self.useManufacture:
                __manufacture = self.__processManufacture(manufactureYear)
                valid = valid and validateManufactureYear(__manufacture)
                item.append(self.normalizeMinMaxScalingValue(__manufacture, self.manufacture_min, self.manufacture_max))
        except Exception as e:
            print(e)

        try:
            if self.useEngine:
                __engine = self.__processEngine(engine)
                valid = valid and validateEngine(__engine)
                item.append(self.normalizeMinMaxScalingValue(__engine, self.engine_min, self.engine_max))
        except Exception as e:
            print(e)

        try:
            if self.useSeats:
                __seats = self.__processSeats(seats)
                item.append(self.normalizeMinMaxScalingValue(__seats, self.seats_min, self.seats_max))
        except Exception as e:
            valid = False
            print(e)

        if valid:
            print("$")
            print(item)
            inputTens = tf.constant([item], dtype=tf.float32)
            return self.model.predict(inputTens)[0][0]
        else:
            print("Invalid Input")

        return None

    def __learn(self,
        useMilage = True,
        useFuelType = True,
        useTransmission = True,
        useOwnership = True,
        useManufacture = True,
        useEngine = True,
        useSeats = True
        ):

        self.useMilage = useMilage
        self.useFuelType = useFuelType
        self.useTransmission = useTransmission
        self.useOwnership = useOwnership
        self.useManufacture = useManufacture
        self.useEngine = useEngine
        self.useSeats = useSeats

        # Car Name,Car Prices (In rupee),kms Driven,Fuel Type,Transmission,Ownership, Manufacture,Engine,Seats
        preprocessed_data = []
        for i in range(0, len(self.data)):
            # self.data[i][0] = processCarName(self.data[i][0])
            carPrice = self.__processCarPrice(self.data[i][1])
            milage = self.__processMilage(self.data[i][2])
            fuelType = self.__processFuelType(self.data[i][3])
            transmission = self.__processTransmission(self.data[i][4])
            ownership = self.__processOwnership(self.data[i][5])
            manufacture = self.__processManufacture(self.data[i][6])
            engine = self.__processEngine(self.data[i][7])
            seats = self.__processSeats(self.data[i][8])

            valid = None != carPrice
            valid = valid and None != milage
            valid = valid and None != fuelType
            valid = valid and None != transmission
            valid = valid and None != ownership
            valid = valid and None != manufacture
            valid = valid and None != engine
            valid = valid and None != seats

            if valid:
                self.Y.append([carPrice])
                preprocessed_data.append(
                    [
                        # self.data[i][0], # Car Name
                        # carPrice, # Car Prices (In rupee)
                        milage, # kms Driven
                        fuelType, # Fuel Type
                        transmission, # Transmission
                        ownership, # Ownership
                        manufacture, # Manufacture
                        engine, # Engine
                        seats, # Seats
                    ])

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

        kms, self.kms_min, self.kms_max                            = self.normalizeMinMaxScaling(kms)
        fuelType, self.fuelType_min, self.fuelType_max             = self.normalizeMinMaxScaling(fuelType)
        transmission, self.transmission_min, self.transmission_max = self.normalizeMinMaxScaling(transmission)
        ownership, self.ownership_min, self.ownership_max          = self.normalizeMinMaxScaling(ownership)
        manufacture, self.manufacture_min, self.manufacture_max    = self.normalizeMinMaxScaling(manufacture)
        engine, self.engine_min, self.engine_max                   = self.normalizeMinMaxScaling(engine)
        seats, self.seats_min, self.seats_max                      = self.normalizeMinMaxScaling(seats)

        if len(kms) == len(fuelType) and len(kms) == len(transmission) and len(kms) == len(ownership) and len(kms) == len(engine) and len(kms) == len(seats):
            for idx in range(0, len(kms)):
                item = []
                if self.useMilage:
                    item.append(kms[idx])
                if self.useFuelType:
                    item.append(fuelType[idx])
                if self.useTransmission:
                    item.append(transmission[idx])
                if self.useOwnership:
                    item.append(ownership[idx])
                if self.useManufacture:
                    item.append(manufacture[idx])
                if self.useEngine:
                    item.append(engine[idx])
                if self.useSeats:
                    item.append(seats[idx])
                self.X.append(item)

            def createModelFilename():
                path = "data/_"
                if self.useMilage:
                    path += "milage_"
                if self.useFuelType:
                    path += "fueltype_"
                if self.useTransmission:
                    path += "transmission_"
                if self.useOwnership:
                    path += "ownership_"
                if self.useManufacture:
                    path += "manufacture_"
                if self.useEngine:
                    path += "engine_"
                if self.useSeats:
                    path += "seats_"
                path += "model"
                return path

            path = createModelFilename() + ".keras"
            self.model = None
            if os.path.isfile(path):
                self.model = tf.keras.models.load_model(path)

                jsonPath = createModelFilename() + ".txt"
                with open(jsonPath, 'r') as file:
                    dictionary = json.load(file)

                    self.useMilage = dictionary["useMilage"]
                    self.useFuelType = dictionary["useFuelType"]
                    self.useTransmission = dictionary["useTransmission"]
                    self.useOwnership = dictionary["useOwnership"]
                    self.useManufacture = dictionary["useManufacture"]
                    self.useEngine = dictionary["useEngine"]
                    self.useSeats = dictionary["useSeats"]
            else:
                X = tf.constant(self.X, dtype=tf.float32)
                Y = tf.constant(self.Y, dtype=tf.float32)

                # Create a new Sequential Model
                self.model = keras.Sequential()

                # Add our layers
                def countDimensions():
                    count = 0
                    if self.useMilage:
                        count = count + 1

                    if self.useFuelType:
                        count = count + 1

                    if self.useTransmission:
                        count = count + 1

                    if self.useOwnership:
                        count = count + 1

                    if self.useManufacture:
                        count = count + 1

                    if self.useEngine:
                        count = count + 1

                    if self.useSeats:
                        count = count + 1

                    return count

                self.model.add(layers.Dense(
                    14, # Amount of Neurons
                    input_dim=countDimensions(), # Define an input dimension because this is the first layer
                    activation='relu' # Use relu activation function because all inputs are positive
                ))
                self.model.add(layers.Dense(
                    14, # Amount of Neurons. We want one output
                    activation='relu' # Use sigmoid because we want to output a binary classification
                ))

                self.model.add(layers.Dense(
                    1, # Amount of Neurons. We want one output
                    activation='linear' # Use sigmoid because we want to output a binary classification
                ))

                # Compile our layers into a model

                self.model.compile(
                    loss='mean_squared_error', # The loss function that is being minimized
                    optimizer='adam', # Our optimization function
                    metrics=['accuracy'] # Metrics are different values that you want the model to track while training
                )

                self.model.fit(
                    X, # Input training data
                    Y, # Output training data
                    batch_size=3,
                    epochs=2000, # Amount of iterations we want to train for
                    verbose=1 # Amount of detail you want shown in terminal while training
                )
                self.model.save(path)
                jsonPath = createModelFilename() + ".txt"
                with open(jsonPath, 'w') as file:
                    dictionary = {
                        "useMilage":self.useMilage,
                        "useFuelType":self.useFuelType,
                        "useTransmission":self.useTransmission,
                        "useOwnership":self.useOwnership,
                        "useManufacture":self.useManufacture,
                        "useEngine":self.useEngine,
                        "useSeats":self.useSeats
                    }
                    json.dump(dictionary, file, indent=2)


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
        if len(words) != 2:
            return None
        var = int(words[0].replace(',', ''))
        lbl = words[1]
        if "kms" != lbl:
            return None

        if lbl in self.mil.keys():
            self.mil[lbl] += 1
        else:
            self.mil[lbl] = 1

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
        if isinstance(var, str):
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
        if len(words) != 2:
            return None
        var = int(words[0])
        lbl = words[1]

        if lbl in self.engine.keys():
            self.engine[lbl] += 1
        else:
            self.engine[lbl] = 1

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

    def process(self,
        useMilage = True,
        useFuelType = True,
        useTransmission = True,
        useOwnership = True,
        useManufacture = True,
        useEngine = True,
        useSeats = True
        ):

        self.X = []
        self.Y = []
        self.__learn(
            useMilage,
            useFuelType,
            useTransmission,
            useOwnership,
            useManufacture,
            useEngine,
            useSeats
            )

def predict(predictor, arrayItem):
    # ['BMW 5 Series 520d M Sport', '31.90 Lakh', '42,000 kms', 'Diesel', 'Automatic', '2nd Owner', '2017', '1991 cc', '5 Seats']
    milage = 42_000
    fuelType = CarPricePredictor.FuelType.diesel
    transmission = CarPricePredictor.Transmission.automatic
    ownership = CarPricePredictor.Ownership._2ndowner
    manufactureYear = 2017
    engine = 1991
    seats = CarPricePredictor.Seats._5seats

    print(arrayItem)
    # ['BMW 5 Series 520d M Sport', '31.90 Lakh', '42,000 kms', 'Diesel', 'Automatic', '2nd Owner', '2017', '1991 cc', '5 Seats']
    milage = arrayItem[2]
    fuelType = arrayItem[3]
    transmission = arrayItem[4]
    ownership = arrayItem[5]
    manufactureYear = arrayItem[6]
    engine = arrayItem[7]
    seats = arrayItem[8]
    price = predictor.predictPrice(milage, fuelType, transmission, ownership, manufactureYear, engine, seats)

    return price

def main():
    predictor = CarPricePredictor()
    predictor.open("Data.csv")
    predictor.process(useEngine = False)

    print(predictor.X[-1])
    print(predictor.Y[-1])

    # 42000 1 2 2 2017 1991 5
    # 38280.0
    price = predict(predictor, predictor.data[-1])
    print("predicted price: " + str(price))





if __name__=="__main__":
    main()


