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

        manufacture = self.__processManufacture(manufactureYear)
        valid = validateMilage(milage)
        valid = valid and validateManufactureYear(manufacture)
        valid = valid and validateEngine(engine)
        # valid = valid and validateSeats(seats)

        if valid:
            try:
                _fuelType = CarPricePredictor.FuelType(fuelType).value
                _transmission = CarPricePredictor.Transmission(transmission).value
                _ownership = CarPricePredictor.Ownership(ownership).value
                _seats = CarPricePredictor.Seats(seats).value

                _input = [
                    self.normalizeMinMaxScalingValue(milage, self.kms_min, self.kms_max),
                    self.normalizeMinMaxScalingValue(_fuelType, self.fuelType_min, self.fuelType_max),
                    self.normalizeMinMaxScalingValue(_transmission, self.transmission_min, self.transmission_max),
                    self.normalizeMinMaxScalingValue(_ownership, self.ownership_min, self.ownership_max),
                    self.normalizeMinMaxScalingValue(manufacture, self.manufacture_min, self.manufacture_max),
                    self.normalizeMinMaxScalingValue(engine, self.engine_min, self.engine_max),
                    self.normalizeMinMaxScalingValue(_seats, self.seats_min, self.seats_max)
                    ]
                inputTens = tf.constant([_input], dtype=tf.float32)
                print(inputTens)
                ans = self.model.predict(inputTens)
                return ans

            except Exception as e:
                print(e)
            return None
        return None

    def __learn(self):
        path = 'my_model.keras'
        self.model = None
        if os.path.isfile(path):
            self.model = tf.keras.models.load_model(path)
        else:
            X = tf.constant(self.X, dtype=tf.float32)
            Y = tf.constant(self.Y, dtype=tf.float32)

            # Create a new Sequential Model
            self.model = keras.Sequential()

            # Add our layers

            self.model.add(layers.Dense(
                14, # Amount of Neurons
                input_dim=7, # Define an input dimension because this is the first layer
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

            # print(self.Y)
            # X = tf.constant(self.X, dtype=tf.float32)
            # Y = tf.constant(self.Y, dtype=tf.float32)

            # # # Define the input layer
            # input_layer = tf.keras.Input(shape=(7,))

            # # # Define the hidden layers
            # hidden_layer_1 = tf.keras.layers.Dense(128, activation='relu')(input_layer)
            # hidden_layer_2 = tf.keras.layers.Dense(128, activation='relu')(hidden_layer_1)
            # hidden_layer_3 = tf.keras.layers.Dense(128, activation='relu')(hidden_layer_2)
            # hidden_layer_4 = tf.keras.layers.Dense(128, activation='relu')(hidden_layer_3)

            # # # Define the output layer
            # output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(hidden_layer_4)

            # # # Compile the model
            # self.model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
            # self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

            # # # Train the model
            # self.model.fit(X, Y, epochs=20)

            # # # Evaluate the model
            # print(self.model.evaluate(X, Y))
            # # self.model.save(path)


            # Define our training input and output data with type 16 bit float
            # Each input maps to an output

            # X = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=tf.float16)
            # Y = tf.constant([[0], [1], [1], [0]], dtype=tf.float16)
            # X = tf.constant(self.X, dtype=tf.float16)
            # Y = tf.constant(self.Y, dtype=tf.float16)

            # Create a new Sequential Model
            # self.model = keras.Sequential()

            # Add our layers

            # self.model.add(layers.Dense(
            #     14, # Amount of Neurons
            #     input_dim=7, # Define an input dimension because this is the first layer
            #     activation='relu' # Use relu activation function because all inputs are positive
            # ))

            # self.model.add(layers.Dense(
            #     1, # Amount of Neurons. We want one output
            #     activation='sigmoid' # Use sigmoid because we want to output a binary classification
            # ))



            # Compile our layers into a model

            # self.model.compile(
            #     loss='mean_squared_error', # The loss function that is being minimized
            #     optimizer='adam', # Our optimization function
            #     metrics=['binary_accuracy'] # Metrics are different values that you want the model to track while training
            # )

            # self.model.fit(
            #     X, # Input training data
            #     Y, # Output training data
            #     epochs=10, # Amount of iterations we want to train for
            #     verbose=0 # Amount of detail you want shown in terminal while training
            # )

            # self.model.save(path)
            # print(self.model.evaluate(X, Y))

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
                self.Y.append([self.data[i][1]])
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
            self.__learn()

        # print(self.mil)
        # print(self.price)
        # print(self.fuelType)
        # print(self.transmission)
        # print(self.ownership)
        # print(self.manufacture)
        # print(self.engine)
        # print(self.seats)





def main():
    predictor = CarPricePredictor()
    predictor.open("Data.csv")
    predictor.process()
    # print(predictor.X)
    # print(predictor.Y)

    # ['BMW 5 Series 520d M Sport', '31.90 Lakh', '42,000 kms', 'Diesel', 'Automatic', '2nd Owner', '2017', '1991 cc', '5 Seats']
    milage = 42_000
    fuelType = CarPricePredictor.FuelType.diesel
    transmission = CarPricePredictor.Transmission.automatic
    ownership = CarPricePredictor.Ownership._2ndowner
    manufactureYear = 2017
    engine = 1991
    seats = CarPricePredictor.Seats._5seats
    price = predictor.predictPrice(milage, fuelType, transmission, ownership, manufactureYear, engine, seats)

    # 42000 1 2 2 2017 1991 5
    # 38280.0
    print("#############")
    print(price)
    print("#############")

if __name__=="__main__":
    main()


