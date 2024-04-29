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
            if None != milage and self.useMilage:
                item.append(milage)
        except Exception as e:
            print(e)

        try:
            if None != fuelType and self.useFuelType:
                item.append(fuelType)
        except Exception as e:
            print(e)

        try:
            if None != transmission and self.useTransmission:
                item.append(transmission)
        except Exception as e:
            print(e)

        try:
            if None != ownership and self.useOwnership:
                item.append(ownership)
        except Exception as e:
            print(e)

        try:
            if None != manufactureYear and self.useManufacture:
                item.append(manufactureYear)
        except Exception as e:
            print(e)

        try:
            if None != engine and self.useEngine:
                item.append(engine)
        except Exception as e:
            print(e)

        try:
            if None != seats and self.useSeats:
                item.append(seats)
        except Exception as e:
            valid = False
            print(e)

        if valid:
            inputTens = tf.constant([item], dtype=tf.float32)
            return self.model.predict(inputTens, verbose=0)[0][0]
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
                    verbose=0 # Amount of detail you want shown in terminal while training
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

def loadRValues(path):
    r_values = None
    if os.path.isfile(path):
        with open(path, 'r') as openfile:
            r_values = json.load(openfile)
    if None == r_values:
        r_values = []
    return r_values

"""
Calculate coorelation coefficient..

Parameters:
- y_try: List of actual values.
- y_pred: List of predicted values.

Returns:
- r_squared: The coorelation coefficient.
"""
def calculate_r_squared(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Length of dependant values and predicted values must be the same.")

    # Calculate the mean of the true values
    mean_y_true = sum(y_true) / len(y_true)

    # Calculate the total sum of squares (TSS) without using sum
    tss = 0
    for y in y_true:
        tss += (y - mean_y_true) ** 2

    # Calculate the residual sum of squares (RSS) without using sum
    rss = 0
    for true_val, pred_val in zip(y_true, y_pred):
        rss += (true_val - pred_val) ** 2

    # Calculate R-squared
    r_squared = 1 - (rss / tss)

    return r_squared

def predictModels(predict_prices=False):

    def appendRValue(path, r_values, val):
        r_values.append(val)
        with open(path, "w") as outfile:
            json.dump(r_values, outfile)

    predictor = CarPricePredictor()
    predictor.open("Data.csv")

    path = "data/r_values.json"
    r_values = loadRValues(path)

    start = 0
    if predict_prices:
        start = len(r_values) 

    k = 0
    for i in range(start, 128):

        useMilage = False
        useFuelType = False
        useTransmission = False
        useOwnership = False
        useManufacture = False
        useEngine = False
        useSeats = False

        if (i & (1<<0)) != 0:
            useMilage = True

        if (i & (1<<1)) != 0:
            useFuelType = True

        if (i & (1<<2)) != 0:
            useTransmission = True

        if (i & (1<<3)) != 0:
            useOwnership = True

        if (i & (1<<4)) != 0:
            useManufacture = True

        if (i & (1<<5)) != 0:
            useEngine = True

        if (i & (1<<6)) != 0:
            useSeats = True

        predictor.process(useMilage=useMilage, useFuelType=useFuelType, useTransmission=useTransmission, useOwnership=useOwnership, useManufacture=useManufacture, useEngine=useEngine, useSeats=useSeats)

        if not predict_prices:
            continue

        actual_prices = []
        for item in predictor.Y:
            actual_prices.append(item[0])

        predicted_prices = []
        for j in range(0, len(predictor.X)):

            print("({}/{}) - ({}/{});".format(i, 127, j+1, len(predictor.X)))
            k = k + 1

            idx = 0
            if useMilage:
                milage = predictor.X[j][idx]
                idx = idx + 1
            else:
                milage = None

            if useFuelType:
                fuelType = predictor.X[j][idx]
                idx = idx + 1
            else:
                fuelType = None

            if useTransmission:
                transmission = predictor.X[j][idx]
                idx = idx + 1
            else:
                transmission = None

            if useOwnership:
                ownership = predictor.X[j][idx]
                idx = idx + 1
            else:
                ownership = None

            if useManufacture:
                manufactureYear = predictor.X[j][idx]
                idx = idx + 1
            else:
                manufactureYear = None

            if useEngine:
                engine = predictor.X[j][idx]
                idx = idx + 1
            else:
                engine = None

            if useSeats:
                seats = predictor.X[j][idx]
                idx = idx + 1
            else:
                seats = None

            price = predictor.predictPrice(milage, fuelType, transmission, ownership, manufactureYear, engine, seats)
            predicted_prices.append(price)

        val = {
            "index":i,
            "r_value":calculate_r_squared(actual_prices, predicted_prices),
            "useMilage":useMilage,
            "useFuelType":useFuelType,
            "useTransmission":useTransmission,
            "useOwnership":useOwnership,
            "useManufacture":useManufacture,
            "useEngine":useEngine,
            "useSeats":useSeats
        }
        appendRValue(path, r_values, val)

def writeCSV():
    path = "data/r_values.json"
    r_values = loadRValues(path)

    # field names
    fields = ['useMilage', 'useFuelType', 'useTransmission', "useOwnership", "useManufacture", "useEngine", "useSeats", "Coorelation Coefficient"]
     
    # name of csv file
    filename = "data/r_values.csv"
     
    mydict = []
    for item in r_values:
        val = {
            "useMilage":item["useMilage"], 
            "useFuelType":item["useFuelType"], 
            "useTransmission":item["useTransmission"], 
            "useOwnership":item["useOwnership"], 
            "useManufacture":item["useManufacture"], 
            "useEngine":item["useEngine"], 
            "useSeats":item["useSeats"], 
            "Coorelation Coefficient":item["r_value"]
            }
        mydict.append(val)

    # writing to csv file
    with open(filename, 'w') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=fields)
     
        # writing headers (field names)
        writer.writeheader()
     
        # writing data rows
        writer.writerows(mydict)

def useBestCoorelationCoefficient():
    # useMilage  useFuelType useTransmission useOwnership    useManufacture  useEngine   useSeats    Coorelation Coefficient 
    # TRUE       TRUE        TRUE            FALSE           TRUE            TRUE        FALSE       0.449060499
    predictor = CarPricePredictor()
    predictor.open("Data.csv")

    useMilage = True
    useFuelType = True
    useTransmission = True
    useOwnership = False
    useManufacture = True
    useEngine = True
    useSeats = False

    predictor.process(useMilage=useMilage, useFuelType=useFuelType, useTransmission=useTransmission, useOwnership=useOwnership, useManufacture=useManufacture, useEngine=useEngine, useSeats=useSeats)

    actual_prices = []
    for item in predictor.Y:
        actual_prices.append(item[0])

    predicted_prices = []
    for j in range(0, len(predictor.X)):

        print("({}/{});".format(j+1, len(predictor.X)))

        idx = 0
        if useMilage:
            milage = predictor.X[j][idx]
            idx = idx + 1
        else:
            milage = None

        if useFuelType:
            fuelType = predictor.X[j][idx]
            idx = idx + 1
        else:
            fuelType = None

        if useTransmission:
            transmission = predictor.X[j][idx]
            idx = idx + 1
        else:
            transmission = None

        if useOwnership:
            ownership = predictor.X[j][idx]
            idx = idx + 1
        else:
            ownership = None

        if useManufacture:
            manufactureYear = predictor.X[j][idx]
            idx = idx + 1
        else:
            manufactureYear = None

        if useEngine:
            engine = predictor.X[j][idx]
            idx = idx + 1
        else:
            engine = None

        if useSeats:
            seats = predictor.X[j][idx]
            idx = idx + 1
        else:
            seats = None

        price = predictor.predictPrice(milage, fuelType, transmission, ownership, manufactureYear, engine, seats)
        predicted_prices.append(price)

    r_squared = calculate_r_squared(actual_prices, predicted_prices)
    print(r_squared)

    predictor.model.summary()

def main():
    # predictModels()
    # writeCSV()
    useBestCoorelationCoefficient()


if __name__=="__main__":
    main()


