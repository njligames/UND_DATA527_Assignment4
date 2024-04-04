# James Folk
# DATA 527 – Predictive Modeling
# Assignment 4
# DEADLINE: April 11, 2024
# Spring 2024

from matplotlib import pyplot
import math
from math import cos, sin, atan
import os
import random
import matplotlib.pyplot as plt;
import json

## View

class NeuronView():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, neuron_radius):
        font = {
            'family': 'serif',
            'color':  'black',
            'weight': 'normal',
            'size': 8,
        }
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)
        pyplot.text(self.x, self.y,  self.get_neuron_text(), horizontalalignment='center', verticalalignment='center', fontdict=font)

    def get_neuron_text(self):
        return "0.0"#"{}\n{}".format(self.x, self.y)

class LayerView():
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer):
        self.scale = 1
        self.vertical_distance_between_layers = 6 * self.scale
        self.horizontal_distance_between_neurons = 2 * self.scale
        self.neuron_radius = 0.5 * self.scale
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)

    def __create_neuron(self):
        return NeuronView(self.x, self.y)

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        self.x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = self.__create_neuron()
            neurons.append(neuron)
            self.x += self.horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return self.horizontal_distance_between_neurons * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)
        line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment), (neuron1.y - y_adjustment, neuron2.y + y_adjustment))
        pyplot.gca().add_line(line)

    def draw(self, layerType=0):
        for neuron in self.neurons:
            neuron.draw( self.neuron_radius )
            if self.previous_layer:
                for previous_layer_neuron in self.previous_layer.neurons:
                    self.__line_between_two_neurons(neuron, previous_layer_neuron)
        # write Text
        x_text = self.number_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons
        if layerType == 0:
            pyplot.text(x_text, self.y, 'Input Layer', fontsize = 12)
        elif layerType == -1:
            pyplot.text(x_text, self.y, 'Output Layer', fontsize = 12)
        else:
            pyplot.text(x_text, self.y, 'Hidden Layer '+str(layerType), fontsize = 12)

class NeuralNetworkView():
    def __init__(self, number_of_neurons_in_widest_layer):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.layers = []
        self.layertype = 0

    def __create_layer(self, number_of_neurons):
        return LayerView(self, number_of_neurons, self.number_of_neurons_in_widest_layer)

    def add_layer(self, number_of_neurons ):
        layer = self.__create_layer(number_of_neurons)
        self.layers.append(layer)

    def draw(self):
        pyplot.figure()
        for i in range( len(self.layers) ):
            layer = self.layers[i]
            if i == len(self.layers)-1:
                i = -1
            layer.draw( i )
        pyplot.axis('scaled')
        pyplot.axis('off')
        pyplot.title( 'Neural Network architecture', fontsize=15 )
        pyplot.show()

## Model

class NeuronEdge():
    def __init__(self, neuron):
        self.neuron = neuron

class Neuron(NeuronView):
    def __init__(self, x, y):
        super().__init__(x, y)

        # neurons coming from previous layer
        self.parents = []

        # neurons going to the next layer
        self.children = []

        # weights between current neuron to child neuron
        self.weights = []

        self.bias = 0

    def __str__(self):
        return "{\"bias\":" + str(self.bias) + ",\"weights\":" + str(self.weights) + "}"

    def __repr__(self):
        return self.__str__()

    def get_number_of_children(self):
        return len(self.children)

    def get_child_neuron(self, index):
        if index >= 0 and index < len(self.children):
            return self.children[index].neuron
        return None

    def get_number_of_parents(self):
        return len(self.parents)

    def get_parent_neuron(self, index):
        if index >= 0 and index < len(self.parents):
            return self.parents[index].neuron
        return None

    def get_neuron_text(self):
        return "{}".format(self.bias)

    def calculate(self, idx_neuron, activation_func = None):
        current_sum = 0
        for idx in range(0, self.get_number_of_parents()):
            current_sum += self.get_parent_neuron(idx).weights[idx_neuron] * self.get_parent_neuron(idx).bias

        if None != activation_func:
            self.bias = activation_func(current_sum)
        else:
            self.bias = current_sum

class Layer(LayerView):
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer):
        super().__init__(network, number_of_neurons, number_of_neurons_in_widest_layer)
        self.__init_layer()

    def __str__(self):
        s = "{\"neurons\":  " + str(self.neurons) + " }"
        return s

    def __repr__(self):
        return self.__str__()

    def __init_layer(self):
        if None != self.previous_layer:
            for prev_neuron in self.previous_layer.neurons:
                for neuron in self.neurons:
                    prev_neuron.weights.append(random.random())

        for neuron in self.neurons:
            if None != self.previous_layer:
                # init children of previous layer neurons
                for prev_neuron in self.previous_layer.neurons:
                    prev_neuron.children.append(NeuronEdge(neuron))
                # init parents of current layer neurons
                for prev_neuron in self.previous_layer.neurons:
                    neuron.parents.append(NeuronEdge(prev_neuron))

    def _LayerView__create_neuron(self):
        return Neuron(self.x, self.y)

    def get_number_of_neurons(self):
        return len(self.neurons)

    def get_neuron(self, index):
        if index >= 0 and index < len(self.neurons):
            return self.neurons[index]
        return None



class NeuralNetwork(NeuralNetworkView):
    def __init__(self, number_of_neurons_in_widest_layer):
        super().__init__(number_of_neurons_in_widest_layer)

    def __str__(self):
        s = "{\"layers\":  " + str(self.layers) + " }"
        return s


    def _NeuralNetworkView__create_layer(self, number_of_neurons):
        return Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer)

    def get_number_of_layers(self):
        return len(self.layers)

    def get_layer(self, index):
        if index >= 0 and index < len(self.layers):
            return self.layers[index]
        return None

    def get_output_layer(self):
        return self.get_layer(self.get_number_of_layers() - 1)


class Model():
    def __init__( self, neural_network ):
        self.neural_network = neural_network
        widest_layer = max( self.neural_network )
        self.network = NeuralNetwork( widest_layer )
        for l in self.neural_network:
            self.network.add_layer(l)

    def __str__(self):
        return str(self.network)

    def draw( self ):
        self.network.draw()

    def get_input_neuron(self, index):
        if self.network.get_number_of_layers() > 0 and index >= 0 and index < self.network.get_layer(0).get_number_of_neurons():
            return self.network.get_layer(0).get_neuron(index)
        return None

    def feed_forward(self, inputs):
        ########################
        # feed the neural network with the input: the output of each node in the hidden
        # layers and the output layer is calculated
        ########################

        input_neurons = []
        input_neurons.append(self.get_input_neuron(0))
        input_neurons.append(self.get_input_neuron(1))

        input_neurons[0].bias = inputs[0]
        input_neurons[1].bias = inputs[1]

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        self.network.get_layer(1).get_neuron(0).calculate(0, sigmoid)
        self.network.get_layer(1).get_neuron(1).calculate(1, sigmoid)
        self.network.get_layer(2).get_neuron(0).calculate(0)

        output_layer = self.network.get_output_layer()
        neuron = output_layer.get_neuron(0)
        return neuron.bias

    def back_propagate(self, error, LR):
        ########################
        # backpropagate the error and calculate the derivative with the respect to each weight
        ########################

        output_layer = self.network.get_output_layer()
        neuron = output_layer.get_neuron(0)

        output_o1 = output_layer.get_neuron(0).bias
        w5        = neuron.get_parent_neuron(0).weights[0]
        w6        = neuron.get_parent_neuron(1).weights[0]
        output_h1 = neuron.get_parent_neuron(0).bias
        output_h2 = neuron.get_parent_neuron(1).bias
        output_i1 = neuron.get_parent_neuron(0).get_parent_neuron(0).bias
        output_i2 = neuron.get_parent_neuron(0).get_parent_neuron(1).bias

        # w5
        # delta_w5 = error * [output_o1 * (1 - output_o1)] * output_h1
        delta_w5 = error * (output_o1 * (1 - output_o1)) * output_h1

        # w6
        # delta_w6 = error * [output_o1 * (1 - output_o1)] * output_h2
        delta_w6 = error * (output_o1 * (1 - output_o1)) * output_h2

        # w1
        # deltaE_w1 = error * [output_o1 * (1 - output_o1)] * w5
        deltaE_w1 = error * (output_o1 * (1 - output_o1)) * w5
        # delta_w1 = deltaE_w1 * (output_h1 * (1 - output_h1)) * output_i1
        delta_w1 = deltaE_w1 * (output_h1 * (1 - output_h1)) * output_i1

        # w2
        # deltaE_w2 = error * [output_o1 * (1 - output_o1)] * w6
        deltaE_w2 = error * (output_o1 * (1 - output_o1)) * w6
        # delta_w2 = deltaE_w2 * (output_h2 * (1 - output_h2)) * output_i1
        delta_w2 = deltaE_w2 * (output_h2 * (1 - output_h2)) * output_i1

        # w3
        # deltaE_w3 = error * [output_o1 * (1 - output_o1)] * w5
        deltaE_w3 = error * (output_o1 * (1 - output_o1)) * w5
        # delta_w3 = deltaE_w3 * (output_h1 * (1 - output_h1)) * output_i2
        delta_w3 = deltaE_w3 * (output_h1 * (1 - output_h1)) * output_i2

        # w4
        # deltaE_w4 = error * [output_o1 * (1 - output_o1)] * w6
        deltaE_w4 = error * (output_o1 * (1 - output_o1)) * w6
        # delta_w4 = deltaE_w4 * (output_h2 * (1 - output_h2)) * output_i2
        delta_w4 = deltaE_w4 * (output_h2 * (1 - output_h2)) * output_i2

        ###################
        # update each weight
        ###################

        # w5_new = w5_old - (LR * delta_w5)
        neuron.get_parent_neuron(0).weights[0] = neuron.get_parent_neuron(0).weights[0] - (LR * delta_w5)
        # w6_new = w6_old - (LR * delta_w6)
        neuron.get_parent_neuron(1).weights[0] = neuron.get_parent_neuron(1).weights[0] - (LR * delta_w6)

        # w1_new = w1_old - (LR * delta_w1)
        neuron.get_parent_neuron(0).get_parent_neuron(0).weights[0] = neuron.get_parent_neuron(0).get_parent_neuron(0).weights[0] - (LR * delta_w1)
        # w2_new = w2_old - (LR * delta_w2)
        neuron.get_parent_neuron(0).get_parent_neuron(0).weights[1] = neuron.get_parent_neuron(0).get_parent_neuron(0).weights[1] - (LR * delta_w2)
        # w3_new = w3_old - (LR * delta_w3)
        neuron.get_parent_neuron(0).get_parent_neuron(1).weights[0] = neuron.get_parent_neuron(0).get_parent_neuron(1).weights[0] - (LR * delta_w3)
        # w4_new = w4_old - (LR * delta_w4)
        neuron.get_parent_neuron(0).get_parent_neuron(1).weights[1] = neuron.get_parent_neuron(0).get_parent_neuron(1).weights[1] - (LR * delta_w4)

    def __RMSE(self, dependantVariables, predictedArray):
        if len(dependantVariables) != len(predictedArray):
            raise ValueError("Length of dependant  values and predicted values must be the same.")

        m = len(dependantVariables)
        def step2Function(actual, predicted):
            return (predicted - actual) ** 2
        return math.sqrt(sum(step2Function(actual, predicted) for actual, predicted in zip(predictedArray, dependantVariables)) / (m))

    def iterate(self, inputs, outputs, LR):
        errors = [0] * len(inputs)
        predictedArray = [0] * len(inputs)
        for idx in range(0, len(inputs)):
            predictedArray[idx] = self.feed_forward(inputs[idx])
            errors[idx] = (outputs[idx] - predictedArray[idx])

        rmse_error = self.__RMSE(outputs, predictedArray)

        for error in errors:
            self.back_propagate(error, LR)

        return rmse_error

    def iterate_training(self, inputs, outputs, LR):

        predicted = outputs[0]

        self.feed_forward(inputs)

        ########################
        # calculate the error, which is the estimated – the actual value
        ########################
        output_layer = self.network.get_output_layer()
        neuron = output_layer.get_neuron(0)
        actual    = neuron.bias

        error = (predicted - actual)

        ########################
        # backpropagate the error and calculate the derivative with the respect to each weight
        ########################

        output_o1 = output_layer.get_neuron(0).bias
        w5        = neuron.get_parent_neuron(0).weights[0]
        output_h1 = neuron.get_parent_neuron(0).bias
        output_h2 = neuron.get_parent_neuron(1).bias
        output_i1 = neuron.get_parent_neuron(0).get_parent_neuron(0).bias
        output_i2 = neuron.get_parent_neuron(0).get_parent_neuron(1).bias

        # w5
        # delta_w5 = error * [output_o1 * (1 - output_o1)] * output_h1
        delta_w5 = error * (output_o1 * (1 - output_o1)) * output_h1

        # w6
        # delta_w6 = error * [output_o1 * (1 - output_o1)] * output_h2
        delta_w6 = error * (output_o1 * (1 - output_o1)) * output_h2

        # w1
        # deltaE_w1 = error * [output_o1 * (1 - output_o1)] * w5
        deltaE_w1 = error * (output_o1 * (1 - output_o1)) * neuron.get_parent_neuron(0).weights[0]
        # delta_w1 = deltaE_w1 * (output_h1 * (1 - output_h1)) * output_i1
        delta_w1 = deltaE_w1 * (output_h1 * (1 - output_h1)) * output_i1

        # w2
        # deltaE_w2 = error * [output_o1 * (1 - output_o1)] * w6
        deltaE_w2 = error * (output_o1 * (1 - output_o1)) * neuron.get_parent_neuron(1).weights[0]
        # delta_w2 = deltaE_w2 * (output_h2 * (1 - output_h2)) * output_i1
        delta_w2 = deltaE_w2 * (output_h2 * (1 - output_h2)) * output_i1

        # w3
        # deltaE_w3 = error * [output_o1 * (1 - output_o1)] * w5
        deltaE_w3 = error * (output_o1 * (1 - output_o1)) * neuron.get_parent_neuron(0).weights[0]
        # delta_w3 = deltaE_w3 * (output_h1 * (1 - output_h1)) * output_i2
        delta_w3 = deltaE_w3 * (output_h1 * (1 - output_h1)) * output_i2

        # w4
        # deltaE_w4 = error * [output_o1 * (1 - output_o1)] * w6
        deltaE_w4 = error * (output_o1 * (1 - output_o1)) * neuron.get_parent_neuron(1).weights[0]
        # delta_w4 = deltaE_w4 * (output_h2 * (1 - output_h2)) * output_i2
        delta_w4 = deltaE_w4 * (output_h2 * (1 - output_h2)) * output_i2

        ###################
        # update each weight
        ###################

        # w5_new = w5_old - (LR * delta_w5)
        neuron.get_parent_neuron(0).weights[0] = neuron.get_parent_neuron(0).weights[0] - (LR * delta_w5)
        # w6_new = w6_old - (LR * delta_w6)
        neuron.get_parent_neuron(1).weights[0] = neuron.get_parent_neuron(1).weights[0] - (LR * delta_w6)

        # w1_new = w1_old - (LR * delta_w1)
        neuron.get_parent_neuron(0).get_parent_neuron(0).weights[0] = neuron.get_parent_neuron(0).get_parent_neuron(0).weights[0] - (LR * delta_w1)
        # w2_new = w2_old - (LR * delta_w2)
        neuron.get_parent_neuron(0).get_parent_neuron(0).weights[1] = neuron.get_parent_neuron(0).get_parent_neuron(0).weights[1] - (LR * delta_w2)
        # w3_new = w3_old - (LR * delta_w3)
        neuron.get_parent_neuron(0).get_parent_neuron(1).weights[0] = neuron.get_parent_neuron(0).get_parent_neuron(1).weights[0] - (LR * delta_w3)
        # w4_new = w4_old - (LR * delta_w4)
        neuron.get_parent_neuron(0).get_parent_neuron(1).weights[1] = neuron.get_parent_neuron(0).get_parent_neuron(1).weights[1] - (LR * delta_w4)

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

def train_neural_network(independentVariablesArray, dependantVariables, LR = 0.001, N = 10000):
    model = Model( [2,2,1] )
    errorArray = []

    for n in range(N):
        error = model.iterate(independentVariablesArray, dependantVariables, LR)
        errorArray.append(error)
    return model, errorArray

def estimate_r_squared(model, independentVariablesArray, dependantVariables):
    predictedArray = []
    for independentVariable in independentVariablesArray:
        predictedArray.append(model.feed_forward(independentVariable))
    rValue = calculate_r_squared(dependantVariables, predictedArray)
    return rValue

def use_model(model, a, b):
    return model.feed_forward([a, b])

def main():
    independentVariablesArray = [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1]
            ]
    dependantVariables = [ 0, 1, 1, 0 ]

    LR = 0.001
    N = 10000
    model, errorArray = train_neural_network(independentVariablesArray, dependantVariables, LR, N)
    model.draw()

    plt.plot(errorArray)
    plt.savefig('data/BatchGradientDescent_MSE.pdf', dpi=150)
    plt.show()

    rValue = estimate_r_squared(model, independentVariablesArray, dependantVariables)

    with open("data/ModelParameters.json", 'w') as file:
        if [] == errorArray:
            dictionary={"learningRate":LR, "iterations":N, "final mse":errorArray, "r value":rValue, "neural network": json.loads(str(model))}
        else:
            dictionary={"learningRate":LR, "iterations":N, "final mse":errorArray[-1], "r value":rValue, "neural network": json.loads(str(model))}

        json.dump(dictionary, file, indent=2)

    i = 0
    for var in independentVariablesArray:
        print("Ran model, got {}, should be {}".format(use_model(model, var[0], var[1]), dependantVariables[i]))
        i = i + 1

if __name__=="__main__":
    main()

