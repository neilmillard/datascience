from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        #seed the random number generator, so the same numbers
        # are generated
        random.seed(1)

        # We model a single neuron, with 3 input connections and 1 output connection
        #we assign random weights to a 3 x 1 matrix with values in range -1 to 1
        # and a mean of 0
        self.synaptic_weights = 2 * random.random((3,1)) - 1

    #The sigmoid function
    # s shaped curve. pass the weighted sum of the inputs to normalise them between
    # 0 and 1
    def __sigmoid(self, x):
        return 1/(1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1-x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_it):
        for iteration in xrange(number_of_training_it):
            #pass the training set through our neural net
            output = self.predict(training_set_inputs)

            # calculate the error
            error = training_set_outputs - output

            #multiply the error by the input and again by the gradient of the sigmoid curve
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            #adjust the weights
            self.synaptic_weights += adjustment

    def predict(self, inputs):
        #pass inputs through our neural network (single neuron)
        return self.__sigmoid(dot(inputs, self.synaptic_weights))



if __name__ == '__main__':
    #initialise single neuron neural net
    neural_network = NeuralNetwork()
    print ("Random starting weights:")
    print (neural_network.synaptic_weights)

    #training set, 4 examples, 3 inputs with 1 output
    training_set_inputs = array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
    training_set_outputs = array([[0,1,1,0]]).T

    #train the neural network using a traingin set#
    #Do it 10,000 times and make small adjustments
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print ('New synaptic weights after training:')
    print (neural_network.synaptic_weights)

    #Test the neural network
    print ('Considering [1,0,0] -> ?: ')
    print (neural_network.predict(array([1,0,0])))

    print ('Considering [0,1,0] -> ?: ')
    print (neural_network.predict(array([0,1,0])))