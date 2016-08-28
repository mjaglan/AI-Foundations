# coding=utf-8
import numpy as np
import random
import math
import sys
import Queue
import operator

class Network:
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def derrivative_sigmoid(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def __init__(self, sizes):
        self.sizes = sizes

        self.in_to_hid_wts = [] # 30 rows x 192 cols   :: size[1] x size[0]
        for i in xrange(self.sizes[1]):
            w = [random.uniform(-0.9,+0.9) for i in xrange(self.sizes[0])]
            self.in_to_hid_wts.append(list(w)) # 30 rows x 192 cols   :: size[1] x size[0]

        self.hid_to_out_wts = [] # 4 rows x 30 cols    :: size[2] x size[1]
        for i in xrange(self.sizes[2]):
            w = [random.uniform(-0.9,+0.9) for i in xrange(self.sizes[1])]
            self.hid_to_out_wts.append(list(w)) # 4 rows x 30 cols    :: size[2] x size[1]

        self.hid_layer_activations = [0] * (self.sizes[1]) # 30
        self.out_layer_activations = [0] * (self.sizes[2]) # 4

    def feed_forward(self, example):
        # Input Layer: Feed Forward
        ai = []
        for i in range(0,self.sizes[0],1):
           ai.append(example[i])

        # Hidden Layer: Feed Forward
        # compute hidden layer activations
        for hid_node_idx in xrange(self.sizes[1]):
            node_input = 0.0
            for in_node_idx in xrange(self.sizes[0]):
                node_input += (self.in_to_hid_wts[hid_node_idx][in_node_idx] * ai[in_node_idx]) # see here 1
            self.hid_layer_activations[hid_node_idx] = self.sigmoid(node_input)

        # Output Layer: Feed Forward
        # compute output layer activations
        for out_node_idx in xrange(self.sizes[2]):
            node_input = 0.0
            for hid_node_idx in xrange(self.sizes[1]):
                node_input += (self.hid_to_out_wts[out_node_idx][hid_node_idx] * self.hid_layer_activations[hid_node_idx])
            self.out_layer_activations[out_node_idx] = self.sigmoid(node_input)

    def test(self, data):
        outputs = []
        for d in data:
            self.feed_forward(d)
            value = max(self.out_layer_activations)
            degree = self.out_layer_activations.index(value) * 90
            outputs.append(degree)

        return outputs

    def backpropagate2(self, examples, alpha=1, epochs=9):
        # Step #1: random initialization of weights
        self.in_to_hid_wts = [] # 30 rows x 192 cols   :: size[1] x size[0]
        for i in xrange(self.sizes[1]):
            w = [random.uniform(-0.9,+0.9) for i in xrange(self.sizes[0])]
            self.in_to_hid_wts.append(list(w)) # 30 rows x 192 cols   :: size[1] x size[0]

        self.hid_to_out_wts = [] # 4 rows x 30 cols    :: size[2] x size[1]
        for i in xrange(self.sizes[2]):
            w = [random.uniform(-0.9,+0.9) for i in xrange(self.sizes[1])]
            self.hid_to_out_wts.append(list(w)) # 4 rows x 30 cols    :: size[2] x size[1]

        epoch = 0
        while epoch < epochs:
            epoch += 1

            # Step #2: for each example in examples
            for degree, example in examples:
                t_v = [0]*4
                t_v[degree/90] = 1

                # Feed Forward
                # Input Layer: Feed Forward
                ai = []
                for i in range(0,self.sizes[0],1):
                   ai.append(example[i])

                # Hidden Layer: Feed Forward
                # compute hidden layer activations
                for hid_node_idx in xrange(self.sizes[1]):
                    node_input = 0.0
                    for in_node_idx in xrange(self.sizes[0]):
                        node_input += (self.in_to_hid_wts[hid_node_idx][in_node_idx] * ai[in_node_idx]) # see here 1
                    self.hid_layer_activations[hid_node_idx] = self.sigmoid(node_input)

                # Output Layer: Feed Forward
                # compute output layer activations
                for out_node_idx in xrange(self.sizes[2]):
                    node_input = 0.0
                    for hid_node_idx in xrange(self.sizes[1]):
                        node_input += (self.hid_to_out_wts[out_node_idx][hid_node_idx] * self.hid_layer_activations[hid_node_idx])
                    self.out_layer_activations[out_node_idx] = self.sigmoid(node_input)


                # Step #3: Propagate deltas backward from output layer to input layer
                # output layer delta
                output_deltas = []
                for out_node_idx in xrange(self.sizes[2]):
                    sum = 0
                    for hid_node_idx in xrange(self.sizes[1]):
                        sum += self.hid_to_out_wts[out_node_idx][hid_node_idx] * self.hid_layer_activations[hid_node_idx]
                    output_deltas.append(self.derrivative_sigmoid(sum) * (t_v[out_node_idx] - self.out_layer_activations[out_node_idx]))

                # hidden layer delta
                hidden_deltas = []
                for hid_node_idx in xrange(self.sizes[1]):
                    sum = 0
                    for in_node_idx in xrange(self.sizes[0]):
                        sum += self.in_to_hid_wts[hid_node_idx][in_node_idx] * ai[in_node_idx]

                    wij = 0
                    for j in xrange(self.sizes[2]):
                        wij += self.hid_to_out_wts[j][hid_node_idx] * output_deltas[j]
                    hidden_deltas.append(self.derrivative_sigmoid(sum) * (wij))


                # Step #4: Update every weight in network using deltas
                # update hidden layer weights
                for in_node_idx in xrange(self.sizes[0]):
                    for hid_node_idx in xrange(self.sizes[1]):
                        change = alpha * example[in_node_idx] * hidden_deltas[hid_node_idx]
                        self.in_to_hid_wts[hid_node_idx][in_node_idx] -= change

                # update output layer weights
                for hid_node_idx in xrange(self.sizes[1]):
                    for out_node_idx in xrange(self.sizes[2]):
                        change = alpha * self.hid_layer_activations[hid_node_idx] * output_deltas[out_node_idx]
                        self.hid_to_out_wts[out_node_idx][hid_node_idx] -= change


def readDataFromFile(fname):
    exemplars = []
    names = []
    file = open(fname, 'r')
    for line in file:
        data = line.split()
        names.append(data[0])
        exemplar = map(int, data[2:])
        exemplar = [x / 255.0 for x in exemplar] # normalize the data
        true_value = int(data[1])%360
        exemplars.append((true_value, exemplar))

    return names, exemplars


def writeOutput_PrintAnalysis(fn, test_names, predictionList, correctLabels):
    assert len(predictionList) == len(correctLabels)
    f = open(fn, 'w')
    correct = 0.0
    for i in xrange(len(predictionList)):
        f.write(test_names[i] + " " + str(predictionList[i]) + "\n")
        if correctLabels[i] == predictionList[i]:
            correct += 1.0
        else:
            pass
    f.close()
    accuracy = (correct / len(predictionList))*100 # percentage
    print("\n Accuracy: " + str(accuracy) + " %" )


def runNnet(algoName, inputNodesCount, kHiddenCount, modelFile, train_data, test_data_features, outputFileName, test_names, test_labels):
    sizes = [inputNodesCount, kHiddenCount, 4]
    n2 = Network(sizes)
    n2.backpropagate2(examples=train_data, alpha=0.0001, epochs=1)
    outputList = n2.test(test_data_features)
    writeOutput_PrintAnalysis(outputFileName, test_names, outputList, test_labels)


def main():
    trainFileName = sys.argv[1]    # case is preserved
    testFileName  = sys.argv[2]    # case is preserved
    algoName = sys.argv[3].lower() # algo name in lower case
    kHiddenCount = int(sys.argv[4])

    ##### Get Data ####################################################################################################
    train_names, train_data = readDataFromFile(trainFileName)
    test_names, test_data = readDataFromFile(testFileName)

    test_data_features = [x[1] for x in test_data]
    test_labels = [x[0] for x in test_data] # true value

    inputNodesCount = (len(train_data[0][1]))
    outputFileName = str(algoName)+"_output.txt"

    ##### NNET ########################################################################################################
    runNnet(algoName=algoName, inputNodesCount=inputNodesCount, kHiddenCount=kHiddenCount, modelFile=None,
            train_data=train_data, test_data_features=test_data_features, outputFileName=outputFileName,
            test_names=test_names, test_labels=test_labels)

# python  orient.py  train-data.txt  test-data.txt   nnet       30
if __name__ == '__main__':
    main()


