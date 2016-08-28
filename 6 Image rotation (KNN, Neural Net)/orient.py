# coding=utf-8
"""
KNN:
We have tried 2 distance functions, Euclidean and Manhattan. 

Neural Net:
We normalized training data, used bias weights for each node in both hidden and output layers. We used tanh as activation
function, as it is more flat than sigmoid, so should predict more accurately. We tried to learn faster by randomizing
input sequence. Also we reduced learning rate by a factor of .9 after each 5 iterations of training. We're saving our model file also.

"""

import numpy as np
import random
import math
import sys
import Queue
import operator
#import time

_ONE_ = 0 # Non-zero values are for test & trial purposes..


class Network:
    # the neuron
    def sigmoid(self, x):
        return math.tanh(x)

    def derrivative_sigmoid(self, y):
        return 1.0 - y ** 2

    def __init__(self, sizes):
        self.sizes = sizes

        self.in_to_hid_wts = []
        for i in xrange(self.sizes[0]):
            w = [random.uniform(-0.9,+0.9) for i in xrange(self.sizes[1]+ _ONE_ )]
            self.in_to_hid_wts.append(list(w))

        self.hid_to_out_wts = []
        for i in xrange(self.sizes[1]):
            w = [random.uniform(-0.9,+0.9) for i in xrange(self.sizes[2] + _ONE_ )]
            self.hid_to_out_wts.append(list(w))

        self.hid_layer_bias = [random.uniform(-0.9, +0.9) for i in xrange(self.sizes[1] + _ONE_ )]
        self.out_layer_bias = [random.uniform(-0.9, +0.9) for i in xrange(self.sizes[2] +  _ONE_)]

        self.hid_layer_activations = [1] * (self.sizes[1] +  _ONE_)
        self.out_layer_activations = [1] * (self.sizes[2] +  _ONE_)


    def print_net(self):
        print "Input to hidden weights"
        for w in self.in_to_hid_wts:
            print w
        print "\nHidden layer activations"
        print self.hid_layer_activations

        print "\nHidden to output weights"
        for w in self.hid_to_out_wts:
            print w
        print "\nOutput layer activations"
        print self.out_layer_activations

    def feed_forward(self, example):
        try:
            # compute hidden layer activations
            for hid_node_idx in xrange(self.sizes[1]):
                node_input = 0.0
                for in_node_idx in xrange(self.sizes[0]):
                    node_input += (self.in_to_hid_wts[in_node_idx][hid_node_idx] * example[in_node_idx]) # see here 1
                self.hid_layer_activations[hid_node_idx] = self.sigmoid(node_input + self.hid_layer_bias[hid_node_idx])
        except:
            # print ("ERROR: ", node_input, self.in_to_hid_wts[in_node_idx][hid_node_idx]  , example[in_node_idx])
            pass


        try:
            # compute output layer activations
            for out_node_idx in xrange(self.sizes[2]):
                node_input = 0.0
                for hid_node_idx in xrange(self.sizes[1]):
                    node_input += (self.hid_to_out_wts[hid_node_idx][out_node_idx] * \
                                   self.hid_layer_activations[hid_node_idx])
                self.out_layer_activations[out_node_idx] = self.sigmoid(node_input + self.out_layer_bias[out_node_idx])
        except:
            # print ("ERROR: ", node_input, self.hid_to_out_wts[hid_node_idx][out_node_idx], self.hid_layer_activations[hid_node_idx])
            pass


    def backpropagate(self, example, true_val, alpha):
        try:
            # compute delta for output layer
            output_deltas = []
            for out_node_idx in xrange(self.sizes[2]):
                output = self.out_layer_activations[out_node_idx]
                output_deltas.append(
                    self.derrivative_sigmoid(output) * (true_val[out_node_idx] - output))

            # compute hidden layer deltas
            hidden_deltas = []
            for hid_node_idx in xrange(self.sizes[1]):
                sum = 0
                for out_node_idx in xrange(self.sizes[2]):
                    sum += self.hid_to_out_wts[hid_node_idx][out_node_idx] * \
                           output_deltas[out_node_idx]
                hidden_deltas.append(sum *
                                     self.derrivative_sigmoid(self.hid_layer_activations[hid_node_idx]))

            # update hidden layer weights
            for in_node_idx in xrange(self.sizes[0]):
                for hid_node_idx in xrange(self.sizes[1]):
                    change = alpha * example[in_node_idx] * hidden_deltas[hid_node_idx]
                    self.in_to_hid_wts[in_node_idx][hid_node_idx] += change

            # update hidden layer bias weights
            for hid_node_idx in xrange(self.sizes[1]):
                self.hid_layer_bias[hid_node_idx] += alpha * hidden_deltas[hid_node_idx]

                # update output layer weights
            for hid_node_idx in xrange(self.sizes[1]):
                for out_node_idx in xrange(self.sizes[2]):
                    change = alpha * self.hid_layer_activations[hid_node_idx] * output_deltas[out_node_idx]
                    self.hid_to_out_wts[hid_node_idx][out_node_idx] += change

            # update output layer bias weights
            for out_node_idx in xrange(self.sizes[2]):
                self.out_layer_bias[out_node_idx] += alpha * output_deltas[out_node_idx]
        except:
            pass

    def train(self, examples, alpha=1, epochs=9, isDebug=False):
        epoch = 0
        while epoch < epochs:
            random.shuffle(examples)

            for t_v, e in examples:
                if isDebug:
                    print "before backprop"
                    self.print_net()

                self.feed_forward(e)
                self.backpropagate(e, t_v, alpha)

                if isDebug:
                    print "after backprop"
                    self.print_net()

            epoch += 1
            if epoch % 5 == 0:
                alpha = alpha * 9.0 / 10.0
                # print "Epoch:", epoch, "\tAlpha:", alpha

    def test(self, data):
        outputs = []
        for d in data:
            self.feed_forward(d)
            outputs.append(list(self.out_layer_activations))
        return outputs

    def store_weights(self, fname):
        f = open(fname, 'w')
        for w in self.in_to_hid_wts:
            f.write("%s\n" % w)
        for w in self.hid_layer_bias:
            f.write("%s\n" % w)
        for w in self.hid_to_out_wts:
            f.write("%s\n" % w)
        for w in self.out_layer_bias:
            f.write("%s\n" % w)

    def load_weights(self, fname):
        f = open(fname, 'r')
        lines = f.readlines()
        self.in_to_hid_wts = []
        self.hid_to_out_wts = []
        self.hid_layer_bias = []
        self.out_layer_bias = []

        l_c = 0
        for i in xrange(self.sizes[0]):
            line = lines[l_c]
            l_c += 1
            weights = line.replace("[", "").replace("]", "").replace("\n", "").split(',')
            self.in_to_hid_wts.append(map(float, weights))

        for i in xrange(self.sizes[1]):
            line = lines[l_c]
            l_c += 1
            weights = line.replace("[", "").replace("]", "").replace("\n", "").split(',')
            self.hid_layer_bias.append(map(float, weights))

        for i in xrange(self.sizes[1]):
            line = lines[l_c]
            l_c += 1
            weights = line.replace("[", "").replace("]", "").replace("\n", "").split(',')
            self.hid_to_out_wts.append(map(float, weights))

        for i in xrange(self.sizes[2]):
            line = lines[l_c]
            l_c += 1
            weights = line.replace("[", "").replace("]", "").replace("\n", "").split(',')
            self.out_layer_bias.append(map(float, weights))


def man_distance(a, b):
    s = 0
    for i in xrange(len(a)):
        s += math.fabs(a[i] - b[i])
    return s


def euclid_distance(a, b):
    s = 0
    for i in xrange(len(a)):
        s += math.pow(a[i] - b[i], 2)
    return math.sqrt(s)


def distance(a, b, v):
    if v == 1: return euclid_distance(a, b)
    if v == 2: return man_distance(a, b)


def knn(k, train_dataset, test_dataset, d_func=1):
    output = []

    for test_data in test_dataset:
        q = Queue.PriorityQueue()
        for label, train_data in train_dataset:
            q.put((distance(test_data[1], train_data, d_func), label))

        r = [0] * 4
        for n in xrange(k):
            t = q.get()
            r[t[1] / 90] += 1
        output.append(np.argmax(r) * 90)

        # print "output:", np.argmax(r) * 90
    return output


def knn_w(k, train_dataset, test_dataset, d_func=1):
    output = []

    for test_data in test_dataset:
        q = Queue.PriorityQueue()
        for label, train_data in train_dataset:
            q.put((distance(test_data[1], train_data, d_func), label))

        r = [0] * 4
        d = [0] * 4
        for n in xrange(k):
            t = q.get()
            r[t[1] / 90] += 1
            d[t[1] / 90] += t[0]

        for i in xrange(4):
            if d[i] == 0:
                continue
            r[i] /= d[i]
        output.append(np.argmax(r) * 90)

    return output


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
    # f1 = open("correct"+fn, 'ab+')
    # f2 = open("wrong"+fn, 'ab+')
    for i in xrange(len(predictionList)):
        f.write(test_names[i] + " " + str(predictionList[i]) + "\n")
        if correctLabels[i] == predictionList[i]:
            # f1.write(str(test_names[i])+ "  " + str(predictionList[i])+ " " + str(correctLabels[i]) + "\n")
            correct += 1.0
        else:
            # f2.write(str(test_names[i])+ "  " + str(predictionList[i])+ "  " + str(correctLabels[i]) + "\n")
            pass
    f.close()
    # f1.close()
    # f2.close()
    accuracy = (correct / len(predictionList))*100 # percentage
    print("\n Accuracy: " + str(accuracy) + " %" )
    confusionMatrix(predictionList, correctLabels)



"""
The confusion matrix is a table with four rows and four columns. The entry at cell i,j in the table should
show the number of test exemplars whose correct label is i, but that were classified as j

(e.g. a 10 in the second column of the third row would mean that 10 test images were actually oriented at 270 degrees,
but were incorrectly classified by your classier as being at 180 degrees).
"""
def confusionMatrix(predictionList, correctLabels):
    assert len(predictionList) == len(correctLabels)
    cmList = np.zeros((4,4), dtype=np.int)
    N = len(predictionList)
    for i in range(0,N,1):
        rowIdx = int(correctLabels[i]/90)%4  # correctLabels
        colIdx = int(predictionList[i]/90)%4 # predictionList
        cmList[rowIdx][colIdx] += 1
        # print (str(correctLabels[i]/90) + ": " + str(correctLabels[i]) + "\t\t" +  str(predictionList[i]/90) + ": " +  str(predictionList[i]) + "\t\t" +  str(cmList[rowIdx][colIdx]))

    print("\n Confusion Matrix:")
    for rowIdx in range(0,4,1):
        line_string = " "
        for colIdx in range(0,4,1):
            line_string = line_string + str(cmList[rowIdx][colIdx]) + "\t"
        print(line_string)


def help():
    print("Example Commands:")
    print(" 1:   python  orient.py  train-data.txt  test-data.txt   best")
    print(" 2:   python  orient.py  train-data.txt  test-data.txt   best       model_file.mj")
    print(" 3:   python  orient.py  train-data.txt  test-data.txt   knn        5")
    print(" 4:   python  orient.py  train-data.txt  test-data.txt   nnet       30")
    print("Try again..\n\n")
    exit(-1)


def runKnn(algoName, k, train_data, test_data, v, fileName, test_names, test_labels):
    output = knn(k, train_data, test_data, v)
    writeOutput_PrintAnalysis(fileName, test_names, output, test_labels)


def runNnet(algoName, inputNodesCount, kHiddenCount, modelFile, train_data, test_data_features, outputFileName, test_names, test_labels):
    sizes = [inputNodesCount, kHiddenCount, 4]
    n2 = Network(sizes)

    if False and (algoName == "best"):
        n2.load_weights(modelFile)

    else: # (algoName == "nnet")
        epochs = 13 # number of iterations
        alpha = 0.1 # back-prop learning rate
        n2.train(train_data, alpha, epochs)
        if False and (modelFile is not None):
            n2.store_weights(modelFile)

    outputList = n2.test(test_data_features)
    output = []
    for op in outputList:
        if len(op) > 4:
            o = op[:4]
        else:
            o = op[:]
        # print(o)
        index, value = max(enumerate(o), key=operator.itemgetter(1))
        predicted = index*90
        output.append(predicted)
    writeOutput_PrintAnalysis(outputFileName, test_names, output, test_labels)


def main():
 #   startTime = time.time()*1000
    # Default Values...
    trainFileName = "train-data.txt"
    testFileName  = "test-data.txt"
    algoName = 'best'
    kHiddenCount = 9
    modelFile = None
    inputNodesCount = 192

    """
    COMMAND LINE ARGUMENTS...
    python  <arg[0]>   <arg[1]>        <arg[2]>        <arg[3]>   <arg[4]>

    python  <solFile>  <train>         <test>          <algo>     <count>
    python  orient.py  train-data.txt  test-data.txt   knn        k
    python  orient.py  train-data.txt  test-data.txt   nnet       hidden_count

    python  <solFile>  <train>         <test>          <algo>     <saved model>
    python  orient.py  train-data.txt  test-data.txt   best       model_file.mj
    """
    if len(sys.argv) >= 4:
        trainFileName = sys.argv[1]    # case is preserved
        testFileName  = sys.argv[2]    # case is preserved
        algoName = sys.argv[3].lower() # algo name in lower case

        if (algoName == 'best'):
            kHiddenCount = 9
            if (len(sys.argv) >= 5):
                modelFile = sys.argv[4]  # Optional parameter
            else:
                modelFile = None

        elif (algoName == 'knn') or (algoName == 'nnet'):
            kHiddenCount = int(sys.argv[4])

        else:
            print("Algo name is not correct! Try again.")
            help()

    else:  # len(sys.argv) < 4
        inVal = raw_input ("Too few arguments given!"
                           "\n"
                           "Should I demo with default values? (y/n): ").lower()[0]
        if (inVal == 'y'):
            print("Running the best method with default values! "
                  "It will take time to complete but it is guaranteed to give results.")
        else:
            help()

    ##### algorithms ###################################################################################################
    train_names, train_data = readDataFromFile(trainFileName)
    test_names, test_data = readDataFromFile(testFileName)

    test_data_features = [x[1] for x in test_data]
    test_labels = [x[0] for x in test_data] # true value


  #  endTime = time.time()*1000
  #  print ("Load Time:  "+str(endTime-startTime))
  #  startTime = time.time()*1000
    if (len(train_data[0][1])) != (len(test_data_features[0])):
        print("\n"
              "ERROR: Number of features in Train Set are Not same as Number of features in Test Set."
              "\n"
              "It has been assumed that both Test Set and Train Set will have same number of features."
              "\n"
              "Contact mjaglan and rakhasan in case of any confusion. Good day!"
              "\n")
        exit(-1)
    else:
        inputNodesCount = (len(train_data[0][1]))

    outputFileName = str(algoName)+"_output.txt"
    # print("Output File Name: "+outputFileName)
    if (algoName == 'knn'):
        ################# KNN ##########################################################################################
        # print("Executing KNN...")
        runKnn(algoName=algoName, k=kHiddenCount, train_data=train_data, test_data=test_data, v=1,
               fileName=outputFileName, test_names=test_names, test_labels=test_labels)

    elif (algoName == 'nnet'):
        ################# NNET #########################################################################################
        # print("Executing NNET...")
        runNnet(algoName=algoName, inputNodesCount=inputNodesCount, kHiddenCount=kHiddenCount, modelFile=modelFile,
                train_data=train_data, test_data_features=test_data_features, outputFileName=outputFileName,
                test_names=test_names, test_labels=test_labels)

    elif (algoName == 'best'):
        if True:
            ################# ALWAYS RUN THIS METHOD ###################################################################
            # print("Executing Best Method...")
            runKnn(algoName=algoName, k=kHiddenCount, train_data=train_data, test_data=test_data, v=1,
                   fileName=outputFileName, test_names=test_names, test_labels=test_labels)

        elif modelFile is None:
            ################# BEST KNN #################################################################################
            # print("Executing Best KNN...")
            runKnn(algoName=algoName, k=kHiddenCount, train_data=train_data, test_data=test_data, v=1,
                   fileName=outputFileName, test_names=test_names, test_labels=test_labels)

        else:
            ################# BEST NNET ################################################################################
            # print("Executing Best NNet...")
            runNnet(algoName=algoName, inputNodesCount=inputNodesCount, kHiddenCount=kHiddenCount, modelFile=modelFile,
                    train_data=train_data, test_data_features=test_data_features, outputFileName=outputFileName,
                    test_names=test_names, test_labels=test_labels)

    else:
        print("Algo name is not correct! Try again.")
        exit(-1)

#    endTime = time.time()*1000
#     print ("Test & Train Time:  "+str(endTime-startTime))

main()


