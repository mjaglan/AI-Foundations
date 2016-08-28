###################################
# CS B551 Fall 2015, Assignment #5
#
# Your names and user ids: Mayank Jaglan (mjaglan), Rakibul Hasan (rakhasan)
# SSH REPOSITORY: git@github.iu.edu:cs-b551/mjaglan-rakhasan-a5.git
#
# (Based on skeleton code by D. Crandall)
#
"""
***************** REPORT CONTENT *************************************************************************************** (1)
(1) a brief description of how your code works:

>> TRAINING:            def train(self, data)
We are keeping the trained meta-data in the following data structures:
    self.priors = dict()        # OBSERVED VARIABLES:       prior probabilities of words P(Wi); We tried using this information for best method implementation but shunned it later!
    self.pos_priors = dict()    # UNOBSERVED VARIABLES:     prior probabilities of POS-labels P(Si), or call it probabilities of POS-tags or states Si
    self.e_p = dict()           # EMISSION PROBABILITIES:   P(Wi | Si), i.e., Emission of observation Wi in state Si
    self.t_p = dict()           # TRANSITION PROBABILITIES: P(Sj | Si), i.e., Transition from past POS state Si to current POS state Sj

We calculate these probabilities as per the math formula present in the lecture slides.


>> NAIVE BAYES:         def naive(self, sentence)
The usage of notations Si and Wi are just explained above and are similar to the assignment question.

Here, first, we simply calculated the formula:
P(Si | Wi) = [ P(Wi | Si) * P(Si) / P(Wi) ]

Because the denominator will remain same for all, we ignore the P(Wi) for better computational performance. As we simply calculate following:
P(Si | Wi) = [ P(Wi | Si) * P(Si) ]

For each word Wi, the pos-tag Si with the highest probability P(Si | Wi) is considered for classifying the word Wi.



>> MCMC SAMPLER:        def mcmc(self, sentence, 5)
First we generated a random sample. From this one the subsequent samples were generated using generate_sample method. To
generate a sample, the probability of a pos tag for a word is calculated, given all the other words and their corresponding
tags observed, that is P(S_i | (S - {S_i}), W1,W2,...,Wn). If any emission or transition sequence is not learned from
training data, a small probability (.000000001) is assigned to them. The first 1000 samples were discarded to pass
the warming period and improve sampling accuracy.



>> MCMC MAX MARGINAL:   def max_marginal(self, sentence)
To calculate marginal distribution from samples, we first generated 3000 samples using mcmc (again after discarding the
first 1000 samples). From these samples we calculate max probability of each tag corresponding to each word in the test
sentence. We assign the pos tag which has the maximum probability for a word. And combining them we get the pos tags
for the whole sentence.



>> VITERBI MAP:         def viterbi(self, sentence)
Based on the training data set, we have learned following relevant parameters:
    INITIALS:    pos_priors = dict()    # UNOBSERVED VARIABLES:     prior probabilities of POS-labels P(Si), or call it probabilities of POS-tags or states Si
    EMISSIONS:   e_p = dict()           # EMISSION PROBABILITIES:   P(Wi | Si), i.e., Emission of observation Wi in state Si
    TRANSITIONS: t_p = dict()           # TRANSITION PROBABILITIES: P(Sj | Si), i.e., Transition from past POS state Si to current POS state Sj


Key Idea: We need to calculate the posterior probability of a state sequence, P(S1,...,Sn | W1,...,Wn);
it's an efficient algorithm based on dynamic programming paradigm!

    Computation Assumption For Algorithm Implementation: Below text is taken from slides!
    -- denominator of  bayes expansion of P(S1,...,Sn | W1,...,Wn), depends only on only on Wi (**observed variable**)
    -- Wi depends only on Si, as per the bayes net from assignment question
    -- Markov property "Si+1" depends only on "Si"
    -- So we ignore the denominator during computations of intermediate viterbi values. This is a harmless assumption and simplifies the calculations!


Handling New Words: For new words a small emission probability is assumed which is equal to 0.00000000000000000000000000000001
and such words are labelled with "noun" pos tag

Handling New Transition Edges: Just in case, there happens to be a new transition edge between two existing POS-states, a small transition
probability is assumed which is equal to 0.00000000000000000000000000000001 and such words are labelled with "noun" pos tag;

Final Implementation of Viterbi Sequence Decoding: With above setting at hand we realized that assuming small probabilities for viterbi decoding is not so much effective. So we created
a wrapper over this core algorithm implementation. Instead of sending a complete sentence to viterbi sequence calculator, we send only
those sub-sequence of words, for which there already exits at least one emission probability in our training data structure. The words that
are completely new, for them we simply tag them as "noun" POS-label. This made the viterbi decoding work in a predictable fashion and the
accuracy of this decoder is around 95% in almost all of the cases.



>> BEST CLASSIFIER:     def best(self, sentence)
Considering the restrictions imposed on the scope of programming and resources we can use,
plus, based on the multiple tweaks and test-trials of above 4 algorithms, we realized that viterbi decoding works best for us.
Here we are making a call to viterbi algorithm. Believe that a dynamic programming implementation will result in the most
number of correct sentence tags, i.e., the most likely POS-tag sequence for the given sequence of words.

NOTE: We have attempted multiple experiments in following sub-directories:
>> mjaglan-rakhasan-a5/rough_work_mjaglan/...
>> mjaglan-rakhasan-a5/rough_work_rakibul/...



>> POSTERIOR:           def posterior(self, sentence, label)
This method required us to calculate  probability P(S1,S2,...,Sn | W1,W2,...,Wn) for which we referred emission
probability table and assumed that current observed variable (**word**) is dependent only on current unobserved variable (**POS-state**).

Going by this, we just need to multiply individual elements. Instead of multiplying all entities first and then taking logarithm,
we preferred to take the logarithm of individual entities first and then sum them all in sequence. This way we did not lose
floating point precision for very low exponents.



************************************************************************************************************************ (2)
(2) the results of the evaluation on the bc.test file (i.e. the percentage of
    correct words and sentences for each algorithm, as displayed by our scoring code):

NAIVE BAYES:         def naive(self, sentence)
93.92% of the test words     are correctly classified with POS labels;
47.45% of the test sentences are correctly classified with POS labels sequence;


MCMC SAMPLER:        def mcmc(self, sentence, 5)
93.83% of the test words     are correctly classified with POS labels;
47.80% of the test sentences are correctly classified with POS labels sequence;


MCMC MAX MARGINAL:   def max_marginal(self, sentence)
93.69% of the test words     are correctly classified with POS labels;
47.40% of the test sentences are correctly classified with POS labels sequence;


VITERBI MAP:         def viterbi(self, sentence)
95.26% of the test words     are correctly classified with POS labels;
53.90% of the test sentences are correctly classified with POS labels sequence;


BEST CLASSIFIER:     def best(self, sentence)
95.26% of the test words     are correctly classified with POS labels;
53.90% of the test sentences are correctly classified with POS labels sequence;


COMPUTATIONAL PERFORMANCE: On average, for a test sentence with 15 words, it takes approximately 2.4 seconds for following:
        all the 5 classifiers to predict the POS-labels
        + logarithmic posterior probability to be computed
        + Score class to compute and print the scores

Code for above performance metrics is present in two files:
    mjaglan-rakhasan-a5/rough_work_mjaglan/label.py
    mjaglan-rakhasan-a5/rough_work_mjaglan/pos_scorer.py


COMMENTS:
>> Viterbi decoding is awesome: better sequence prediction + fast computation
>> Naive Bayes is cranky!
>> Gibbs Sampling takes more time as the sample size limit is increased!
>> Max Marginal suffers in performance and accuracy because of above sampler()



************************************************************************************************************************ (3)
(3) a discussion of any problems you faced, any assumptions, simplifications, and/or design decisions you made.

>> ASSUMPTIONS TAKEN: (3 things)
If we encounter a new word in test set, then we take priors and emissions as very small probability number: 0.00000000000000000000000000000001;
If we encounter a new word in test set, then we take associated POS tag as "noun"; based on multiple test runs, it comes out to be the most correct guess;
If we encounter a new POS tag in test set, then we take priors, transitions and emissions as very small probability number: 0.00000000000000000000000000000001;


>> SIMPLIFICATIONS & SOLVER SYSTEM DESIGN:
To keep things simple we have used some hard codings like:
>> Add small probability number for missing or new data.
>> Assume new word as "noun" POS label. This works better than other models implemented in /rough_work_mjaglan/...
>> Add small probability number for new state transitions.


>> PROBLEMS FACED:
It is very difficult to correctly classify a word that has never been in our training set. Further, since we are implementing the
statistical classification techniques, we need more and more and more training data for better results. "bc.train" file contains only
44204 sentences. Resulting accuracies of classifiers are not practical figures. Limited training data resulted in less mature POS tagging
system; And limited scope of experiments allowed around the traning data set constrained us w.r.t. to gradable algorithms to implement :-)


>> FUTURE IMPROVEMENTS:
As part of graded work, we really wished we could implement feedback model to improve training whenever there is a wrong classification. Such a system is more closely
related to learning from mistakes philosophy. This type of mechanism may slow down the computational performance (**call it recovery time**) but it is a scalable solution. 
In future we could implement such mechanism, if allowed.

Another approach we think is to have a local dictionary where it could break down the words (Wi) based on certain pattern of suffices (Ex: -ing, -ed, es, etc); such suffixed 
may help correctly classify new words which are not nouns.


Cheers!!
TEAM: mjaglan-rakhasan

"""

import random
import math


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#

class Viterbi:
    def __init__(self, ps, cs, val):
        self.prev_state = ps
        self.curr_state = cs
        self.viterbiValue = val


class Solver:
    def __init__(self):
        self.priors = dict()  # prior probabilities of words P(Wi)
        self.pos_priors = dict()  # prior probabilities of tags P(Si)
        self.e_p = dict()  # emission probabilities P(Wi|Si)
        self.t_p = dict()  # transition probabilities P(Si+1 | Si)

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        # Calculate P(S1,S2,...,Sn | W1,W2,...,Wn) for this one instance

        logBase = 10  # another assumption
        wordList = list(sentence)
        # assert len(wordList) == len(label)
        N = len(wordList)

        result = 0
        for idx in range(0, N, 1):
            if wordList[idx] in self.e_p[label[idx]].keys():
                x = self.e_p[label[idx]][wordList[idx]]
            else:
                x = 0.00000000000000000000000000000001  # assumption :P
            result += math.log(x, logBase)

        result += math.log(self.pos_priors[label[0]], logBase)
        prev_label = label[0]
        for idx in range(1, N, 1):
            if label[idx] in self.e_p[prev_label].keys():
                x = self.t_p[prev_label][label[idx]]
            else:
                x = 0.00000000000000000000000000000001  # assumption :P
            result += math.log(x, logBase)
            prev_label = label[idx]

        return result

    # Do the training!
    #
    def train(self, data):
        # P(O): prior words O=Wi
        # P(S): prior lablels S=Si
        # E-S (O): emission of O=Wi in state S=Si
        # T (S1, S2): Transition from past S1=Si to current S2=Sj

        total_word_count = 0
        for d in data:
            total_word_count += len(d[0])
            for i in xrange(len(d[0])):
                word = d[0][i]
                pos = d[1][i]
                # calculate prior for word
                if word in self.priors:
                    self.priors[word] += 1
                else:
                    self.priors[word] = 1
                # calculate prior for pos
                if pos in self.pos_priors:
                    self.pos_priors[pos] += 1
                else:
                    self.pos_priors[pos] = 1
                if pos not in self.e_p: # get actual probability P(Wi | Si)
                    self.e_p[pos] = {word: 1}
                elif word not in self.e_p[pos]:
                    self.e_p[pos][word] = 1
                else:
                    self.e_p[pos][word] += 1
                # calculate first order transition probabilities
                if i > 0:
                    prev_pos = d[1][i - 1]
                    if prev_pos not in self.t_p:
                        self.t_p[prev_pos] = {pos: 1}
                    elif pos not in self.t_p[prev_pos]:
                        self.t_p[prev_pos][pos] = 1
                    else:
                        self.t_p[prev_pos][pos] += 1

        for word in self.priors:
            self.priors[word] = float(self.priors[word]) / float(total_word_count)
        for pos in self.pos_priors:
            self.pos_priors[pos] = float(self.pos_priors[pos]) / float(total_word_count)

        for pos in self.e_p:
            total = 0
            for word in self.e_p[pos]:
                total += self.e_p[pos][word]
            for word in self.e_p[pos]:
                self.e_p[pos][word] = \
                    float(self.e_p[pos][word]) / float(total)
        for prev_pos in self.t_p:
            total = 0
            for pos in self.t_p[prev_pos]:
                total += self.t_p[prev_pos][pos]
            for pos in self.t_p[prev_pos]:
                self.t_p[prev_pos][pos] = \
                    float(self.t_p[prev_pos][pos]) / float(total)

    # Functions for each algorithm.
    #
    def naive(self, sentence):
        # P(Sj | Wj) = [ P(Wj | Sj) * P(Sj) / P(Wj) ]
        # P(Sj | Wj) = [ P(Wj | Sj) * P(Sj) ]

        globalResultList = []
        for aWord in list(sentence):
            globalMaxProb = 0
            globalPOS = 'noun'

            for pos in self.e_p.keys():

                if aWord in self.e_p[pos].keys():
                    localPW = self.e_p[pos][aWord]  # get actual probability P(Wj | Sj)
                    localProb = localPW * self.pos_priors[pos]

                    if (globalMaxProb < localProb):
                        globalMaxProb = localProb
                        globalPOS = pos

            globalResultList.extend([globalPOS])

        # assert len(list(sentence)) == len(globalResultList)
        return [[globalResultList], []]

    def generate_sample(self, sentence, sample):
        sentence_len = len(sentence)
        tags = self.t_p.keys()
        for index in xrange(sentence_len):
            word = sentence[index]
            probabilities = [0] * len(self.t_p)

            s_1 = sample[index - 1] if index > 0 else " "
            s_3 = sample[index + 1] if index < sentence_len - 1 else " "

            for j in xrange(len(self.t_p)):  # try by assigning every tag
                s_2 = tags[j]

                ep = self.e_p[s_2][word] if s_2 in self.e_p and word in self.e_p[s_2] else .000000001
                j_k = self.t_p[s_2][s_3] if s_2 in self.t_p and s_3 in self.t_p[s_2] else .000000001
                i_j = self.t_p[s_1][s_2] if s_1 in self.t_p and s_2 in self.t_p[s_1] else .000000001

                if index == 0:
                    probabilities[j] = j_k * ep * self.pos_priors[s_2]
                elif index == sentence_len - 1:
                    probabilities[j] = i_j * ep * self.pos_priors[s_1]
                else:
                    probabilities[j] = i_j * j_k * ep * self.pos_priors[s_1]

            s = sum(probabilities)
            probabilities = [x / s for x in probabilities]
            rand = random.random()
            p_sum = 0
            for i in xrange(len(probabilities)):
                p = probabilities[i]
                p_sum += p
                if rand < p_sum:
                    sample[index] = tags[i]
                    break

        return sample

    def mcmc(self, sentence, sample_count):
        sample = ["noun"] * len(sentence)  # initial sample, all noun
        for i in xrange(1000):  # ignore first 1000 samples
            sample = self.generate_sample(sentence, sample)
        samples = []
        for p in xrange(sample_count):
            sample = self.generate_sample(sentence, sample)
            samples.append(sample)
        return [samples, []]

    def best(self, sentence):
        return self.viterbi(sentence)
    
    def max_marginal(self, sentence):
        sample_count = 3000
        samples = self.mcmc(sentence, sample_count)[0]
        probabilities = []
        final_sample = []

        for i in xrange(len(sentence)):
            tag_count = dict.fromkeys(self.e_p.keys(), 0)
            for sample in samples:
                tag_count[sample[i]] += 1
            final_sample.append(max(tag_count, key=tag_count.get))
            probabilities.append(tag_count[final_sample[i]] / sample_count)

        return [[final_sample], [probabilities]]

    def viterbi(self, sentence):
        wordList = list(sentence)
        N = len(wordList)
        checkList = [0] * N
        resultPOS = ['noun'] * N

        idx = -1
        for eachWord in wordList:
            idx += 1
            for eachPOS in self.e_p.keys():
                if eachWord in self.e_p[eachPOS].keys():
                    checkList[idx] = 1
                    break

        startIdx = 0
        while (startIdx < N):
            if (checkList[startIdx] == 1):
                endIdx = startIdx
                while ((endIdx < N) and (checkList[endIdx] == 1)):
                    endIdx += 1
                subResult = self.viterbiCalculator(wordList[startIdx:endIdx:1])
                i = startIdx
                j = 0
                # assert len(subResult) == (endIdx-startIdx)
                while (i < endIdx):
                    resultPOS[i] = subResult[j]
                    i += 1
                    j += 1
            startIdx += 1

        # assert len(resultPOS) == len(wordList)
        return [[resultPOS[::1]], []]

    def viterbiCalculator(self, sentence):
        """
            Next, implement the Viterbi Algorithm to find the most likely sequence of state variables,
            (s*1,...,s*N)   =   arg max (s1,...,sN) P(Si = si | W):
        """

        wordList = list(sentence)
        globalResultList = []  # viterbi sequence (reverse list)
        posStates = self.e_p.keys()

        tData = {}
        localMaxPrevState = ''
        for eachStateCurr in posStates:
            if wordList[0] in self.e_p[eachStateCurr].keys():
                tData[eachStateCurr] = Viterbi(localMaxPrevState, eachStateCurr,
                                               self.pos_priors[eachStateCurr] * self.e_p[eachStateCurr][wordList[0]])
            else:
                tData[eachStateCurr] = Viterbi(localMaxPrevState, eachStateCurr, self.pos_priors[
                    eachStateCurr] * 0.00000000000000000000000000000001)  # assumption :P
        firstHalf = [tData]
        secondHalf = list(globalResultList)
        firstHalf.extend(secondHalf)
        globalResultList = firstHalf

        for eachWord in wordList[1:]:
            tData = {}
            for eachStateCurr in posStates:
                localMax = 0.00000000000000000000000000000001  # assumption :P
                localMaxPrevState = '.'  # assumption :P

                for eachStatePrev in posStates:
                    vi = globalResultList[0][eachStatePrev].viterbiValue
                    # print(vi)
                    pij1 = self.t_p[eachStatePrev]
                    if (eachStateCurr in pij1.keys()):
                        pij2 = pij1[eachStateCurr]
                    else:
                        pij2 = 0.00000000000000000000000000000001  # assumption :P
                    # print(pij2)
                    temp = vi * pij2
                    if localMax < temp:
                        localMax = temp
                        localMaxPrevState = eachStatePrev

                if eachWord in self.e_p[eachStateCurr].keys():
                    tData[eachStateCurr] = Viterbi(localMaxPrevState, eachStateCurr,
                                                   localMax * self.e_p[eachStateCurr][eachWord])
                else:
                    tData[eachStateCurr] = Viterbi(localMaxPrevState, eachStateCurr,
                                                   localMax * 0.00000000000000000000000000000001)  # assumption :P

            firstHalf = [tData]
            secondHalf = list(globalResultList)
            firstHalf.extend(secondHalf)
            globalResultList = firstHalf

        # assert len(globalResultList) == len(wordList)

        # return most likely sequence
        resultStates = []
        globalMax = 0
        globalMaxState = ''
        globalMaxPrevState = ''
        for viterbiState in globalResultList[0].keys():
            if globalMax < globalResultList[0][viterbiState].viterbiValue:
                globalMax = globalResultList[0][viterbiState].viterbiValue
                globalMaxState = globalResultList[0][viterbiState].curr_state
                globalMaxPrevState = globalResultList[0][viterbiState].prev_state
        resultStates.extend([globalMaxState])

        if globalMaxPrevState != '':
            resultStates.extend([globalMaxPrevState])
            for eachLevel in globalResultList[1:]:
                temp = eachLevel[resultStates[-1]].prev_state
                if temp != '':
                    resultStates.extend([temp])

        # assert len(resultStates) == len(wordList)
        return resultStates[::-1]

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself.
    # It's supposed to return a list with two elements:
    #
    #  - The first element is a list of part-of-speech labelings of the sentence.
    #    Each of these is a list, one part of speech per word of the sentence.
    #    Most algorithms only return a single labeling per sentence, except for the
    #    mcmc sampler which is supposed to return 5.
    #
    #  - The second element is a list of probabilities, one per word. This is
    #    only needed for max_marginal() and is the marginal probabilities for each word.
    #
    def solve(self, algo, sentence):
        if algo == "Naive":
            return self.naive(sentence)
        elif algo == "Sampler":
            return self.mcmc(sentence, 5)
        elif algo == "Max marginal":
            return self.max_marginal(sentence)
        elif algo == "MAP":
            return self.viterbi(sentence)
        elif algo == "Best":
            return self.best(sentence)
        else:
            print "Unknown algo!"
