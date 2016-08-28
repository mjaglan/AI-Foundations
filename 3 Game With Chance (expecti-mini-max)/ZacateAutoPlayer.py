"""
Automatic Zacate game player
B551 Fall 2015
----------------------------------------------------------------------------

PUT YOUR NAME AND USER ID HERE!
Modified by Mayank Jaglan, mjaglan@umail.iu.edu
Based on skeleton code by D. Crandall

PUT YOUR REPORT HERE!
Question 1> What is the average score your program is able to obtain?
[mjaglan]: Ran the code multiple times, as of 11:49 PM, 18 Oct., the average score recorded is 200.27;
Refer "ROUGH NOTES FOR THIS SOLUTION" section mentioned below, for knowing more about how I tested my smart Zacate player .


How the new smart Zacate player works?
[mjaglan]: We have had two dimensions in the initial code which are creating randomness (i.e., game with chance):
 >>> (a) Score category selection
 >>> (b) dice selections for re-roll

Objective has been to reduce the randomness for both (a) and (b). In the new smart Zacate player, both (a) and (b) have
 been handled separately. Let's discuss one by one -
 (a) Score category selection: Because it is a game with chance, let's go greedy! For the current dice pattern select
 a valid category which offers the high possible score; If 2 or more categories offers the same score,
 select the one with least probability of occurence.

 (b) dice selections for re-roll: Idea is to minimize the randomness by picking the maximum matching dice pattern;
 Re-roll the minimun number of dice which are not part of that matching dice pattern;

For all the homework done for desining this solution, please feel free to refer the README.MD file or below metioned
"ROUGH NOTES FOR THIS SOLUTION" section.



ROUGH NOTES FOR THIS SOLUTION:
#1 FILE: github.iu.edu/cs-b551/mjaglan-a3/rough_work/part2/probability calculations/*
>>>> contains raw images of probability calculations done for the code on the note book

#2 FILE: github.iu.edu/cs-b551/mjaglan-a3/rough_work/part2/question.txt
>>>> contains max score, and final probability calculations for max possible score;
>>>> contains algorithm design strategy for selecting dice re-roll, and score-category selection;

#3 FILE: github.iu.edu/cs-b551/mjaglan-a3/rough_work/part2/zacate.py
>>>> modified the existing algorithm for the purpose of generating performace metrics for my smart Zacate player;



REFERECES FOR CODING THIS SOLUTION:
#1 BOOK: Artificial Intelligence: A Modern Approach (3rd Edition) by Stuart Russell and Peter Norvig
      TOPIC: Game with chance

#2 PROGRAMMING: http://stackoverflow.com
#3 PROGRAMMING: http://www.tutorialspoint.com
#4 PROGRAMMING: https://docs.python.org/2/reference/




----------------------------------------------------------------------------


This is the file you should modify to create your new smart Zacate player.
The main program calls this program three times for each turn.
  1. First it calls first_roll, passing in a Dice object which records the
     result of the first roll (state of 5 dice) and current Scorecard.
     You should implement this method so that it returns a (0-based) list
     of dice indices that should be re-rolled.

  2. It then re-rolls the specified dice, and calls second_roll, with
     the new state of the dice and scorecard. This method should also return
     a list of dice indices that should be re-rolled.

  3. Finally it calls third_roll, with the final state of the dice.
     This function should return the name of a scorecard category that
     this roll should be recorded under. The names of the scorecard entries
     are given in Scorecard.Categories.
"""

from ZacateState import Dice
from ZacateState import Scorecard
import random

class ZacateAutoPlayer:

      # Check which dice should we consider for re-roll
      def selectDice2(self, rollObj, availableCatList):
            #availableCatList = list(set(Scorecard.Categories) - set(scorecard.scorecard.keys()))
            dice = rollObj.dice # actual dice-value pattern
            constDiceIndices = [0,1,2,3,4]

            maxDiceMatchCount = 0   # global max value
            unMatchDiceIndicesSet = [0,1,2,3,4] # the smallest size dice-positions that we should re-roll

            maxDiceIdealMatchScore = 0 # not using as yet
            maxDiceCurrentMatchScore = 0 # not using as yet

            # category matches with any of the available category
            # conditions: lowest score category on top, highest score category at the bottom
            # terminate: if myLocalDiceMatchCount is a 100% match, then return empty unmatchDiceIndexSet from there only
            # if myLocalDiceMatchCount is NOT 100% match, then keep searching for smallest size unmatch-Dice-Index-Set and return it


            # 5a
            if "quintupulo" in availableCatList:
                  dValFreq = [dice.count(i) for i in range(1,7)] # here we know frequency of occurence of each number

                  maxFreq = 0
                  maxFreqDiceVal = 0
                  diceVal = 1
                  while(diceVal <= 6):
                        if(maxFreq < dValFreq[diceVal-1]):
                              maxFreq = dValFreq[diceVal-1]
                              maxFreqDiceVal = diceVal
                        diceVal = diceVal + 1
                  localMatchDiceIdx = [i for i, x in enumerate(dice) if x == maxFreqDiceVal]
                  myLocalDiceMatchCount = maxFreq

                  # find local-UnMatch-Dice-Idx
                  if(myLocalDiceMatchCount >= 5):
                        return [] # perfectMatch
                  else: # calculate this?
                        localUnMatchDiceIdx = list(set(constDiceIndices) - set(localMatchDiceIdx))

                  # check if localMatch is largest match
                  if (maxDiceMatchCount <= myLocalDiceMatchCount):
                        maxDiceMatchCount = myLocalDiceMatchCount
                        unMatchDiceIndicesSet = []
                        for idx in localUnMatchDiceIdx:
                              unMatchDiceIndicesSet = unMatchDiceIndicesSet + [idx]


            # all 5 of 5 specific combination: [a, a+1, a+2, a+3, a+4]
            if "pupusa de queso" in availableCatList:
                  type1 = [1,2,3,4,5]
                  type2 = [2,3,4,5,6]

                  # type1: calculate max-dice-match-count, local-UnMatch-Dice-Idx
                  i = 0
                  n = len(dice)
                  localUnMatchDiceIdx = []
                  myLocalDiceMatchCount = 0
                  while(i<n):
                        if (dice[i] in type1):
                              myLocalDiceMatchCount = (myLocalDiceMatchCount + 1)
                              type1.remove(dice[i])
                        else:
                              localUnMatchDiceIdx = localUnMatchDiceIdx + [i]
                        i = (i + 1)
                  # check if localMatch is largest match
                  if (maxDiceMatchCount <= myLocalDiceMatchCount):
                        maxDiceMatchCount = myLocalDiceMatchCount
                        unMatchDiceIndicesSet = []
                        for idx in localUnMatchDiceIdx:
                              unMatchDiceIndicesSet = unMatchDiceIndicesSet + [idx]


                  # type2: calculate max-dice-match-count, local-UnMatch-Dice-Idx
                  i = 0
                  n = len(dice)
                  localUnMatchDiceIdx = []
                  myLocalDiceMatchCount = 0
                  while(i<n):
                        if (dice[i] in type2):
                              myLocalDiceMatchCount = (myLocalDiceMatchCount + 1)
                              type2.remove(dice[i])
                        else:
                              localUnMatchDiceIdx = localUnMatchDiceIdx + [i]
                        i = (i + 1)
                  # check if localMatch is largest match
                  if (maxDiceMatchCount <= myLocalDiceMatchCount):
                        maxDiceMatchCount = myLocalDiceMatchCount
                        unMatchDiceIndicesSet = []
                        for idx in localUnMatchDiceIdx:
                              unMatchDiceIndicesSet = unMatchDiceIndicesSet + [idx]


            # 4a
            if "cuadruple" in availableCatList:
                  dValFreq = [dice.count(i) for i in range(1,7)] # here we know frequency of occurence of each number

                  maxFreq = 0
                  maxFreqDiceVal = 0
                  diceVal = 1
                  while(diceVal <= 6):
                        if(maxFreq < dValFreq[diceVal-1]):
                              maxFreq = dValFreq[diceVal-1]
                              maxFreqDiceVal = diceVal
                        diceVal = diceVal + 1
                  localMatchDiceIdx = [i for i, x in enumerate(dice) if x == maxFreqDiceVal]
                  myLocalDiceMatchCount = maxFreq

                  # find local-UnMatch-Dice-Idx
                  if(myLocalDiceMatchCount >= 4):
                        return [] # perfectMatch
                  else: # calculate this?
                        localUnMatchDiceIdx = list(set(constDiceIndices) - set(localMatchDiceIdx))

                  # check if localMatch is largest match
                  if (maxDiceMatchCount <= myLocalDiceMatchCount):
                        maxDiceMatchCount = myLocalDiceMatchCount
                        unMatchDiceIndicesSet = []
                        for idx in localUnMatchDiceIdx:
                              unMatchDiceIndicesSet = unMatchDiceIndicesSet + [idx]


            # 1 or more '6' in 5-dice-set
            if "seises" in availableCatList:
                  myLocalDiceMatchCount = dice.count(Scorecard.Numbers["seises"])
                  if (maxDiceMatchCount <= myLocalDiceMatchCount):
                        maxDiceMatchCount = myLocalDiceMatchCount
                        unMatchDiceIndicesSet = []
                        for idx in constDiceIndices:
                              if (Scorecard.Numbers["seises"] != dice[idx]):
                                    unMatchDiceIndicesSet = unMatchDiceIndicesSet + [idx]


            # (3a)
            if "triple" in availableCatList:
                  dValFreq = [dice.count(i) for i in range(1,7)] # here we know frequency of occurence of each number

                  maxFreq = 0
                  maxFreqDiceVal = 0
                  diceVal = 1
                  while(diceVal <= 6):
                        if(maxFreq < dValFreq[diceVal-1]):
                              maxFreq = dValFreq[diceVal-1]
                              maxFreqDiceVal = diceVal
                        diceVal = diceVal + 1
                  localMatchDiceIdx = [i for i, x in enumerate(dice) if x == maxFreqDiceVal]
                  myLocalDiceMatchCount = maxFreq

                  # find local-UnMatch-Dice-Idx
                  if(myLocalDiceMatchCount >= 3):
                        return [] # perfectMatch
                  else: # calculate this?
                        localUnMatchDiceIdx = list(set(constDiceIndices) - set(localMatchDiceIdx))

                  # check if localMatch is largest match
                  if (maxDiceMatchCount <= myLocalDiceMatchCount):
                        maxDiceMatchCount = myLocalDiceMatchCount
                        unMatchDiceIndicesSet = []
                        for idx in localUnMatchDiceIdx:
                              unMatchDiceIndicesSet = unMatchDiceIndicesSet + [idx]


            # all 4 of 5 specific combination: [a, a+1, a+2, a+3]
            if "pupusa de frijol" in availableCatList:
                  type1 = [1,2,3,4]
                  type2 = [2,3,4,5]
                  type3 = [3,4,5,6]

                  # type1: calculate max-dice-match-count, local-UnMatch-Dice-Idx
                  i = 0
                  n = len(dice)
                  localUnMatchDiceIdx = []
                  myLocalDiceMatchCount = 0
                  while(i<n):
                        if (dice[i] in type1):
                              myLocalDiceMatchCount = (myLocalDiceMatchCount + 1)
                              type1.remove(dice[i])
                        else:
                              localUnMatchDiceIdx = localUnMatchDiceIdx + [i]
                        i = (i + 1)
                  # check if localMatch is largest match
                  if (maxDiceMatchCount <= myLocalDiceMatchCount):
                        maxDiceMatchCount = myLocalDiceMatchCount
                        unMatchDiceIndicesSet = []
                        for idx in localUnMatchDiceIdx:
                              unMatchDiceIndicesSet = unMatchDiceIndicesSet + [idx]


                  # type2: calculate max-dice-match-count, local-UnMatch-Dice-Idx
                  i = 0
                  n = len(dice)
                  localUnMatchDiceIdx = []
                  myLocalDiceMatchCount = 0
                  while(i<n):
                        if (dice[i] in type2):
                              myLocalDiceMatchCount = (myLocalDiceMatchCount + 1)
                              type2.remove(dice[i])
                        else:
                              localUnMatchDiceIdx = localUnMatchDiceIdx + [i]
                        i = (i + 1)
                  # check if localMatch is largest match
                  if (maxDiceMatchCount <= myLocalDiceMatchCount):
                        maxDiceMatchCount = myLocalDiceMatchCount
                        unMatchDiceIndicesSet = []
                        for idx in localUnMatchDiceIdx:
                              unMatchDiceIndicesSet = unMatchDiceIndicesSet + [idx]


                  # type3: calculate max-dice-match-count, local-UnMatch-Dice-Idx
                  i = 0
                  n = len(dice)
                  localUnMatchDiceIdx = []
                  myLocalDiceMatchCount = 0
                  while(i<n):
                        if (dice[i] in type3):
                              myLocalDiceMatchCount = (myLocalDiceMatchCount + 1)
                              type3.remove(dice[i])
                        else:
                              localUnMatchDiceIdx = localUnMatchDiceIdx + [i]
                        i = (i + 1)
                  # check if localMatch is largest match
                  if (maxDiceMatchCount <= myLocalDiceMatchCount):
                        maxDiceMatchCount = myLocalDiceMatchCount
                        unMatchDiceIndicesSet = []
                        for idx in localUnMatchDiceIdx:
                              unMatchDiceIndicesSet = unMatchDiceIndicesSet + [idx]


            # 1 or more '5' in 5-dice-set
            if "cincos" in availableCatList:
                  myLocalDiceMatchCount = dice.count(Scorecard.Numbers["cincos"])
                  if (maxDiceMatchCount <= myLocalDiceMatchCount):
                        maxDiceMatchCount = myLocalDiceMatchCount
                        unMatchDiceIndicesSet = []
                        for idx in constDiceIndices:
                              if (Scorecard.Numbers["cincos"] != dice[idx]):
                                    unMatchDiceIndicesSet = unMatchDiceIndicesSet + [idx]


            # (3a, 2b): How to reach to this combo? How to find partial patterns for this combo?
            if "elote" in availableCatList:
                  dcounts = [dice.count(i) for i in range(1,7)] # here we know frequency of occurence of each number
                  i = len(dcounts)
                  localMaximaIdx = -1
                  localMaxima = 0
                  globalMaximaIdx1 = -1
                  globalMaxima1 = 0
                  aDice = -1
                  aDiceCount = -1 # 1st most frequent
                  bDice = -1
                  bDiceCount = -1 # 2nd most frequent
                  while(i>0):
                        localMaximaIdx = i-1
                        localMaxima = dcounts[localMaximaIdx]
                        if (globalMaxima1 <= localMaxima):
                              bDice = aDice
                              bDiceCount = aDiceCount

                              globalMaxima1 = localMaxima
                              aDiceCount = globalMaxima1
                              globalMaximaIdx1 = localMaximaIdx
                              aDice = globalMaximaIdx1+1
                        i = i-1

                  # find local-UnMatch-Dice-Idx
                  aDicePositions = [i for i, x in enumerate(dice) if x == aDice]
                  bDicePositions = [i for i, x in enumerate(dice) if x == bDice]

                  myLocalDiceMatchCount = 0
                  if (aDiceCount==5):
                        myLocalDiceMatchCount = myLocalDiceMatchCount + aDiceCount
                        localMatchDiceIdx = aDicePositions
                        return [] # perfectMatch
                  elif (aDiceCount==4):
                        myLocalDiceMatchCount = myLocalDiceMatchCount + 3 + bDiceCount
                        localMatchDiceIdx = aDicePositions[0:3] + bDicePositions
                        # create localMatchDiceIdx
                  elif (aDiceCount<=3):
                        myLocalDiceMatchCount = myLocalDiceMatchCount + aDiceCount + bDiceCount
                        localMatchDiceIdx = aDicePositions + bDicePositions
                        # create localMatchDiceIdx


                  if( (aDiceCount==3 and bDiceCount==2) ):
                        return [] # perfectMatch
                  else:
                        localUnMatchDiceIdx = list(set(constDiceIndices) - set(localMatchDiceIdx))

                  # check if localMatch is largest match
                  if (maxDiceMatchCount <= myLocalDiceMatchCount):
                        maxDiceMatchCount = myLocalDiceMatchCount
                        unMatchDiceIndicesSet = []
                        for idx in localUnMatchDiceIdx:
                              unMatchDiceIndicesSet = unMatchDiceIndicesSet + [idx]


            # 1 or more '4' in 5-dice-set
            if "cuatros" in availableCatList:
                  myLocalDiceMatchCount = dice.count(Scorecard.Numbers["cuatros"])
                  if (maxDiceMatchCount <= myLocalDiceMatchCount):
                        maxDiceMatchCount = myLocalDiceMatchCount
                        unMatchDiceIndicesSet = []
                        for idx in constDiceIndices:
                              if (Scorecard.Numbers["cuatros"] != dice[idx]):
                                    unMatchDiceIndicesSet = unMatchDiceIndicesSet + [idx]

            # 1 or more '3' in 5-dice-set
            if "treses" in availableCatList:
                  myLocalDiceMatchCount = dice.count(Scorecard.Numbers["treses"])
                  if (maxDiceMatchCount <= myLocalDiceMatchCount):
                        maxDiceMatchCount = myLocalDiceMatchCount
                        unMatchDiceIndicesSet = []
                        for idx in constDiceIndices:
                              if (Scorecard.Numbers["treses"] != dice[idx]):
                                    unMatchDiceIndicesSet = unMatchDiceIndicesSet + [idx]


            # 1 or more '2' in 5-dice-set
            if "doses" in availableCatList:
                  myLocalDiceMatchCount = dice.count(Scorecard.Numbers["doses"])
                  if (maxDiceMatchCount <= myLocalDiceMatchCount):
                        maxDiceMatchCount = myLocalDiceMatchCount
                        unMatchDiceIndicesSet = []
                        for idx in constDiceIndices:
                              if (Scorecard.Numbers["doses"] != dice[idx]):
                                    unMatchDiceIndicesSet = unMatchDiceIndicesSet + [idx]


            # 1 or more '1' in 5-dice-set
            if "unos" in availableCatList:
                  myLocalDiceMatchCount = dice.count(Scorecard.Numbers["unos"])
                  if (maxDiceMatchCount <= myLocalDiceMatchCount):
                        maxDiceMatchCount = myLocalDiceMatchCount
                        unMatchDiceIndicesSet = []
                        for idx in constDiceIndices:
                              if (Scorecard.Numbers["unos"] != dice[idx]):
                                    unMatchDiceIndicesSet = unMatchDiceIndicesSet + [idx]


            return unMatchDiceIndicesSet


      # choose next best matching high score category
      def firstMatchHighScoreCategory(self, rollObj, availableCatList):
            #availableCatList = list(set(Scorecard.Categories) - set(scorecard.scorecard.keys()))
            dice = rollObj.dice
            counts = [dice.count(i) for i in range(1,7)]
            gCategory = random.choice( availableCatList ) # avoid disQualification
            gScore = -1

            # category matches with any of the available category
            if "quintupulo" in availableCatList:
                  score = 50 if max(counts) == 5 else 0
                  if (  ((score != 0) and (score > gScore))  or  ((score == 0) and (score >= gScore))  ):
                        gScore = score
                        gCategory = "quintupulo"

            if "pupusa de queso" in availableCatList:
                  score = 40 if sorted(dice) == [1,2,3,4,5] or sorted(dice) == [2,3,4,5,6] else 0
                  if (  ((score != 0) and (score > gScore))  or  ((score == 0) and (score >= gScore))  ):
                        gScore = score
                        gCategory = "pupusa de queso"

            if "cuadruple" in availableCatList:
                  score = sum(dice) if max(counts) >= 4 else 0
                  if (  ((score != 0) and (score > gScore))  or  ((score == 0) and (score >= gScore))  ):
                        gScore = score
                        gCategory = "cuadruple"

            if "seises" in availableCatList:
                  score = dice.count(Scorecard.Numbers["seises"]) * Scorecard.Numbers["seises"]
                  if (  ((score != 0) and (score > gScore))  or  ((score == 0) and (score >= gScore))  ):
                        gScore = score
                        gCategory = "seises"

            if "triple" in availableCatList:
                  score = sum(dice) if max(counts) >= 3 else 0
                  if (  ((score != 0) and (score > gScore))  or  ((score == 0) and (score >= gScore))  ):
                        gScore = score
                        gCategory = "triple"

            if "pupusa de frijol" in availableCatList:
                  score = 30 if (len(set([1,2,3,4]) - set(dice)) == 0 or len(set([2,3,4,5]) - set(dice)) == 0 or len(set([3,4,5,6]) - set(dice)) == 0) else 0
                  if (  ((score != 0) and (score > gScore))  or  ((score == 0) and (score >= gScore))  ):
                        gScore = score
                        gCategory = "pupusa de frijol"

            if "cincos" in availableCatList:
                  score = dice.count(Scorecard.Numbers["cincos"]) * Scorecard.Numbers["cincos"]
                  if (  ((score != 0) and (score > gScore))  or  ((score == 0) and (score >= gScore))  ):
                        gScore = score
                        gCategory = "cincos"

            if "elote" in availableCatList:
                  score = 25 if (2 in counts) and (3 in counts) else 0
                  if (  ((score != 0) and (score > gScore))  or  ((score == 0) and (score >= gScore))  ):
                        gScore = score
                        gCategory = "elote"

            if "cuatros" in availableCatList:
                  score = dice.count(Scorecard.Numbers["cuatros"]) * Scorecard.Numbers["cuatros"]
                  if (  ((score != 0) and (score > gScore))  or  ((score == 0) and (score >= gScore))  ):
                        gScore = score
                        gCategory = "cuatros"

            if "treses" in availableCatList:
                  score = dice.count(Scorecard.Numbers["treses"]) * Scorecard.Numbers["treses"]
                  if (  ((score != 0) and (score > gScore))  or  ((score == 0) and (score >= gScore))  ):
                        gScore = score
                        gCategory = "treses"

            if "doses" in availableCatList:
                  score = dice.count(Scorecard.Numbers["doses"]) * Scorecard.Numbers["doses"]
                  if (  ((score != 0) and (score > gScore))  or  ((score == 0) and (score >= gScore))  ):
                        gScore = score
                        gCategory = "doses"

            if "unos" in availableCatList:
                  score = dice.count(Scorecard.Numbers["unos"]) * Scorecard.Numbers["unos"]
                  if (  ((score != 0) and (score > gScore))  or  ((score == 0) and (score >= gScore))  ):
                        gScore = score
                        gCategory = "unos"

            # do not use it unless this is the only option, i.e.,  ( (gScore <= 0) and (score >= 5) and (score > gScore) )
            if "tamal" in availableCatList:
                  score = sum(dice)
                  if ( (gScore <= 0) and (score >= 5) and (score > gScore) ):
                        gScore = score
                        gCategory = "tamal"

            return [gScore, gCategory]


      # initialize class object, if any
      def __init__(self):
            pass  


      # first re-roll
      """
            Write an algo to know which dice are to fix to match a state. Which dice to allow for re-roll.
      """
      def first_roll(self, diceObj, scorecard):
            # let's NOT choose dice counts randomly
            _N = [0,1,2,3,4] # dice-indices
            _diceSet = []
            availableCategory = list(set(Scorecard.Categories) - set(scorecard.scorecard.keys()))
			
			# A: If current dice scores below a particular threshold then consider re-roll
            # currentDiceScoreAndCategoryList = self.firstMatchHighScoreCategory(diceObj, availableCategory)

            # B: Find minimal number of dice required to re-roll
            _diceSet = self.selectDice2(diceObj, availableCategory)

            # Should I do this? : if (len(_diceSet)==0 and (currentDiceScoreAndCategoryList[0]<5)) do random re-roll??
            return _diceSet


      # second re-roll
      """
            Write an algo to know which dice are to fix to match a state. Which dice to allow for re-roll.
      """
      def second_roll(self, diceObj, scorecard):
            # let's NOT choose dice counts randomly
            _N = [0,1,2,3,4] # dice-indices
            _diceSet = []
            availableCategory = list(set(Scorecard.Categories) - set(scorecard.scorecard.keys()))
			
			# A: If current dice scores below a particular threshold then consider re-roll
            # currentDiceScoreAndCategoryList = self.firstMatchHighScoreCategory(diceObj, availableCategory)

            # B: Find minimal number of dice required to re-roll
            _diceSet = self.selectDice2(diceObj, availableCategory)

            # Should I do this? : if (len(_diceSet)==0 and (currentDiceScoreAndCategoryList[0]<5)) do random re-roll??
            return _diceSet


      # Choose a state
      def third_roll(self, dice, scorecard):
            availableCategory = list(set(Scorecard.Categories) - set(scorecard.scorecard.keys()))

            # (1) write code to know which category has max score
            resultList = self.firstMatchHighScoreCategory(dice, availableCategory)
            nextCat = resultList[1]
            return nextCat



