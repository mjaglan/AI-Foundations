"""
Analysis of the results:
As the value of 'k' increases the size of tree inreases considerably very much.
Time taken to arrange the numbers in the tree is dependent on how the algorithm works + the size of the input.

In our case, algorithm takes more and more time to find the lowest possible score as the size of tree increases;
an exception to this trend is the k==5 value for which our algorithm finds the minimal possible score in lesser
time it took for k==4; this may be due to the way initial population of numbers alloted to the tree - the pattern
of alloting numbers in the tree made more pruning of extra computations it otherwise could have done.


(1) the best solution this code is able to find for k = 3, 4, 5, 6, and 7,
    including both the score and the actual tile arrangement;

BEST SOLUTION FOUND FOR K=3:
SCORE: 3
BFS TREE: 5  2  6  8  1  3  4  7  9  10


BEST SOLUTION FOUND FOR K=4:
SCORE: 4
BFS TREE: 11  7  12  15  3  4  8  16  18  19  1  2  5  6  9  10  13  14  17  20  21  22


BEST SOLUTION FOUND FOR K=5:
SCORE: 6
BFS TREE: 23  17  24  29  11  12  18  30  34  35  5  6  7  8  15  16  31  32  38  39  40  41  1  2  3  4  9  10  13  14  19  20  21  22  25  26  27  28  33  36  37  42  43  44  45  46


BEST SOLUTION FOUND FOR K=6:
SCORE: 10
BFS TREE: 47  37  48  57  27  28  38  58  66  67  17  18  19  25  31  32  63  64  74  75  76  77  7  8  9  10  11  22  20  26  33  34  35  36  59  60  61  62  80  65  82  71  84  85  86  87  1  2  3  4  5  6  15  16  12  21  23  24  13  14  29  30  39  40  41  42  43  44  45  46  49  50  51  52  53  54  55  56  81  70  68  69  83  72  73  78  79  88  89  90  91  92  93  94


BEST SOLUTION FOUND FOR K=7:
SCORE: 19
BFS TREE: 95  79  96  111  63  64  80  112  126  127  47  48  49  50  65  66  125  128  140  141  142  143  31  32  33  34  41  45  53  57  61  62  67  75  115  124  129  130  134  137  139  155  156  157  158  159  15  16  17  18  19  20  21  39  35  42  36  46  37  54  38  58  69  70  71  72  73  74  76  91  116  99  117  118  119  120  121  122  152  131  153  135  154  144  166  147  168  169  170  171  172  173  174  175  1  2  3  4  5  6  7  8  9  10  11  12  13  14  22  40  23  24  43  44  25  26  51  52  27  28  55  56  29  30  59  60  77  78  81  82  83  84  85  86  87  88  89  90  68  92  93  94  123  100  97  98  101  102  103  104  105  106  107  108  109  110  113  114  160  161  132  133  162  163  136  138  164  165  145  146  167  150  148  149  151  176  177  178  179  180  181  182  183  184  185  186  187  188  189  190




(2) the amount of time it took to find each solution.

k       time taken (approx micro seconds)
3       16000
4       281000
5       66000
6       837000
7       7525000



REFERENCES FOR CODING:
BOOK: Artificial Intelligence: A Modern Approach (3rd Edition) by Stuart Russell and Peter Norvig

PROGRAMMING:
http://stackoverflow.com
http://www.tutorialspoint.com
https://docs.python.org/2/reference/

"""


# --------------------------------------------------------- PACKAGES ---------------------------------------------------
import sys
import math
import itertools
import time

# --------------------------------------------------------- UTILITY METHODS --------------------------------------------

def getScore(inList):
    if type(inList) is not list:
        return 0
    _size = len(inList)
    gListScore = -sys.maxint - 1

    if ( _size <= 1 ):
        gListScore = 0

    else:
        parentIndex    = 0
        currentNodeIdx = 1
        while(currentNodeIdx < _size):
            pEdge = -sys.maxint - 1
            lEdge = -sys.maxint - 1
            rEdge = -sys.maxint - 1

            if ((0 < currentNodeIdx) and (0 <= parentIndex) and (parentIndex < _size)):
                pEdge = abs(inList[currentNodeIdx]-inList[parentIndex])

            leftChildIdx   = 2*(currentNodeIdx + 1)
            if (leftChildIdx < _size):
                lEdge = abs(inList[currentNodeIdx]-inList[leftChildIdx])

            rightChildIdx  = 2*(currentNodeIdx + 1) + 1
            if (rightChildIdx < _size):
                rEdge = abs(inList[currentNodeIdx]-inList[rightChildIdx])

            lScore = max(pEdge, lEdge, rEdge)
            if (lScore > gListScore):
                gListScore = lScore

            currentNodeIdx = currentNodeIdx+1
            parentIndex = int((currentNodeIdx-2)/2)

    return gListScore



def getNodesAtK(i):
    if (i > 0):
        nodesCount = int(math.floor( 3 * math.pow(2, i-2) ))
    else:
        nodesCount = 0
    return nodesCount



def getNodesCount(k):
    if (k > 0):
        i=1
        _sum = 0
        while(i <= k):
            _sum = _sum + getNodesAtK(i)
            i = i+1
    else:
        _sum = 0

    return _sum


# get the median
def getMidElementsList(inList):
    inList.sort(key=int)
    n = len(inList) # 0 to (n-1)
    if (n%2 == 0):
        midIdx = [((n/2)-1), (((n/2)+1)-1)]
        _elements = [inList[midIdx[0]], inList[midIdx[1]]]
    else:
        midIdx = [(n/2)]
        _elements = [inList[midIdx[0]]]
    return _elements



# get individual score of a node
def myScore(inList, currentNodeIdx):
    if type(inList) is not list:
        print("ERROR: INPUT IS NOT A LIST")
        return 0

    _size = len(inList)
    myScore = -sys.maxint - 1

    if ( _size <= 1 ):
        myScore = 0

    else:
        pEdge = -sys.maxint - 1
        lEdge = -sys.maxint - 1
        rEdge = -sys.maxint - 1

        parentIndex = int((currentNodeIdx-1)/2)
        if ((0 < currentNodeIdx) and (0 <= parentIndex) and (parentIndex < _size)):
            pEdge = abs(inList[currentNodeIdx]-inList[parentIndex])

        leftChildIdx   = (2*currentNodeIdx + 1)
        if (leftChildIdx < _size):
            lEdge = abs(inList[currentNodeIdx]-inList[leftChildIdx])

        rightChildIdx  = (2*currentNodeIdx + 1) + 1
        if (rightChildIdx < _size):
            rEdge = abs(inList[currentNodeIdx]-inList[rightChildIdx])

        lScore = max(pEdge, lEdge, rEdge)
        if (lScore > myScore):
            myScore = lScore

    return myScore




def checkCorrectFill(result__list, idx, mySetList):
    if (0 == result__list[idx]):
        result__list[idx] = mySetList.pop(0)


# top-down approach: try to assign numbers to each node such that its personal score is "k_bar" or a bit close it (if it has to be a bigger score)
def makeTreeOfScore_td_1 (N, k, k_bar):
    if ( (N <= 0) or (k <= 0) ):
        return []

    if ((N==1) or (k==1)):
        return range(1,2,1)

    result__list = [0] * N
    mySetList = range(1,N+1,1)

    iK = k
    X = len(result__list)

    # find mid elements
    medianList = getMidElementsList(mySetList)

    # choose 1st element as the head at k==1
    result__list[0] = medianList.pop(0)
    mySetList.remove(result__list[0])

    # choose 1st element as the head at k==2
    leftHeadIdx  = 1
    midHeadIdx   = 2
    rightHeadIdx = 3
    result__list[midHeadIdx] = medianList.pop()
    mySetList.remove(result__list[midHeadIdx])

    # handle left-Sub-Tree
    # LEFT HEAD: set the head
    for i in range((result__list[0] - k_bar), N+1, 1):
        if i in mySetList:
            result__list[leftHeadIdx] = i
            mySetList.remove(i)
            break
    checkCorrectFill(result__list, leftHeadIdx, mySetList) # (reference, copy, reference)

    # handle right-Sub-Tree
    # RIGHT HEAD: set the head
    for i in range((result__list[0] + k_bar), 0, -1):
        if i in mySetList:
            result__list[rightHeadIdx] = i
            mySetList.remove(i)
            break
    checkCorrectFill(result__list, rightHeadIdx, mySetList) # (reference, copy, reference)

    # handle mid-tree
    # MID HEAD: already present at result__list[2]



    # LEFT TREE: populate it in top-down manner
    myQueue = []
    myQueue = myQueue + [leftHeadIdx]
    while (len(myQueue) > 0):
        parentIndex = myQueue.pop(0)

        lChildIdx = ((parentIndex*2) + 2)
        if (lChildIdx < N):
            myQueue = myQueue + [lChildIdx]
            for i in range((result__list[parentIndex] - k_bar), N+1, 1):
                if i in mySetList:
                    result__list[lChildIdx] = i
                    mySetList.remove(i)
                    break
            checkCorrectFill(result__list, lChildIdx, mySetList) # (reference, copy, reference)
        else:
            lChildIdx = -1

        rChildIdx = ((parentIndex*2) + 3)
        if (rChildIdx < N):
            myQueue = myQueue + [rChildIdx]
            for i in range((result__list[lChildIdx]), N+1, 1):
                if i in mySetList:
                    result__list[rChildIdx] = i
                    mySetList.remove(i)
                    break
            checkCorrectFill(result__list, rChildIdx, mySetList) # (reference, copy, reference)
        else:
            rChildIdx = -1


    # RIGHT TREE: populate it in top-down manner
    myQueue = []
    myQueue = myQueue + [rightHeadIdx]
    while (len(myQueue) > 0):
        parentIndex = myQueue.pop(0)

        rChildIdx = ((parentIndex*2) + 3)
        if (rChildIdx < N):
            myQueue = myQueue + [rChildIdx]
            for i in range((result__list[parentIndex] + k_bar), 0, -1):
                if i in mySetList:
                    result__list[rChildIdx] = i
                    mySetList.remove(i)
                    break
            checkCorrectFill(result__list, rChildIdx, mySetList) # (reference, copy, reference)
        else:
            rChildIdx = -1

        lChildIdx = ((parentIndex*2) + 2)
        if (lChildIdx < N):
            myQueue = myQueue + [lChildIdx]
            for i in range((result__list[rChildIdx]), 0, -1):
                if i in mySetList:
                    result__list[lChildIdx] = i
                    mySetList.remove(i)
                    break
            checkCorrectFill(result__list, lChildIdx, mySetList) # (reference, copy, reference)
        else:
            lChildIdx = -1


    # MID TREE: break it and make it!
    # MID LEFT HEAD: set the head
    midLeftHeadIdx = ((midHeadIdx*2)+2)
    if midLeftHeadIdx < N:
        for i in range((result__list[midHeadIdx] - k_bar), N+1, 1):
            if i in mySetList:
                result__list[midLeftHeadIdx] = i
                mySetList.remove(i)
                break
        checkCorrectFill(result__list, midLeftHeadIdx, mySetList) # (reference, copy, reference)

    # MID RIGHT HEAD: set the head
    midRightHeadIdx = ((midHeadIdx*2)+3)
    if midRightHeadIdx < N:
        for i in range((result__list[midHeadIdx] + k_bar), 0, -1):
            if i in mySetList:
                result__list[midRightHeadIdx] = i
                mySetList.remove(i)
                break
        checkCorrectFill(result__list, midRightHeadIdx, mySetList) # (reference, copy, reference)

    if (midLeftHeadIdx < N):
        # MID LEFT TREE: populate it in top-down manner
        myQueue = []
        myQueue = myQueue + [midLeftHeadIdx]
        while (len(myQueue) > 0):
            parentIndex = myQueue.pop(0)

            lChildIdx = ((parentIndex*2) + 2)
            if (lChildIdx < N):
                myQueue = myQueue + [lChildIdx]
                for i in range((result__list[parentIndex] - k_bar), N+1, 1):
                    if i in mySetList:
                        result__list[lChildIdx] = i
                        mySetList.remove(i)
                        break
                checkCorrectFill(result__list, lChildIdx, mySetList) # (reference, copy, reference)
            else:
                lChildIdx = -1

            rChildIdx = ((parentIndex*2) + 3)
            if (rChildIdx < N):
                myQueue = myQueue + [rChildIdx]
                for i in range((result__list[lChildIdx]), N+1, 1):
                    if i in mySetList:
                        result__list[rChildIdx] = i
                        mySetList.remove(i)
                        break
                checkCorrectFill(result__list, rChildIdx, mySetList) # (reference, copy, reference)
            else:
                rChildIdx = -1

    if (midRightHeadIdx < N):
        # MID RIGHT TREE: populate it in top-down manner
        myQueue = []
        myQueue = myQueue + [midRightHeadIdx]
        while (len(myQueue) > 0):
            parentIndex = myQueue.pop(0)

            rChildIdx = ((parentIndex*2) + 3)
            if (rChildIdx < N):
                myQueue = myQueue + [rChildIdx]
                for i in range((result__list[parentIndex] + k_bar), 0, -1):
                    if i in mySetList:
                        result__list[rChildIdx] = i
                        mySetList.remove(i)
                        break
                checkCorrectFill(result__list, rChildIdx, mySetList) # (reference, copy, reference)
            else:
                rChildIdx = -1

            lChildIdx = ((parentIndex*2) + 2)
            if (lChildIdx < N):
                myQueue = myQueue + [lChildIdx]
                for i in range((result__list[rChildIdx]), 0, -1):
                    if i in mySetList:
                        result__list[lChildIdx] = i
                        mySetList.remove(i)
                        break
                checkCorrectFill(result__list, lChildIdx, mySetList) # (reference, copy, reference)
            else:
                lChildIdx = -1


    # Optimize the whole tree from (k-2) to k

    gScore = getScore(result__list)
    gList = list(result__list)

    # permute sub trees from (k-2) to k
    subTreeAtKMinus2 = getNodesAtK(k-2)           # total sub-trees at (k-2) level
    headIdxAtKMinus2 = getNodesCount(k-3) - 1     # index of each sub-trees at (k-2) level; initialize to a negative index

    while(subTreeAtKMinus2 > 0):
        subTreeList = [] # it's just 7 items

        headIdxAtKMinus2 = headIdxAtKMinus2 + 1   # move to current sub-tree head, start permute operation on it, put the best sub-tree back;
        idx = headIdxAtKMinus2

        myQueue = []
        myQueue = myQueue + [result__list[idx]]
        i = 0
        while(len(myQueue) > 0):

            subTreeList = subTreeList + [myQueue.pop(0)]
            idx = result__list.index(subTreeList[-1])
            if (2*idx + 2) < N:
                myQueue = myQueue + [result__list[(2*idx + 2)]]
            if (2*idx + 3) < N:
                myQueue = myQueue + [result__list[(2*idx + 3)]]
            i = i+1

        parentIndex = int((headIdxAtKMinus2-2)/2)
        delta = abs(result__list[parentIndex] - subTreeList[0])
        listScore = max(myScore(subTreeList,1), myScore(subTreeList,2)) # for a size 7 list

        if ( max(delta, listScore) > k_bar ):
            lowScoreCache = sys.maxint

            for tpls in itertools.permutations(subTreeList):

                p = list(tpls)
                delta = abs(result__list[parentIndex] - p[0])
                listScore = max(myScore(p,1), myScore(p,2))

                if ( max(delta, listScore) <= k_bar ):
                    # put the permuted list back
                    idx = headIdxAtKMinus2
                    myQueue = []
                    myQueue = myQueue + [idx]

                    while(len(p) > 0):
                        idx = myQueue.pop(0)
                        result__list[idx] = p.pop(0)
                        if (2*idx + 2) < N:
                            myQueue = myQueue + [(2*idx + 2)]
                        if (2*idx + 3) < N:
                            myQueue = myQueue + [(2*idx + 3)]
                    # We got what we want, now just exit
                    break

                else:
                    # handle the situation gracefully, save the best record
                    if lowScoreCache > max(delta,listScore):
                        lowScoreCache = max(delta,listScore)

                        idx = headIdxAtKMinus2
                        myQueue = []
                        myQueue = myQueue + [idx]

                        while(len(p) > 0):
                            idx = myQueue.pop(0)
                            result__list[idx] = p.pop(0)
                            if (2*idx + 2) < N:
                                myQueue = myQueue + [(2*idx + 2)]
                            if (2*idx + 3) < N:
                                myQueue = myQueue + [(2*idx + 3)]
                    #break # do not break out until you have explored all your options

        # target next sub-tree at level (k-2)
        subTreeAtKMinus2 = subTreeAtKMinus2 - 1

    rScore = getScore(result__list)
    if rScore < gScore:
        return result__list
    else:
        return gList



# IDEA: Find and arrangement with lowest score of 'i' for a 'k' depth tree
# Kind of a simple local search
def wrapper_makeTreeOfScore_kBar (N, k):

    globalScore = sys.maxint
    globalList = [0]*N

    localScore = sys.maxint
    localList = [0]*N

    for i in range(k, N+1, 1):
        localList = list(makeTreeOfScore_td_1 (N, k, i))

        localScore = getScore(localList)
        if(localScore <= globalScore):
            globalList = localList
            globalScore = localScore

    return globalList



def printOutput(resultList):
        print( getScore(resultList) )
        myStr0 = str(resultList)

        myStr1 = myStr0.replace("[", "")
        myStr2 = myStr1.replace("]", "")

        myStr3 = myStr2.replace(",", " ")

        print( myStr3 )
        print("\n")


# --------------------------------------------------------- STUBS ------------------------------------------------------
def myTest():
    print("DEMOSTRATION FOR k = 0 to 7 :")

    for i in range(0,8,1):
        k = i
        N = getNodesCount(k)
        resultList =  wrapper_makeTreeOfScore_kBar (N, k)
        printOutput(resultList)



def main():
    # start time recorded
    #start = int(round(time.time() * 1000000))

    # No custom input: just the method name
    if len(sys.argv) == 1:
        myTest();

    # 1 or more custom input
    elif len(sys.argv) >= 2:
        k = int(sys.argv[1])  # 1st custom input
        N = getNodesCount(k)

        # 1 custom input: k
        if len(sys.argv) == 2:
            resultList =  wrapper_makeTreeOfScore_kBar (N, k)

        # 2 or more custom input: (k, k_bar)
        else:
            k_bar = int(sys.argv[2]) # 2nd custom input
            resultList = list(makeTreeOfScore_td_1 (N, k, k_bar))

        # print formatted output for 1 or more custom input
        printOutput(resultList)

    # end time recorded
    #end = int(round(time.time() * 1000000))
    #print("Time : " + str(end-start) + " micro seconds")



# --------------------------------------------------------- START MAIN -------------------------------------------------
main()




