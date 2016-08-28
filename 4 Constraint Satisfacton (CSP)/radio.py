"""
Write a Python program that assigns a frequency A, B, C, and D to each state, subject to the constraints that
(1) no two adjacent states share the same frequency, and
(2) the states that have legacy equipment that supports only one frequency are assigned to that frequency.

Your program should be run like this:
    python  radio.py  legacy_constraints_file
where legacy constraints file is an input to your program and has the legacy constraints listed in a format like this:
    Indiana A
    New_York B
    Washington A

The output from your program should be a file called results.txt which lists all fifty states and a frequency, in a format like:
    Alabama C
    Alaska A
    Arkansas D
...and so on.

Your code can also display results (or debugging information) to the screen, but the final line of output should be:
    Number of backtracks: x
where x is the number of times your algorithm backtracked.

-----------------------------------------------------------------------------------------------------------------------

Remember to include a detailed comments section at the top of your code that describes:

(1) a description of how you formulated the problem and how your solution works,
[mjaglan]: This problem is similar to graph-nodes coloring problem. Here we are given 4 frequencies (**colors**) which
are to be assigned to each node such that no two adjacent nodes are having same frequency. We are also given zero or more
constraints that initially some nodes are already assigned a specific frequency. I am having this as an input to my backtracking
algorithm. In backtracking algorithm, before every frequency assignment (**color**) to a node, I am checking if this is
a valid assignment. If none of the 4 values are valid assignment then backtrack. And this works as brute force solution.

To reduce the backtracks, a few optimization heuristics are added in the backtracking algorithm. They are:
<a> SELECT THE MOST CONSTRAINED VARIABLE FIRST: Select the one which has the most number of neighbors;

<b> SELECT THE LEAST CONSTRAINING VALUE FIRST: Select the one with less constraints in the neighborhood;

(2) any problems you faced, assumptions you made, etc.
[mjaglan]: Assuming That The Following Files Will Be In The Same Directory: "adjacent-states" and "sys.argv[1]", where "sys.argv[1]" will be "legacy-constraints-file"

Problem faced in handling the scenario where files are absent or empty. I partially handled this and for the rest I made some fair assumptions as stated above.

Initially it looked tricky to check the resulting graph with frequency assignment in that if it is valid but I guess the two methods were sufficent to validate
the outcome. That code is present in the debug file of this release code.


(3) a brief analysis of how well your program works and how it could be improved in the future.
[mjaglan]: Following are the stats on the performance of the algorithm after multiple runs:
for test case with legacy-constraints-1: the backtrack count is 0 with average execution time around 3.00 milliseconds (with no backtracks it is done faster)
for test case with legacy-constraints-2: the backtrack count is 0 with average execution time around 3.00 milliseconds (with no backtracks it is done faster)
for test case with legacy-constraints-3: the backtrack count is 1 with average execution time around 3.99 milliseconds (more time may be because of backtracking)

FUTURE IMPROVEMENTS:
The heuristics algorithm for selecting the LEAST CONSTRAINING VALUE FIRST can be further improved so that the backtrack count
can be further reduced (may be up to zero).


REFERECES FOR CODING THIS SOLUTION:
#1 LECTURE SLIDES: 551_lec11_csp.pptx
#2 PROGRAMMING: http://stackoverflow.com
#3 PROGRAMMING: http://www.tutorialspoint.com
#4 PROGRAMMING: https://docs.python.org/2/reference/

"""

import sys
from time import time

# ---------- globals --------------------------------------------------------------------------------------------------
globalBackTrackCount = 0   # Keep count, how many backtracks happened

# ---------- Data Structures ------------------------------------------------------------------------------------------
class graphNodes:
    myName = ""
    myNeighbors = [ ]
    myNeighborsCount = 0
    isColored = False
    myColor = ''

    contrainingValues = ['A','B','C','D']
    contrainingValuesCount = 4

    def __init__(self, iName, iNeighborList):
        self.myName = iName
        self.myNeighbors = list(iNeighborList)
        self.myNeighborsCount = len(iNeighborList)


# ---------- build graph ----------------------------------------------------------------------------------------------
def getGraph(fname):
    with open(fname) as f:
        content = f.readlines()
    f.close()
    # TODO: Handle Empty Files
    adjListDict = {}
    for eachLine in content:
        wordList = eachLine.split()
        adjListDict[wordList[0]] = graphNodes(wordList[0], wordList[1:])

    return adjListDict


# ---------- add constraints to above graph ---------------------------------------------------------------------------
def updateGraphWithConstraints(adjListDict, colAssignmentNodesList, fname):
    constraintCount = 0
    with open(fname) as f:
        content = f.readlines()
    f.close()
    for eachLine in content:
        if eachLine.strip() != '':          # WHY?: Handled Empty Files
            wordList = eachLine.split()
            adjListDict[wordList[0]].isColored = True
            adjListDict[wordList[0]].myColor = wordList[1]
            colAssignmentNodesList.extend([wordList[0]])
            constraintCount += 1
    return constraintCount


# ---------- map-coloring with backtracking ---------------------------------------------------------------------------
def recursive_backtracking(colAssignmentNodesList, cspGraph):
    global globalBackTrackCount # Needed to modify global copy of globalBackTrackCount
    if len(colAssignmentNodesList) == len(cspGraph.keys()):
        return cspGraph.keys()

    X = select_most_contrained_variable (colAssignmentNodesList, cspGraph)
    D = select_least_constraining_value (X, colAssignmentNodesList, cspGraph)
    for val in D:
        if isConsistentWithCurrentAssignments(cspGraph, X, val, colAssignmentNodesList) == True:
            colAssignmentNodesList.extend([X])
            cspGraph[X].isColored = True
            cspGraph[X].myColor = val

            result = recursive_backtracking(colAssignmentNodesList, cspGraph)
            if result is not None:
                return result

            colAssignmentNodesList.remove(X)
            cspGraph[X].isColored = False
            cspGraph[X].myColor = ''

    globalBackTrackCount += 1
    return None # Failure, hence we backTrack!


# ---------- utility methods ------------------------------------------------------------------------------------------
# Step #1: As per algorithm from slides!
def select_most_contrained_variable (colAssignmentNodesList, cspGraph):
    # TODO: Among same, select most constraining variable; assignment is not updated here.
    unassignedList = [v for v in cspGraph.keys() if v not in colAssignmentNodesList]
    #unassignedList = [v for v in cspGraph.keys() if cspGraph[v].isColored != True]
    gCount = -1
    gKey = ""
    for uv in unassignedList:
        if gCount < cspGraph[uv].myNeighborsCount:
            gCount = cspGraph[uv].myNeighborsCount
            gKey = uv
    return gKey


# Step #2: As per algorithm from slides!
def select_least_constraining_value (X, colAssignmentNodesList, cspGraph):
    contrainingValues = ['A','B','C','D'] # keep order fixed. unless it saves time!
    for eachNeighbor in cspGraph[X].myNeighbors:
        if ( eachNeighbor in colAssignmentNodesList ) :
            if ( cspGraph[eachNeighbor].myColor in contrainingValues ) :
                contrainingValues.remove(cspGraph[eachNeighbor].myColor)
    return contrainingValues


# Make sure current choosen to-be-assigned-value is not same as its neighbors!
def isConsistentWithCurrentAssignments(cspGraph, var, val, colAssignmentNodesList):
    neighbors = cspGraph[var].myNeighbors
    for eachNeighbor in neighbors:
        if ( (cspGraph[eachNeighbor].isColored == True) and (cspGraph[eachNeighbor].myColor == val) ):
            return False
    return True


# ---------- main method ----------------------------------------------------------------------------------------------
def main():
    # Assuming That The Following Files Will Be In The Same Directory: "adjacent-states" and "sys.argv[1]", where "sys.argv[1]" will be "legacy-constraints-file"

    if (2 != len(sys.argv)):
        print "INPUT: Please provide legacy-constraints-file as the first and only argument"
        return -1

    # get the graph data, suck it into an adjacency-list as a dictionary, use this dictionary
    graphDataFile = "adjacent-states"
    cspGraph = getGraph(graphDataFile)

    # Visited/ NOT-Visited checklist
    colAssignmentNodesList = []

    # get the graph data, read through it, update the adjacency-list and the colAssignmentNodesList
    constraintDataFile = sys.argv[1]
    legacyConstraintsCount = updateGraphWithConstraints(adjListDict=cspGraph, colAssignmentNodesList=colAssignmentNodesList, fname=constraintDataFile)

    # EXECUTE GRAPH COLORING
    responseStatus = recursive_backtracking(colAssignmentNodesList, cspGraph)

    # OUTPUT: write formatted output to "results.txt" file
    resultFileObj = open('results.txt','w')
    for eachItem in sorted(cspGraph.keys()):
        resultFileObj.write (cspGraph[eachItem].myName + ' ' + cspGraph[eachItem].myColor + '\n')
    resultFileObj.close()
    print("Number of backtracks: " + str(globalBackTrackCount))


# ---------- program start point --------------------------------------------------------------------------------------
main()

