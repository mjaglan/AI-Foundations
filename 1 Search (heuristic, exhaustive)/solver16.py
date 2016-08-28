'''
heuristic = max(horizontal move required to put a cell in correct position ) + max(vertical move required to put a cell in correct position)
'''

import sys
import math
import Queue

class Node:
    def __init__(self, board):
        self.board = board
        self.parent = None


def ParseBoard(filename):
    board = []
    for line in open(filename):
        nums = line.split(" ")
        row = []
        for num in nums:
            row.append(int(num))
        board.append(row)
    return board

def PrintBoard(board):
    for row in board:
        print row
    print "\n"

def GetHeuristicValue(board):
    maxDeltaI = 0
    maxDeltaJ = 0
    for i in range(0,4):
        for j in range(0,4):
            num = board[i][j]
            (goalI,goalJ) = ((num-1)/4,(num-1)%4)
            deltaI =  int(math.fabs(i - goalI))
            deltaJ = int(math.fabs(j - goalJ))
            if deltaI == 3:
                deltaI = 1;
            if deltaJ == 3:
                deltaJ = 1
            if maxDeltaI < deltaI:
                maxDeltaI = deltaI
            if maxDeltaJ < deltaJ:
                maxDeltaJ = deltaJ
            #print num, goalI, goalJ, deltaI, deltaJ
    return maxDeltaI + maxDeltaJ

def CopyBoard(board):
    successor = []
    for row in board:
        newRow = []
        for num in row:
            newRow.append(num)
        successor.append(newRow)
    return successor

def RotateRow(board, row, direction):
    successor = CopyBoard(board)
    for i in range(0,4):
        successor[row][(i+direction+4)%4] = board[row][i]
    return successor

def RotateCol(board, col, direction):
    successor = CopyBoard(board)
    for i in range(0,4):
        successor[i][col] = board[(i+direction+4)%4][col]
    return successor

def GenerateSuccessors(board):
    successors = []
    for i in range(0,4):
        successor = Node(RotateRow(board, i, 1))
        successor.move = "R"+str(i+1)
        successors.append(successor)

        successor = Node(RotateRow(board, i, -1))
        successor.move = "L"+str(i+1)
        successors.append(successor)

        successor = Node(RotateCol(board, i, 1))
        successor.move = "U"+str(i+1)
        successors.append(successor)

        successor = Node(RotateCol(board, i, -1))
        successor.move = "D"+str(i+1)
        successors.append(successor)

    return successors

def EqualBoard(board1,board2):
    for i in range(0,4):
        for j in range(0,4):
            if board1[i][j] != board2[i][j]:
                return False
    return True
def Expanded(board, expandedNodes):
    for node in expandedNodes:
        if EqualBoard(board,node.board):
            return True
    return False

def PrintPath(startNode,goalNode):
    #print "Path:"
    pathStr = ""
    path = []
    current = goalNode
    while current.parent!=None:
        path.append(current)
        current = current.parent
    path.append(startNode)
    path.reverse()
    for node in path:
        #PrintBoard(node.board)
        if node.move:
            #print node.move
            pathStr += node.move+" "
    print pathStr

def AStar(board,goal):
    q = Queue.PriorityQueue()
    h = GetHeuristicValue(board)
    g = 0
    f = g + h
    startNode = Node(board)
    startNode.move = None
    q.put((f,g,startNode))
    closedNodes = []
    while q.not_empty:
        f,prev_g,currentNode = q.get()
        #print "Expanding:", current.name
        if EqualBoard(currentNode.board,goal):
            PrintPath(startNode,currentNode)
            return
        closedNodes.append(currentNode)
        #print "Move:", currentNode.move
        #print "G:",g,"H",h,"F:",f
        #PrintBoard(currentNode.board)

        #print "Successors:"

        for successorNode in GenerateSuccessors(currentNode.board):
            #PrintBoard(successorNode.board)
            if Expanded(successorNode.board,closedNodes):
                continue
            g = prev_g + 1
            h = GetHeuristicValue(successorNode.board)
            f = g + h
            successorNode.parent = currentNode
            q.put((f, g, successorNode))



def main():
    goal = [
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12],
        [13,14,15,16]
    ]
    board = ParseBoard(sys.argv[1])
    AStar(board,goal)

main()
