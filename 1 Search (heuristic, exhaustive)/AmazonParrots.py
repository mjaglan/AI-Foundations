# create a list of 4 lists - each sub-list is of size 4
# Brute Force: make combinations and verify it against verification function

# verification function is based on below mentioned TEN COMMANDMENTS

"""
TEN COMMANDMENTS:

0 The customer on Maxwell Street received the Amplifier.
1 The Elephant arrived in North Avenue;
6 Frank received a Doorknob.
4 George's package went to Kirkwood Street ====> G does not live on K

2 the customer who ordered the Banister received the package that Irene had ordered.
3 The delivery that should have gone to Kirkwood Street was sent to Lake Avenue.
5 The customer who ordered the Candelabrum received the Banister, while

7 the person who had ordered Elephant received the package that should have gone to Maxwell Street.

8 Heather received the package that was to go to Orange Drive.

9 Jerry received Heather's order.

"""

# ASSUMPTION: Positions of streets is fixed.
# LOGIC: just check if the triplet (WRONG DELIVERY, NAME, ACTUAL ORDER) is at a place
# where above mentioned conditions are not violated.
def verifyState (iWorld2):

    # A cannot be on M
    if ((iWorld2[2][2] != " " and iWorld2[2][2] == "A") or (iWorld2[1][2] != " ") and iWorld2[1][2] != "A"):
#        print "1"
        return False

    # E cannot be on N
    if ((iWorld2[2][3] != " " and iWorld2[2][3] == "E") or (iWorld2[1][3] != " " and iWorld2[1][3] != "E")):
 #       print "2"
        return False

    # F has not ordered D
    iList3 = iWorld2[3]
    idx = iList3.index("F") if "F" in iList3 else None
    if (idx is not None and iWorld2[2][idx] == "D"):
  #      print "3"
        return False

    # G cannot live on K
    if (iWorld2[3][0] != " " and iWorld2[3][0] == "G"):
   #     print "4"
        return False

    # H cannot live on O
    if (iWorld2[3][4] != " " and iWorld2[3][4] == "H"):
    #    print "5"
        return False

    # E cannot be on M
    if (iWorld2[2][2] != " " and iWorld2[2][2] == "E"):
   #     print "6"
        return False

    # I has not ordered B
    iList3 = iWorld2[3]
    idx = iList3.index("I") if "I" in iList3 else None
    if (idx is not None and iWorld2[2][idx] == "B"):
   #     print "7"
        return False

    # I's order was delivered to Owner of B
    iList1 = iWorld2[1] # Wrong
    iList2 = iWorld2[2] # Ordered
    iList3 = iWorld2[3] # Name
    idxI =  iList3.index("I") if "I" in iList3 else None
    if idxI is not None:
        idxPI = iList2[idxI]
        idxB  = iList2.index("B") if "B" in iList2 else None
        if (idxB is not None and iList1[idxB] != idxPI):
   #         print "8"
            return False

    # B should be delivered where C was ordered
    iList1 = iWorld2[1] # Wrong
    iList2 = iWorld2[2] # Ordered
    iList3 = iWorld2[3] # Name
    idxI =  iList2.index("C") if "C" in iList2 else None
    if (idxI is not None and iList1[idxI] != "B"):
   #     print "9"
        return False

    # K's order went to L
    iList1 = iWorld2[1] # Wrong
    iList2 = iWorld2[2] # Ordered
    iList3 = iWorld2[3] # Name
    idxI =  iList2.index("C")  if "C" in iList2 else None
    if (idxI is not None and iList2[0] != iList1[1]):
   #     print "10"
        return False

    # H's order was delivered to J
    iList1 = iWorld2[1] # Wrong
    iList2 = iWorld2[2] # Ordered
    iList3 = iWorld2[3] # Name
    idxH =  iList3.index("H") if "H" in iList3 else None
    idxJ =  iList3.index("J") if "J" in iList3 else None
    if (idxH is not None and idxJ is not None):
        if (iList2[idxH] != iList1[idxJ]):
   #         print "11"
            return False

    # G's order was delivered to K
    iList1 = iWorld2[1] # Wrong
    iList2 = iWorld2[2] # Ordered
    iList3 = iWorld2[3] # Name
    idxH =  iList3.index("G") if "G" in iList3 else None
    if (idxH is not None):
        if (iList2[idxH] != iList1[0]):
   #         print "13"
            return False

    # G's order was delivered to K
    iList1 = iWorld2[1] # Wrong
    iList2 = iWorld2[2] # Ordered
    iList3 = iWorld2[3] # Name
    idxH =  iList2.index("C") if "C" in iList2 else None
    if (idxH is not None):
        if ("B" != iList1[idxH]):
   #         print "14"
            return False

    # The person who had ordered E received the package that should have gone to M.
    iList1 = iWorld2[1] # Wrong
    iList2 = iWorld2[2] # Ordered
    iList3 = iWorld2[3] # Name
    idxH =  iList2.index("E") if "E" in iList2 else None
    if (idxH is not None):
        if (iList2[2] != iList1[idxH]):
   #         print "14"
            return False

    # F received a D.
    iList1 = iWorld2[1] # Wrong
    iList2 = iWorld2[2] # Ordered
    iList3 = iWorld2[3] # Name
    idxH =  iList3.index("F") if "F" in iList3 else None
    if (idxH is not None):
        if (iList1[idxH] != "D"):
   #         print "15"
            return False

#    print "16"
    return True


def printProperly(solutionMatrix):
    symbolTable = {
                    "A" : "Amplifier",
                    "B" : "Banister",
                    "C" : "Candelabrum",
                    "D" : "Doorknob",
                    "E" : "Elephant",
                    "F" : "Frank",
                    "G" : "George",
                    "H" : "Heather",
                    "I" : "Irene",
                    "J" : "Jerry",
                    "K" : "Kirkwood Steet",
                    "L" : "Lake Avenue",
                    "M" : "Maxwell Street",
                    "N" : "North Avenue",
                    "O" : "Orange Drive"
                }
    i = 0
    while (i<5):
        output = symbolTable[solutionMatrix[3][i]] + " lives on " + symbolTable[solutionMatrix[0][i]] + " and actually ordered " + symbolTable[solutionMatrix[2][i]] + "." # + " BUT " + symbolTable[solutionMatrix[1][i]] + " was delivered!"
        print output
        i = i+1


def getThingsDone(world, deliverList, personsList, ordersList, i, dd , oo , pp, count):
    if (i<5):
        for d in deliverList:
            world[1][i] = d
            for o in ordersList:
                world[2][i] = o
                for p in personsList:
                    world[3][i] = p
                    getThingsDone(world, list(set(deliverList)-set(world[1][dd])), list(set(personsList)-set(world[3][pp])), list(set(ordersList)-set(world[2][oo])), 1+i, 1+dd, 1+oo, 1+pp, count)
    else:
        status = verifyState(world)
        if (status):
            print "\n"
            printProperly(world)
            print "\n"
            return

def main():
    # INITIAL STATE: We do not know any correct position initially. We just start filling and at the same time verify our combination against the 10 rules mentioned in the question
    world1 = [
        #   ["0","1","2","3","4"]
        ["K","L","M","N","O"], # 0 FIXED STREET
        [" "," "," "," "," "], # 1 STREET WRONG DELIVERY
        [" "," "," "," "," "], # 2 ORDER
        [" "," "," "," "," "]  # 3 PERSON
    ]

    # swap-fill-verify until it is True is received
    deliverList  = set(["A","B","C","D","E"]) # constant
    personsList  = set(["F","G","H","I","J"]) # constant
    ordersList   = set(["A","B","C","D","E"]) # constant

    getThingsDone(world1, deliverList, personsList, ordersList, 0, 0 , 0 , 0, 0)



main()

