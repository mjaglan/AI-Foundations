'''

ASTAR (with segment heuristic):
We used (straight_line_distance_to_goal/max_len) as the heuristic. Here max_len is the length of longest road in whole data set. The idea is that, 
the agent must go through at least that many of edges to reach goal.

ASTAR (with distance heuristic):
We used straight_line_distance_to_goalas the heuristic for obvious reason.

ASTAR (with time heuristic):
We used (straight_line_distance_to_goal/max_speed) as the heuristic. Here max_speed is the maximum speed given in whole data set. The agent must spend at least that many of hours to reach goal.
 
Handling Wrong Data: If gps for a city is missing, then if it has a neighboring city with gps values, we used the same values for the first city. This did not work always, as there were cities with
no other neighboring city with gps. In that case we removed that city from node graph. We though about going further, but using gps from a distant city did not seem reasonable. For road with missing speed/length, we used the maximum values for them. Since many of the nodes/edges are missing, algorithms do not run as expected for certain input values. 

On average, BFS took least time to complete searching, followed by A*. In almost half of our test cases, DFS did not finish search after waiting a long time. Both A* and BFS found optimal solution from routing option 'segments', but A* took longer time.

One significant place of improvement of our code would be  to try to estimate values for missing data.

We used graph search version of AStar, because all of our heuristics are consistence

REFERENCES FOR CODING:
BOOK: Artificial Intelligence: A Modern Approach (3rd Edition) by Stuart Russell and Peter Norvig

WEB:
CONCEPTS: https://en.wikipedia.org/wiki/Admissible_heuristic
CONCEPTS: https://en.wikipedia.org/wiki/Consistent_heuristic

PROGRAMMING:
http://stackoverflow.com
http://www.tutorialspoint.com
https://docs.python.org/2/reference/



'''

import Queue
import math
import sys
import random

global max_len 
max_len= 1
global max_speed 
max_speed = 1

class City:
    def __init__(self, name, latitude, longitude):
        self.name = name        
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.expanded = False
        self.roads = [] # keeps the list of roads connected to it ( list of Road class objects )


class Road:
    def __init__(self, city1, city2, length, speed, name):
        self.city1 = city1
        self.city2 = city2
        if not length or length=='':            
            self.length = -1
        else:
            self.length = float(length)
        if not speed  or speed=='':
            self.speed = -1
        else:
            self.speed = float(speed)
        self.name = name


def parse_cities(filename, cities):
    for line in open(filename):
        tokens = line.split(' ')
       # tokens = [x for x in tokens if x]
       # if len(tokens) < 3:
           # continue
        city = City(tokens[0],tokens[1],tokens[2])
        cities[city.name] = city


def parse_roads(filename, cities):
    global max_speed
    global max_len
    cities_without_gps = set()
    for line in open(filename):
        tokens = line.split(' ')
      #  tokens = [x for x in tokens if x]
        #print line
       # if len(tokens) <5:
            #continue           
        city1 = tokens[0]
        city2 = tokens[1]
        forward_road = Road(city1,city2,tokens[2].strip(),tokens[3].strip(),tokens[4].strip())
        backward_road = Road(city2,city1,tokens[2].strip(),tokens[3].strip(),tokens[4].strip())
        if city1 in cities:
            cities[city1].roads.append(forward_road)
        else:
            cities_without_gps.add(city1)
        if city2 in cities:
            cities[city2].roads.append(backward_road)
        else:
            cities_without_gps.add(city2)
            
	if max_len < forward_road.length :
            max_len = forward_road.length
	if max_speed < forward_road.speed:
	    max_speed = forward_road.speed
    return cities_without_gps

def AddMissingCities(cities,missingCities):
    for missingCity in missingCities:
        #print "Adding city:",missingCity
        minLen = sys.float_info.max
        closestCity = None
        roads = []
        for cityName in cities:
            city = cities[cityName]
            for road in city.roads:
                if road.city2 == missingCity:
                    if road.length<minLen:
                        minLen = road.length
                        closestCity = city
                    roads.append(Road(missingCity,city.name,road.length,road.speed,road.name))
        if closestCity is not None:
            newCity = City(missingCity,random.uniform(-0.0,0.0)+ closestCity.latitude,random.uniform(-0.0,0.0)+ closestCity.longitude)
            newCity.roads = roads
            cities[newCity.name]=newCity
        else:
            pass
            #print "No closes city found for:",missingCity
                        
                    
def AddMissingRoadInfo(cities):
    global max_len
    global max_speed
    for cityName in cities:
        city = cities[cityName]
        for road in city.roads:
            if road.length == -1 or road.length==0:
                road.length = max_len
            if road.speed == -1 or road.speed==0:
                road.speed = max_speed
                


# distances using GPS coordinates
# idea and most of the code were taken from http://www.movable-type.co.uk/scripts/latlong.html
# and http://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula
def GetStDis(city1,city2):
    earthRad = 6371000
    dLat = math.radians(city1.latitude -  city2.latitude)
    dLng = math.radians(city1.longitude -  city2.longitude)
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(math.radians(city1.latitude)) * math.cos(math.radians(city2.latitude)) * math.sin(dLng/2) * math.sin(dLng/2);
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a));
    dist = (float) (earthRad * c)
    return dist/1.60934
    
def GetLowestNumOfEdges(city1,city2): 
    global max_len
    return GetStDis(city1,city2)/max_len
    
def GetLowestAmtOfTime(city1,city2): 
    global max_speed
    return GetStDis(city1,city2)/max_speed

def GetHrStr(val):
    val = int(val*60)
    out =" "
    hr = int(val/60)
    if hr>1:
        out+=str(hr)+" hours"
    else:
        out+=str(hr)+" hour"
    min = str(int(val%60))
    return out+" "+str(min)+" min"


def PrintPath(parent, startCity, endCity):
    reverse_path = [(endCity,None)]
    current  = endCity.name
    while current != startCity.name:
        reverse_path.append(parent[current])
        current = parent[current][0].name
    reverse_path.reverse()
    total_dist = 0
    total_time = 0
    humanOutputStr = ""
    machineOutputStr = ""
    for city,road in reverse_path:
        if road:
            total_time += road.length/road.speed
            #print total_time
            total_dist += road.length
            humanOutputStr+= "Take "+road.name+" to "+road.city2+ "(" + str(road.length)+ " miles " + GetHrStr(road.length/road.speed)+" )\n"
            machineOutputStr += city.name+" "
        else:
            humanOutputStr += "Arrive at "+endCity.name
            machineOutputStr += endCity.name

    humanOutputStr = "\n\nTotal time:"+GetHrStr(total_time)+"\tDistance:"+str(total_dist)+" miles\n\n"+humanOutputStr
    machineOutputStr = str(total_dist)+" " + str(total_time)+" "+machineOutputStr
    print humanOutputStr
    #print(total_dist)
    #print(total_time)
    print machineOutputStr


def GetHeuristicValue(node1,node2,routing_option):
    if routing_option == "segments":
    	return GetLowestNumOfEdges(node1,node2)
    elif routing_option == "distance":
    	return GetStDis(node1,node2)
    elif routing_option == "time":
    	return GetLowestAmtOfTime(node1,node2)
    return 0 


def AStar(cities, start, end, routing_option):
    q = Queue.PriorityQueue()
    parent = {}
    startCity = cities[start]
    endCity = cities[end]
    h = GetHeuristicValue(startCity, endCity, routing_option)
    g = 0
    f = g + h
    q.put((f,g,startCity))
    while q.not_empty:
        f,prev_g,current = q.get()
        #print "Expanding:", current.name
        if current.name == end:
            PrintPath(parent,startCity,endCity)
            return
        current.expanded = True
        for road in current.roads:
            neighbor = cities[road.city2]
            if neighbor.expanded == True:
                continue
 	    if routing_option == "distance":
	        g = prev_g + road.length
 	    elif routing_option == "segments":
		g = prev_g + 1
	    else: 
		g = prev_g+ (road.length/road.speed)
            h = GetHeuristicValue(current, endCity,routing_option)
            f = g + h
            #print (neighbor.name,"g:",g,"h:",h,"f:",f)
            q.put((f, g, neighbor))
            parent[neighbor.name] = (current,road)


# Page 82: Artificial Intelligence: A Modern Approach (3rd Edition)
def BFS (cities, startCityString, endCityString):
    parent = {}
    queueList = [ (startCityString, [startCityString]) ] # string, string-list
    while queueList:
        # Get city1 as "cityName" key
        (cityName, path) = queueList.pop(0) # FIFO
        cities[cityName].expanded = True

        # Get all city2 which are connected to city1: "cities[cityName]" object
        connectedRoadsList = cities[cityName].roads
        city2List = []
        for city2Obj in connectedRoadsList:
            if city2Obj.city2 not in city2List:
                city2List = city2List + [city2Obj.city2]

        nodes = set(city2List) # get all connected nodes
        for nextCity in nodes:
            # ignore if already visited
            if cities[nextCity].expanded == True:
                continue

            for aRoads in cities[cityName].roads:
                if (aRoads.city2 == nextCity):
                    parent[nextCity] = [cities[cityName], aRoads] # neighbor of current city has current city as the value

            # not a goal node?
            if (nextCity != endCityString):
                newEntry = ( nextCity, path + [nextCity] )
                queueList.append(newEntry)
            else:
                # goal node?
                p = path + [nextCity]
                PrintPath(parent,cities[startCityString],cities[endCityString])
                return
                #print (p)
                #return p
    #return

# Replace queue with stack in BFS algorithm: Artificial Intelligence: A Modern Approach (3rd Edition)
def DFS (cities, startCityString, endCityString):
    parent = {}
    stackList = [ (startCityString, [startCityString]) ] # string, string-list
    while stackList:
        # Get city1 as "cityName" key
        (cityName, path) = stackList.pop() # LIFO
        cities[cityName].expanded = True

        # Get all city2 which are connected to city1: "cities[cityName]" object
        connectedRoadsList = cities[cityName].roads
        city2List = []
        for city2Obj in connectedRoadsList:
            city2List = city2List + [city2Obj.city2]

        nodes = set(city2List) # get all connected nodes
        for nextCity in nodes:
            # ignore if already visited
            if cities[nextCity].expanded == True:
                continue

            for aRoads in cities[cityName].roads:
                if (aRoads.city2 == nextCity):
                    parent[nextCity] = [cities[cityName], aRoads] # neighbor of current city has current city as the value

            # not a goal node?
            if (nextCity != endCityString):
                newEntry = ( nextCity, path + [nextCity] )
                stackList.append(newEntry)
            else:
                # goal node?
                p = path + [nextCity]
                PrintPath(parent,cities[startCityString],cities[endCityString])
                return
                #print (p)
                #return p
    #return


def main():
    # dictionary { CityNameKey : CityClassObject }
    cities = {}

    # get commandline args
    start_city = sys.argv[1]
    end_city = sys.argv[2]
    routing_option = sys.argv[3]
    routing_algorithm = sys.argv[4]

    parse_cities("city-gps.txt", cities)
    missingCities = parse_roads("road-segments.txt", cities)
    AddMissingCities(cities,missingCities)
    AddMissingRoadInfo(cities)

    # condition based algo run, followed output stirng format
    if routing_algorithm == "astar":
        AStar(cities, start_city, end_city, routing_option)
    elif routing_algorithm == "dfs":
        DFS (cities, start_city, end_city)
    elif routing_algorithm == "bfs":
        BFS (cities, start_city, end_city)

main()



