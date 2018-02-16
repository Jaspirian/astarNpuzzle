import copy
import heapq
import time
from collections import defaultdict

## Created by Jasper Raynolds, 2/9/18 ##

# Takes a 2D list of integers, from 0 to n, in equal-length rows, and returns a list of single-letter string moves (e.g. "U", "R")
def solve(cellArr):
    # Error handling
    if type(cellArr) is not list or type(cellArr[0]) is not list:
        print("please pass a 2D integer list to this function.")
        return None
    rowLength = len(cellArr[0])
    for row in cellArr:
        if len(row) != rowLength:
            print("all rows in the 2D integer list must be of equal length.")
            return None

    # from the cell list provided, create a Board.
    board = Board(cellArr, None)
    # is this board solvable?
    if not board.isSolvable():
        return None
    # find the solution board for these dimensions.
    goal = createSolvedBoard(board.width, board.height)
    # run A* search algorithm.
    return aStar(board, goal)
    

# Defines a Point object with an X coordinate and a Y coordinate.
class Point:

    # Takes an X and Y value.
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    # "Equals" function that compares X and Y values.
    def __eq__(self, other):
        if other == None:
            return False
        return self.x == other.x and self.y == other.y
    
    # Returns the orthogonal distance between two points, also known as the Manhattan Distance.
    def distBetween(self, point):
        return abs(point.x - self.x) + abs(point.y - self.y)

# Defines a Board object with a dictionary of int-Point "Tile" pairs.
class Board:
    
    # Takes a 2D array of integers, including "0", or a dictionary of tiles with int keys and Point values
    def __init__(self, cellArr, tiles):
        self.tiles = tiles
        if tiles == None:
            self.tiles = self.__arrToTiles__(cellArr)  
            
        bottomRightPoint = self.__getBottomCorner__()

        self.height = bottomRightPoint.y + 1
        self.width = bottomRightPoint.x + 1

    # Returns the Point of the bottom right corner of the board.
    def __getBottomCorner__(self):
        bottomCorner = None
        for key in self.tiles:
            if bottomCorner == None:
                bottomCorner = self.tiles[key]
            elif self.tiles[key].x > bottomCorner.x or self.tiles[key].y > bottomCorner.y:
                bottomCorner = self.tiles[key]
        
        return bottomCorner

    # Less-than function, comparing the manhattan distance between boards and the goal state.
    def __lt__(self, other):
        if other == None:
            return -1
        goalBoard = createSolvedBoard(self.width, self.height)
        return self.manhattanDist(goalBoard) < other.manhattanDist(goalBoard)

    # Equals function, comparing the locations of all tiles.
    def __eq__(self, other):
        if other == None:
            return False
        for tile in self.tiles:
            if self.tiles[tile] != other.tiles[tile]:
                return False
        return True

    # Hash
    def __hash__(self):
        # Flatten board, adding separators for lines and tiles
        flat = []
        for row in range(self.height):
            for col in range(self.width):
                for key in self.tiles:
                    if self.tiles[key].x == col and self.tiles[key].y == row:
                        flat.append(key)
                        flat.append("/")
            flat.append("|")
                        
        string = "".join(str(x) for x in flat)
        return hash(string)

    # Returns the board in an easy-to-read fashion
    def toString(self):
        # Find the highest number of digits--then add 1.
        mostDigits = len(str(len(self.tiles)-1)) + 1
        # Define separator:
        separator = "-----\n"

        output = ""
        output += separator
        # Add the board state to "output"
        for row in range(self.height):
            for col in range(self.width):
                for key in self.tiles:
                    if self.tiles[key].x == col and self.tiles[key].y == row:
                        output += str(key)
                        # Monospacing
                        for space in range(mostDigits - len(str(key))):
                            output += " "
            output += "\n"
        output += separator
        return output

    # Returns a dictionary of number-point pairs when given a 2d array of integers.
    def __arrToTiles__(self, array):
        tiles = {}

        for row in range(len(array)):
            for col in range(len(array[0])):
                integer = array[row][col]
                tiles[integer] = Point(col, row)

        return tiles

    # Returns true if the board is solvable, false if not.
    def isSolvable(self):
        # Flatten list, remove 0
        flat = []
        row0 = 0
        for row in range(self.height):
            for col in range(self.width):
                for key in self.tiles:
                    if self.tiles[key].x == col and self.tiles[key].y == row:
                        if key == 0:
                            row0 = row
                        else:
                            flat.append(key)
            
        # Count inversions
        count = 0
        for tile in flat:
            for tile2 in flat:
                if flat.index(tile2) < flat.index(tile) and tile2 > tile:
                    count += 1

        if self.height % 2 == 0:
            if count % 2 == 0:
                return True
        elif (count + (self.height - row0 - 1)) % 2 == 0:
            return True
        return False

    # Returns the total number of moves to arrive at a destination board state, assuming tiles may move through each other.
    def manhattanDist(self, destBoard):
        distance = 0        
        
        for tile in self.tiles:
            distance += destBoard.tiles[tile].distBetween(self.tiles[tile])
    
        return distance
    
    # Returns the number of misplaced tiles added to the number of moves made to reach this state.
    def hammingDist(self, movesSoFar, destBoard):
        distance = movesSoFar
        
        for tile in self.tiles:
            if self.tiles[tile] != destBoard.tiles[tile]: 
                distance += 1
        
        return distance
    
    # Interprets another board as a move "U", "D", "L" or "R" from this one.
    def translateEmptyMove(self, pastBoard):
        # Compare both empty spaces
        thisEmpty = self.tiles[0]
        thatEmpty = pastBoard.tiles[0]
        move = Point(thisEmpty.x - thatEmpty.x, thisEmpty.y - thatEmpty.y)

        # ugly case switching
        if move.x == 0 and move.y == -1:
            return "D"
        elif move.x == 0 and move.y == 1:
            return "U"
        elif move.x == -1 and move.y == 0:
            return "R"
        elif move.x == 1 and move.y == 0:
            return "L"
        return None
    
    # Returns a set of states that are one move away, not including moves that travel off the board or match the previous state.
    def getNeighbors(self, previousBoard):
        neighBoards = []
        
        emptyLoc = self.tiles[0]
        # Grab location of previous board's empty space
        if previousBoard != None:
            previousEmpty = previousBoard.tiles[0]
        # Create a set of possible moves for the empty space to take
        orthogonal = [Point(-1, 0), Point(0, 1), Point(0, -1), Point(1, 0)]
        # For each of those moves:
        for move in orthogonal:
            newTiles = copy.deepcopy(self.tiles)
            newEmptyLoc = Point(emptyLoc.x + move.x, emptyLoc.y + move.y)
            newTiles[0] = newEmptyLoc
            # Discard this move if we exit the board
            if newEmptyLoc.x < 0 or newEmptyLoc.x > self.width - 1 or newEmptyLoc.y < 0 or newEmptyLoc.y > self.height - 1:
                continue
            # Discard this move if it matches the previous board
            if previousBoard != None:
                if newEmptyLoc.x == previousEmpty.x and newEmptyLoc.y == previousEmpty.y:
                    continue
            # If we got this far, set the tile in the new empty location to the previous empty location
            for key in newTiles:
                if key != 0 and newTiles[key].x == newEmptyLoc.x and newTiles[key].y == newEmptyLoc.y:
                    newTiles[key] = Point(emptyLoc.x, emptyLoc.y)
                    break
            neighBoard = Board(None, newTiles)
            
            # Add the board to the array
            neighBoards.append(neighBoard)

        return neighBoards

# Searches through available moves following the A* algorithm and returns a string 
# Available heuristics are below this function: heuristic_manhattan, heuristic_hamming, heuristic_n_maxSwap
# The heuristic used can be changed by editing the three heuristic method calls within this function.
def aStar(start, goal):
    
    startingTime = int(round(time.time() * 1000))
    # The set of states already evaluated
    closedSet = set()

    # For each node, which node it can most efficiently be reached from.
    # If a node can be reached from many nodes, cameFrom will eventually contain the
    # most efficient previous step.
    cameFrom = {}
    
    # For each node, the cost of getting from the start node to that node.
    gScore = defaultdict(lambda: float("inf"))

    # The cost of going from start to start is zero.
    gScore[start] = 0
    
    # For each node, the total cost of getting from the start node to the goal
    # by passing by that node. That value is partly known, partly heuristic.
    fScore = defaultdict(lambda: float("inf"))

    # For the first state, that value is completely heuristic.
    fScore[start] = heuristic_manhattan(start, goal)
    
    # The heap of currently discovered states that are not evaluated yet.
    # Initially, only the start state is known.
    openHeap = []
    heapq.heappush(openHeap, (fScore[start], start))

    # While there are moves in the open heap:
    while openHeap:
        # Pop the lowest F-score move from the open heap.
        current = heapq.heappop(openHeap)[1]
        
        # Check if we're at the goal:
        if current == goal:
            # Get the states we came through to reach this goal
            path = reconstruct_path(current, [], cameFrom)
            path.reverse()
            path.append(goal)
            # Convert these states into single-letter moves.
            moveset = boardlistToMoves(path)
            # inform the user how long the search took.
            print("total solution took",int(round(time.time() * 1000)) - startingTime,"ms.")
            return moveset

        # Add this state to the set of closed states.
        closedSet.add(current)
        
        # Get the neighboring states.
        neighbors = current.getNeighbors(cameFrom.get(current))
        for neighbor in neighbors:
            neighborLoop = int(round(time.time() * 1000))
            if neighbor in closedSet:
                continue        # Ignore the neighbor which is already evaluated.
            
            # The distance from start to a neighbor
            tentative_gScore = gScore[current] + heuristic_manhattan(current, goal)
            if tentative_gScore >= gScore[neighbor]:
                continue        # This is not a better path.

            # This path is the best until now. Record it!
            cameFrom[neighbor] = current
            gScore[neighbor] = tentative_gScore
            fScore[neighbor] = gScore[neighbor] + heuristic_manhattan(neighbor, goal)
            
            # Discover a new state
            if neighbor not in [k for v, k in openHeap]:
                heapq.heappush(openHeap, (fScore[neighbor], neighbor))

    return None

# A Manhattan distance heuristic: orthogonal movement of points
def heuristic_manhattan(board1, board2):
    distance = 0
    
    # for each tile number, increment "distance" by the orthogonal distance between the two tile locations
    for tile in board1.tiles:
        distance += board1.tiles[tile].distBetween(board2.tiles[tile])
        
    return distance

# A Hamming distance heuristic: number of tiles out of place plus the number of moves to reach this board
def heuristic_hamming(board, goal, movesSoFar):
    distance = movesSoFar
    
    # for each tile number, increment "distance" by one if it's out of place.
    for tile in board.tiles:
        if board.tiles[tile] != goal.tiles[tile]:
            distance += 1
    
    return distance

# An N-MaxSwap distance heuristic: the number of steps it would take to solve the puzzle if all tiles could be swapped with the empty space.
def heuristic_n_maxSwap(board, goal):
    distance = 0
    
    newTiles = copy.deepcopy(board.tiles)
    for tile in newTiles:
        if tile != 0 and goal.tiles[tile] == newTiles[0]:
            #swap
            newTiles[0] = newTiles[tile]
            newTiles[tile] = goal.tiles[tile]
            distance += 1
    
    return distance

# Reconstructs the path, from the goal state to the start
def reconstruct_path(current, total_path, cameFrom):
    if(cameFrom.get(current) != None):
        total_path.append(cameFrom[current])
        return reconstruct_path(cameFrom[current], total_path, cameFrom)
    else :
        return total_path

# Returns a list of strings denoting moves ("U", "R", etc), from a list of Boards.
def boardlistToMoves(list):
    moveset = []
    for index in range(len(list)):
        if index > 0:
            moveset.append(list[index].translateEmptyMove(list[index-1]))
    return moveset

# Returns a solved board, given a width and height.
def createSolvedBoard(width, height):
    cellArr = []

    for row in range(height):
        boardRow = []
        for col in range(width):
            boardRow.append((row * width) + col + 1)
        cellArr.append(boardRow)

    cellArr[height - 1][width - 1] = 0

    solvedBoard = Board(cellArr, None)
    
    return solvedBoard
