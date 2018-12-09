from heapq import heappush, heappop
from collections import namedtuple

MOVES = 'UDLR'
MAX_ALLOWED_MOVES = 100

# This class represents a game board size N by N
# it is initialized using a list of  
class board:
    def __init__(self, N, init_board_config):
        """
        :param N: the size of the boards (number of rows. Total squares are N*N)
        :param init_board_config: a list of integers of length N*N. must be a permutations of numbers from 0 to N*N-1
        """
        self.N = N
        # check validity of init_board_config:
        if not len(init_board_config) == N*N:
            raise ValueError("expecting {} entries in init_board_config, found {}".format(N*N, len(init_board_config)))
        if not set(init_board_config) == set(range(N*N)):
            raise ValueError("expecting integers from 0 to {}, found {}".format(N*N, set(init_board_config)-set(range(N*N))))
        b = []
        loc = 0
        for i in range(N):
            b.append([])
            for j in range(N):
                b[i].append(init_board_config[loc])
                if init_board_config[loc]==0: # represnts the blank spot
                    self.blank=(i,j)
                loc += 1
        self.board = b

    def is_valid_move(self, mv):
        "check if move mv is valid by its location and direction"
        blank = self.blank
        N = self.N
        if mv not in set(MOVES):
            return False
        if blank[0] == N-1 and mv == 'U' or \
            blank[0] == 0 and mv == 'D' or \
            blank[1] == N-1 and mv == 'L' or \
            blank[1] == 0 and mv == 'R':
            return False
        return True
    
    def make_move(self, mv):
        "modifies the board according to the move mv"
        blank = self.blank
        if not self.is_valid_move(mv):
            print("Non valid move!")
            return
        if mv == 'U':
            move_from = blank[0]+1, blank[1]
        elif mv == 'D':
            move_from = blank[0]-1, blank[1]
        elif mv == 'R':
            move_from = blank[0], blank[1] - 1
        else:
            move_from = blank[0], blank[1] + 1

        self.board[blank[0]][blank[1]] = self.board[move_from[0]][move_from[1]]
        self.board[move_from[0]][move_from[1]] = 0
        self.blank = move_from
    
    def __repr__(self):
        "Nice output format for board object"
        ret = ''
        for row in self.board:
            ret += " ".join([str(x) if x > 0 else "-" for x in row]) + "\n"
        return ret

    
    def copy(self):
        "create a new board with same configuration"
        current_config = sum(self.board,[])
        return board(self.N, current_config)


    def correct_position(self,i):
        """
        calculate the correct position [i,j] based on the piece's ordinal
        """
        N = self.N
        if i == 0:
            return N-1, N-1
        return int((i-1)/N), (i-1) % N 


    def manhattan(self):
        """
        calculate Manhattan distance from current board configuration to ordered configuration
        """
        N = self.N
        ret = 0
        for i in range(N):
            for j in range(N):
                e = self.board[i][j]
                cp = self.correct_position(e)
                dist = abs(cp[0]-i) + abs(cp[1]-j) if e > 0 else 0
                ret += dist
        return ret

    def is_solved(self):
        "check if current configuration is the solution state"
        return self.manhattan() == 0


def IDS_solver(brd):
    """
    IDS algorithm highlights: perform DFS but stop at a depth, then goes again with depth increased.
    """
    # initialize the depth limit
    levels = 1 
    found_solution = False
    while not found_solution and levels < MAX_ALLOWED_MOVES:
        found_solution, board_moves, move_count = dfs_search(brd, levels, '', 0)
        levels += 1
        
    return found_solution, board_moves, move_count

def dfs_search(brd, limit, board_moves, move_count):
    """
    recursive DFS search
    """
    if len(board_moves) == limit:
        return False, board_moves, move_count 
    valid_moves = [mv for mv in MOVES if brd.is_valid_move(mv)]
    for mv in valid_moves:
        new_brd =  brd.copy()
        new_brd.make_move(mv)
        move_count += 1
        if new_brd.is_solved():
            return True, board_moves + mv, move_count
        found_solution, moves, move_count = dfs_search(new_brd, limit, board_moves + mv, move_count)
        if found_solution:
            return found_solution, moves, move_count
    return False, board_moves, move_count

def BFS_solver(brd):
    """
    BFS algorithm highlights: process the nodes in FIFO order
    """
    open_list = [mv for mv in MOVES if brd.is_valid_move(mv)]
    found_solution = False
    move_count = 0
    while len(open_list) and not found_solution:
        curr_move = open_list.pop(0) #extracts first element
        move_count += 1
        new_brd = brd.copy()
        for mv in curr_move: 
            new_brd.make_move(mv)
        if new_brd.is_solved():
            found_solution = True
        new_moves = [curr_move + mv for mv in MOVES if new_brd.is_valid_move(mv)]
        open_list += new_moves
    return found_solution, curr_move, move_count

def A_STAR_solver(brd):
    """
    A* algorithm highlights:
    1. processes the lowest- cost nodes first
    2. cost is based on actual price so far and heuristic
    """
    #queue_ele is a tuple that is sorted by the distance and then entry number, to retain insertion order
    queue_ele = namedtuple('queue_ele',['f', 'entry_number', 'moves_so_far'])
    found_solution = False
    pop_count = 0
    open_list = []
    entry_count = 0
    # initialize the heap with first set of moves
    for mv in MOVES:
        if not brd.is_valid_move(mv):
            continue
        new_brd = brd.copy()
        new_brd.make_move(mv)
        h = new_brd.manhattan()
        qe = queue_ele(h,entry_count,mv)
        heappush(open_list, qe)
        entry_count += 1

    while not found_solution and len(qe.moves_so_far) < MAX_ALLOWED_MOVES:
        qe = heappop(open_list)
        pop_count += 1
        curr_brd = brd.copy()
        for mv in qe.moves_so_far:
            curr_brd.make_move(mv)
        if curr_brd.is_solved():
            found_solution = True
        else:
            next_moves = [mv for mv in MOVES if curr_brd.is_valid_move(mv)]
            for mv in next_moves:
                new_brd = curr_brd.copy()
                new_brd.make_move(mv)
                h = new_brd.manhattan() # this calculates the heuristic (Manhattan distance)
                heappush(open_list,queue_ele(len(qe.moves_so_far) + h, entry_count, qe.moves_so_far + mv))
                entry_count +=1 

    return found_solution, qe.moves_so_far, pop_count

def parse_input_file(inp):
    algo_code = int(inp.readline().strip())
    board_size = int(inp.readline().strip())
    configuration = list(map(int,inp.readline().split("-")))
    return algo_code, board_size, configuration


if __name__ == "__main__":
    with open('input.txt','rt') as inp:
        algo_code, board_size, init_board_config = parse_input_file(inp)
    brd = board(board_size, init_board_config)
    
    if algo_code == 1:
        algo = IDS_solver
    elif algo_code == 2:
        algo = BFS_solver
    elif algo_code == 3:
        algo = A_STAR_solver
    else:
        raise ValueError("cannot parse algo code: {}".format(algo_code))
    
    found_solution, moves, move_count = algo(brd)
    output_line = "{} {} {}".format(moves, move_count+1, 0 if algo_code==2 else len(moves) )
    
    with open('output.txt','wt') as out:
        out.write(output_line)
    


def test():
    #b1 = board(3, map(int,"4,5,7,1,2,0,9,3,8,6".split(",")))
    #b1 = board(2, map(int,"0,3,2,1".split(",")))
    b1 = board(3, range(1,9)+[0])
    print(b1.is_solved())
    for mv in 'RRDLUR':
        b1.make_move(mv)
    found, moves, move_count = A_STAR_solver(b1)
    print(found)
    print(moves)
    print(move_count)
    print(str(b1))
    for mv in moves:
        b1.make_move(mv)
        print(str(b1))
