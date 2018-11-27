import math
MOVES = 'UDLR'
class board:
    def __init__(self, N, init_board_config):
        self.N = N
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
        ret = ''
        for row in self.board:
            ret += " ".join([str(x) if x > 0 else "-" for x in row]) + "\n"
        return ret

    
    def copy(self):
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
        return self.manhattan() == 0


def IDS_solver(brd):
    """
    IDS algorithm highlights: perform DFS but stop at a depth
    """
    levels = 1
    found_solution = False
    while not found_solution and levels < 100:
        found_solution, moves = dfs_search(brd, levels, '')
        levels += 1
        print(levels)
    return found_solution, moves

def dfs_search(brd, limit, moves_so_far):
    if len(moves_so_far) == limit:
        return False, moves_so_far
    valid_moves = [mv for mv in MOVES if brd.is_valid_move(mv)]
    for mv in valid_moves:
        new_brd =  brd.copy()
        new_brd.make_move(mv)
        if new_brd.is_solved():
            return True, moves_so_far + mv
        found_solution, moves = dfs_search(new_brd, limit, moves_so_far + mv)
        if found_solution:
            return found_solution, moves
    return False, moves_so_far

if __name__ == "__main__":
    #b1 = board(3, map(int,"4,5,7,1,2,0,9,3,8,6".split(",")))
    #b1 = board(2, map(int,"0,3,2,1".split(",")))
    b1 = board(3, range(1,9)+[0])
    print(b1.is_solved())
    for mv in 'RRDLUR':
        b1.make_move(mv)
    found, moves =IDS_solver(b1)
    print(found)
    print(moves)
    print(str(b1))
    for mv in moves:
        b1.make_move(mv)
        print(str(b1))