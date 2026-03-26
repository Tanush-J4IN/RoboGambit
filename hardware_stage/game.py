"""add your game.py code here"""
"""
RoboGambit 2025-26 — Task 1: Autonomous Game Engine
Organised by Aries and Robotics Club, IIT Delhi

Board: 6x6 NumPy array
  - 0  : Empty cell
  - 1  : White Pawn
  - 2  : White Knight
  - 3  : White Bishop
  - 4  : White Queen
  - 5  : White King
  - 6  : Black Pawn
  - 7  : Black Knight
  - 8  : Black Bishop
  - 9  : Black Queen
  - 10 : Black King

Board coordinates:
  - Bottom-left  = A1  (index [0][0])
  - Columns   = A–F (left to right)
  - Rows      = 6-1 (top to bottom)(from white's perspective)

Move output format:  "<piece_id>:<source_cell>-><target_cell>"
  e.g.  "1:B3->B4"   (White Pawn moves from B3 to B4)
"""

import numpy as np
import time
import random
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


remaining_time = 600 #Change to 900 for final matches
start_time_of_search = 0
time_limit = 0

EMPTY = 0

# Piece IDs
WHITE_PAWN   = 1
WHITE_KNIGHT = 2
WHITE_BISHOP = 3
WHITE_QUEEN  = 4
WHITE_KING   = 5
BLACK_PAWN   = 6
BLACK_KNIGHT = 7
BLACK_BISHOP = 8
BLACK_QUEEN  = 9
BLACK_KING   = 10

WHITE_PIECES = {WHITE_PAWN, WHITE_KNIGHT, WHITE_BISHOP, WHITE_QUEEN, WHITE_KING}
BLACK_PIECES = {BLACK_PAWN, BLACK_KNIGHT, BLACK_BISHOP, BLACK_QUEEN, BLACK_KING}

BOARD_SIZE = 6

PIECE_VALUES = {
    WHITE_PAWN:   100,
    WHITE_KNIGHT: 320,
    WHITE_BISHOP: 310,
    WHITE_QUEEN:  950,
    WHITE_KING:  20000,
    BLACK_PAWN:  -100,
    BLACK_KNIGHT:-320,
    BLACK_BISHOP:-310,
    BLACK_QUEEN: -950,
    BLACK_KING: -20000,
}

# Column index → letter
COL_TO_FILE = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'}
FILE_TO_COL = {v: k for k, v in COL_TO_FILE.items()}

PIECE_LIMITS = {
    WHITE_PAWN: 6, WHITE_KNIGHT: 2, WHITE_BISHOP: 2, WHITE_QUEEN: 1, WHITE_KING: 1,
    BLACK_PAWN: 6, BLACK_KNIGHT: 2, BLACK_BISHOP: 2, BLACK_QUEEN: 1, BLACK_KING: 1
}

TT_EXACT = 0
TT_ALPHA = 1 # Upper bound (fail low)
TT_BETA  = 2 # Lower bound (fail high)
np.random.seed(42) 
ZOBRIST_TABLE = np.random.randint(1, 2**63 - 1, size=(11, 6, 6), dtype=np.uint64)
ZOBRIST_TURN  = np.uint64(random.randint(1, 2**63 - 1))

transposition_table = {}

def get_initial_hash(board: np.ndarray, playing_white: bool) -> np.uint64:
    """Calculates the starting hash of the board."""
    h = np.uint64(0)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            piece = board[r][c]
            if piece != EMPTY:
                h ^= ZOBRIST_TABLE[piece][r][c]
    if not playing_white:
        h ^= ZOBRIST_TURN
    return h

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------
    
def idx_to_cell(row: int, col: int) -> str:
    return f"{COL_TO_FILE[col]}{row + 1}"
       
def cell_to_idx(cell: str):
    col = FILE_TO_COL[cell[0].upper()]
    row = int(cell[1]) - 1
    return row, col
    
def in_bounds(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE
    
def is_white(piece: int) -> bool:
    return piece in WHITE_PIECES
    
def is_black(piece: int) -> bool:
    return piece in BLACK_PIECES
    
def same_side(p1: int, p2: int) -> bool:
    return (is_white(p1) and is_white(p2)) or (is_black(p1) and is_black(p2))

# ---------------------------------------------------------------------------
# Move generation
# ---------------------------------------------------------------------------

def get_pawn_moves(board: np.ndarray, row: int, col: int, piece: int, offboard:dict): 
    moves = []
    if is_white(piece):
        promo_pieces = [WHITE_QUEEN, WHITE_BISHOP, WHITE_KNIGHT]
        if in_bounds(row+1, col) and board[row+1][col] == EMPTY:
            if row+1 == 5:
                for new_piece in promo_pieces:
                    if offboard[new_piece] > 0:
                        moves.append((piece, row, col, row+1, col,new_piece))
            else:
                moves.append((piece, row, col, row+1, col, None))

        for dc in [-1, 1]:
            if in_bounds(row+1, col+dc) and is_black(board[row+1][col+dc]):
                if row+1 == 5:
                    for new_piece in promo_pieces:
                        if offboard[new_piece] > 0:
                            moves.append((piece, row, col, row+1, col+dc,new_piece))
                else:
                    moves.append((piece, row, col, row+1, col+dc, None))
    else:
        promo_pieces = [BLACK_QUEEN, BLACK_BISHOP, BLACK_KNIGHT]
        if in_bounds(row-1, col) and board[row-1][col] == EMPTY:
            if row-1 == 0:
                for new_piece in promo_pieces:
                    if offboard[new_piece] > 0:
                        moves.append((piece, row, col, row-1, col, new_piece))
            else:
                moves.append((piece, row, col, row-1, col, None))
        for dc in [-1, 1]:
            if in_bounds(row-1, col+dc) and is_white(board[row-1][col+dc]):
                if row-1 == 0:
                    for new_piece in promo_pieces:
                        if offboard[new_piece] > 0:
                            moves.append((piece, row, col, row-1, col+dc, new_piece))
                else:
                    moves.append((piece, row, col, row-1, col+dc,None))
    return moves
    
def get_knight_moves(board: np.ndarray, row: int, col: int, piece: int, offboard):
    moves = []
    knight_jumps = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
    for (r,c) in knight_jumps:
        new_row, new_col = row + r, col + c
        if not in_bounds(new_row, new_col): continue
        target = board[new_row][new_col]
        if target == EMPTY or (is_white(piece) and is_black(target)) or (is_black(piece) and is_white(target)):
            moves.append((piece,row,col,new_row,new_col,None))
    return moves
    
def get_sliding_moves(board: np.ndarray, row: int, col: int, piece: int, directions, offboard):
    moves = []
    for (dr,dc) in directions:
        r, c = row + dr, col + dc
        while in_bounds(r, c):
            target = board[r][c]
            if target == EMPTY:
                moves.append((piece,row,col,r,c,None))
            elif (is_white(piece) and is_black(target)) or (is_black(piece) and is_white(target)):
                moves.append((piece,row,col,r,c,None))
                break 
            else: break
            r += dr; c += dc
    return moves

def get_bishop_moves(board: np.ndarray, row: int, col: int, piece: int, offboard):
    diagonals = [(-1,-1),(-1,1),(1,-1),(1,1)]
    return get_sliding_moves(board, row, col, piece, diagonals,offboard)

def get_queen_moves(board: np.ndarray, row: int, col: int, piece: int, offboard):
    all_dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    return get_sliding_moves(board, row, col, piece, all_dirs,offboard)
    
def get_king_moves(board: np.ndarray, row: int, col: int, piece: int, offboard):
    moves = []
    k = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)]
    for (dr,dc) in k:
        r,c = row+dr,col+dc
        if in_bounds(r,c):
            target = board[r][c]
            if (target == EMPTY) or ((is_white(piece) and is_black(target)) or (is_black(piece) and is_white(target))):
                moves.append((piece,row,col,r,c,None))
    return moves
    
MOVE_GENERATORS = {
    WHITE_PAWN:   get_pawn_moves, WHITE_KNIGHT: get_knight_moves,
    WHITE_BISHOP: get_bishop_moves, WHITE_QUEEN:  get_queen_moves,
    WHITE_KING:   get_king_moves, BLACK_PAWN:   get_pawn_moves,
    BLACK_KNIGHT: get_knight_moves, BLACK_BISHOP: get_bishop_moves,
    BLACK_QUEEN:  get_queen_moves, BLACK_KING:   get_king_moves,
}

def king_under_attack(bd, kr, kc, is_king_white):
    enemy_pawn = BLACK_PAWN if is_king_white else WHITE_PAWN
    enemy_knight = BLACK_KNIGHT if is_king_white else WHITE_KNIGHT
    enemy_bishop = BLACK_BISHOP if is_king_white else WHITE_BISHOP
    enemy_queen = BLACK_QUEEN if is_king_white else WHITE_QUEEN
    enemy_king = BLACK_KING if is_king_white else WHITE_KING

    pawn_dir = 1 if is_king_white else -1
    for dc in [-1, 1]:
        pr, pc = kr + pawn_dir, kc + dc
        if 0 <= pr < 6 and 0 <= pc < 6:
            if bd[pr, pc] == enemy_pawn: return True
    knight_jumps = [(-2,-1), (-2,1), (-1,-2), (-1,2), (1,-2), (1,2), (2,-1), (2,1)]
    for dr, dc in knight_jumps:
        tr, tc = kr + dr, kc + dc
        if 0 <= tr < 6 and 0 <= tc < 6:
            if bd[tr, tc] == enemy_knight: return True
    directions = {"diag": [(-1,-1), (-1,1), (1,-1), (1,1)], "ortho": [(-1,0), (1,0), (0,-1), (0,1)]}
    for dr, dc in directions["diag"]:
        for i in range(1, 6):
            tr, tc = kr + dr * i, kc + dc * i
            if not (0 <= tr < 6 and 0 <= tc < 6): break
            p = bd[tr, tc]
            if p != EMPTY:
                if p in {enemy_bishop, enemy_queen}: return True
                break
    for dr, dc in directions["ortho"]:
        for i in range(1, 6):
            tr, tc = kr + dr * i, kc + dc * i
            if not (0 <= tr < 6 and 0 <= tc < 6): break
            p = bd[tr, tc]
            if p != EMPTY:
                if p in {enemy_queen}: return True
                break
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0: continue
            tr, tc = kr + dr, kc + dc
            if 0 <= tr < 6 and 0 <= tc < 6:
                if bd[tr, tc] == enemy_king: return True
    return False

def get_all_moves(board: np.ndarray, playing_white: bool, offboard,king,captures_only=False):
    moves = []
    king_piece = WHITE_KING if playing_white else BLACK_KING
    king_row, king_col = king[king_piece]
    in_check = king_under_attack(board, king_row, king_col, playing_white)
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            piece = board[i][j]
            if piece == EMPTY or is_white(piece) != playing_white: continue
            piece_moves = MOVE_GENERATORS[piece](board, i, j, piece,offboard)
            for move_candidate in piece_moves:  
                if len(move_candidate) == 5:
                    move = (*move_candidate, None)
                else:
                    move = move_candidate
                p, r1, c1, r2, c2, promo = move
                if captures_only and board[r2, c2] == EMPTY: continue
                if board[r2][c2] in {WHITE_KING, BLACK_KING}: continue
                captured = board[r2][c2]
                board[r1][c1] = EMPTY
                board[r2][c2] = promo if promo is not None else p
                kr, kc = (r2, c2) if p == king_piece else (king_row, king_col)
                king_safe = not king_under_attack(board, kr, kc, playing_white)
                board[r1][c1] = p
                board[r2][c2] = captured
                if in_check and not king_safe: continue
                if not in_check and not king_safe: continue
                moves.append(move)
    return moves
    
# ---------------------------------------------------------------------------
# Board evaluation heuristic
# ---------------------------------------------------------------------------
    
# ---------------------------------------------------------------------------
# Board evaluation heuristic
# ---------------------------------------------------------------------------
#PST for black is simply flipped of white
PST1 = np.array([
        [ 0,  0,  0,  0,  0,  0],
        [10, 10, 10, 10, 10, 10],
        [15, 25, 35, 35, 25, 15],
        [25, 35, 45, 45, 35, 25],
        [50, 50, 50, 50, 50, 50],
        [ 0,  0,  0,  0,  0,  0]])
PST6=-1*np.flipud(PST1)

PST2 = np.array([[-50, -35, -20, -20, -35, -50,],
        [-35,   0,  10,  10,   0, -35,],
        [-20,  15,  25,  25,  15, -20,],
        [-20,  15,  25,  25,  15, -20,],
        [-20,  15,  20,  20,  15, -20,],
        [-50, -40, -30, -30, -40, -50]])
PST7=-1*np.flipud(PST2)

PST3 = np.array([[-20, -10, -10, -10, -10, -20,],
        [-10,  10,   5,   5,  10, -10,],
        [-10,  15,  20,  20,  15, -10,],
        [-10,  15,  20,  20,  15, -10,],
        [-10,  10,  10,  10,  10, -10,],
        [-20, -10, -10, -10, -10, -20]])
PST8=-1*np.flipud(PST3)

PST4=np.array([
    [-20, -10, -5, -5, -10, -20],
    [-10,   0,  5,  5,   0, -10],
    [ -5,   5, 10, 10,   5,  -5],
    [ -5,   5, 10, 10,   5,  -5],
    [-10,   0,  5,  5,   0, -10],
    [-20, -10, -5, -5, -10, -20]])
PST9=-1*np.flipud(PST4)

PST5=np.array([
    [ 20,  20,  10,  10,  20,  20], 
    [ 20,  10,   0,   0,  10,  20],
    [-10, -20, -20, -20, -20, -10], 
    [-20, -30, -30, -30, -30, -20], 
    [-30, -40, -40, -40, -40, -30], 
    [-30, -40, -40, -40, -40, -30]])
PST10=-1*np.flipud(PST5)

PST15 = np.array([
    [-20, -10, -10, -10, -10, -20],
    [-10,   0,   5,   5,   0, -10],
    [-10,   5,  15,  15,   5, -10],
    [-10,   5,  15,  15,   5, -10],
    [-10,   0,   5,   5,   0, -10],
    [-20, -10, -10, -10, -10, -20]
])
PST20 = -1*np.flipud(PST15)

PSTS = {
    WHITE_PAWN: PST1, BLACK_PAWN: PST6,
    WHITE_KNIGHT: PST2, BLACK_KNIGHT: PST7,
    WHITE_BISHOP: PST3, BLACK_BISHOP: PST8,
    WHITE_QUEEN: PST4, BLACK_QUEEN: PST9,
    WHITE_KING: PST5, BLACK_KING: PST10
}

#piece tracker to support incremental logic
def get_piece_tracker(board: np.ndarray) -> dict:
    tracker = {
        WHITE_PAWN: set(), BLACK_PAWN: set(),
        WHITE_KNIGHT: set(), BLACK_KNIGHT: set(),
        WHITE_QUEEN: set(), BLACK_QUEEN: set()
    }
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = board[r][c]
            if p in tracker:
                tracker[p].add((r, c))
    return tracker
#defined endgame when most pieces are off the board
def is_endgame(offboard):
    return offboard["Total"]>=6
#initially pawn positions depend on back rank
def dynamic_pawn_pst(board: np.ndarray, base_pst: np.ndarray, is_white: bool) -> np.ndarray:
    new_pst = base_pst.copy()
    backrank = 0 if is_white else 5
    pawn_dir = 1 if is_white else -1
    
    king_piece = WHITE_KING if is_white else BLACK_KING
    queen_piece = WHITE_QUEEN if is_white else BLACK_QUEEN
    bishop_piece = WHITE_BISHOP if is_white else BLACK_BISHOP

    for c in range(BOARD_SIZE):
        piece = board[backrank][c]
        if piece == king_piece:
            for dc in [-1, 0, 1]:
                if 0 <= c + dc < BOARD_SIZE:
                    new_pst[backrank + pawn_dir][c + dc] += 15*pawn_dir
                    new_pst[backrank + (pawn_dir*2)][c + dc] += 10*pawn_dir
        elif piece in {queen_piece, bishop_piece}:
            for i in range(1, 4):
                r = backrank + (pawn_dir * i)
                if 0 <= r < BOARD_SIZE:
                    if c - i >= 0: new_pst[r][c - i] -= 10*pawn_dir
                    if c + i < BOARD_SIZE: new_pst[r][c + i] -= 10*pawn_dir
                    
    return new_pst

#penalizes pieces for staying on pin
def get_pin_penalty(board, king_pos, is_white_king, tracker):
    penalty = 0
    r, c = king_pos
    enemy_queen = BLACK_QUEEN if is_white_king else WHITE_QUEEN
    enemy_slider = BLACK_BISHOP if is_white_king else WHITE_BISHOP
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),   # Orthogonal (Queen)
        (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonal (Queen/Bishop)
    ]
    
    for dr, dc in directions:
        found_friendly = None
        curr_r, curr_c = r + dr, c + dc
        
        while 0 <= curr_r < 6 and 0 <= curr_c < 6:
            piece = board[curr_r][curr_c]
            if piece != EMPTY:
                if same_side(piece, WHITE_KING if is_white_king else BLACK_KING):
                    if found_friendly is None:
                        found_friendly = (curr_r, curr_c)
                    else:
                        break
                else:
                    is_diag = (dr != 0 and dc != 0)
                    if piece == enemy_queen or (is_diag and piece == enemy_slider):
                        if found_friendly:
                            penalty += 40 
                        else:
                            penalty += 20
                    break 
            curr_r += dr
            curr_c += dc
            
    return penalty
#simple evaluation function with piece values and PST
def evaluate(board: np.ndarray) -> float:
    score = 0.0
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = board[row][col]
            if piece != EMPTY:
                score += PIECE_VALUES.get(piece, 0)+PSTS[piece][row][col]
    return score
#complex evaluation function only called at leaf nodes for in depth positional understanding
def evaluate_complex(board, tracker, offboard, king_tracker):
    bonus = 0
    wp = tracker[WHITE_PAWN]
    bp = tracker[BLACK_PAWN]
    if offboard[WHITE_BISHOP] <= 0: bonus += 50#Bishop Pair
    if offboard[BLACK_BISHOP] <= 0: bonus -= 50
    w_files = [0] * BOARD_SIZE
    b_files = [0] * BOARD_SIZE
    for r, c in wp: w_files[c] += 1
    for r, c in bp: b_files[c] += 1
    for r, c in wp:
        if w_files[c] > 1: bonus -= 10 # Doubled
        is_iso = True
        if c > 0 and w_files[c-1] > 0: is_iso = False
        if c < 5 and w_files[c+1] > 0: is_iso = False
        if is_iso: bonus -= 20 # Isolated
        is_passed = True
        for i in [c-1, c, c+1]:
            if 0 <= i <= 5:
                for br, bc in bp:
                    if bc == i and br > r:
                        is_passed = False
                        break
        if is_passed: bonus += 30 + (r * 15) # Passed
        if (r-1, c-1) in wp or (r-1, c+1) in wp: bonus += 15 # Chained
    for r, c in bp:
        if b_files[c] > 1: bonus += 10
        is_iso = True
        if c > 0 and b_files[c-1] > 0: is_iso = False
        if c < 5 and b_files[c+1] > 0: is_iso = False
        if is_iso: bonus += 20 
        is_passed = True
        for i in [c-1, c, c+1]:
            if 0 <= i <= 5:
                for wr, wc in wp:
                    if wc == i and wr < r:
                        is_passed = False
                        break
        if is_passed: bonus -= 30 + ((5 - r) * 15)
        if (r+1, c-1) in bp or (r+1, c+1) in bp: bonus -= 15
    for r, c in tracker[WHITE_KNIGHT]: #Outpost Knight
        if r >= 3: 
            if (r-1, c-1) in wp or (r-1, c+1) in wp: 
                can_be_attacked = False
                if c > 0 and b_files[c-1] > 0:
                    for br, bc in bp:
                        if bc == c-1 and br > r: can_be_attacked = True
                if c < 5 and b_files[c+1] > 0:
                    for br, bc in bp:
                        if bc == c+1 and br > r: can_be_attacked = True
                if not can_be_attacked: bonus += 40
    for r, c in tracker[BLACK_KNIGHT]:
        if r <= 2:
            if (r+1, c-1) in bp or (r+1, c+1) in bp:
                can_be_attacked = False
                if c > 0 and w_files[c-1] > 0:
                    for wr, wc in wp:
                        if wc == c-1 and wr < r: can_be_attacked = True
                if c < 5 and w_files[c+1] > 0:
                    for wr, wc in wp:
                        if wc == c+1 and wr < r: can_be_attacked = True
                if not can_be_attacked: bonus -= 40
    wk_r, wk_c = king_tracker[WHITE_KING]    #King safety
    bk_r, bk_c = king_tracker[BLACK_KING]
    
    if tracker[BLACK_QUEEN]:              #distance from opp QUEEN
        bq_r, bq_c = next(iter(tracker[BLACK_QUEEN]))
        if abs(wk_r - bq_r) + abs(wk_c - bq_c) < 3: bonus -= 30
    
    if tracker[WHITE_QUEEN]:
        wq_r, wq_c = next(iter(tracker[WHITE_QUEEN]))
        if abs(bk_r - wq_r) + abs(bk_c - wq_c) < 3: bonus += 30
    bonus -= get_pin_penalty(board, (wk_r,wk_c), True, tracker)    #pins
    bonus += get_pin_penalty(board, (bk_r,bk_c), False, tracker) 
    w_escape = 0           #Escape Squares for king
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr==0 and dc==0: continue
            nr, nc = wk_r+dr, wk_c+dc
            if 0<=nr<6 and 0<=nc<6 and board[nr][nc] == EMPTY: w_escape += 1
    bonus += w_escape * 3

    b_escape = 0
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr==0 and dc==0: continue
            nr, nc = bk_r+dr, bk_c+dc
            if 0<=nr<6 and 0<=nc<6 and board[nr][nc] == EMPTY: b_escape += 1
    bonus -= b_escape * 3

    return bonus
#to help engine find mate patterns
def Cornering(current_score, king_tracker,offboard):
    if abs(current_score) < 500 or not is_endgame(offboard):
        return 0     
    w_king = king_tracker[WHITE_KING]
    b_king = king_tracker[BLACK_KING]
    winning_king = w_king if current_score > 0 else b_king
    losing_king = b_king if current_score > 0 else w_king
    edge_dist = abs(2.5 - losing_king[0]) + abs(2.5 - losing_king[1])
    edge_bonus = edge_dist * 15 
    king_dist = abs(winning_king[0] - losing_king[0]) + abs(winning_king[1] - losing_king[1])
    proximity_bonus = (10 - king_dist) * 5 
    total_bonus = edge_bonus + proximity_bonus
    return total_bonus if current_score > 0 else -total_bonus
# ---------------------------------------------------------------------------
# Incremental State Management
# ---------------------------------------------------------------------------
#The apply_move is implemented in an incremental manner it makes changes to corresponding tracker, and unapply reverts those changes   
def apply_move(board, piece, r1, c1, r2, c2, promo, offboard, king_tracker, current_hash, tracker):
    captured = board[r2][c2]
    new_hash = current_hash
    new_hash ^= ZOBRIST_TABLE[piece][r1][c1]
    
    if captured != EMPTY:
        if captured in offboard:
            offboard[captured] += 1
            offboard["Total"] += 1
        new_hash ^= ZOBRIST_TABLE[captured][r2][c2]
        if captured in tracker: 
            tracker[captured].remove((r2, c2)) 
            
    if promo is not None and promo in offboard:
        offboard[promo] -= 1
        offboard["Total"] -= 1
        
    if piece == WHITE_KING or piece == BLACK_KING:
        king_tracker[piece] = (r2, c2)
        
    placed_piece = promo if promo is not None else piece
    new_hash ^= ZOBRIST_TABLE[placed_piece][r2][c2]
    new_hash ^= ZOBRIST_TURN
    
    if promo is not None:
        board[r2][c2] = promo
    else:
        board[r2][c2] = piece
    board[r1][c1] = EMPTY
    if piece in tracker: 
        tracker[piece].remove((r1, c1))
    if placed_piece in tracker: 
        tracker[placed_piece].add((r2, c2))
        
    return captured, new_hash

def unapply_move(board, piece, r1, c1, r2, c2, promo, captured, offboard, king_tracker, tracker):
    if captured != EMPTY and captured in offboard:
        offboard[captured] -= 1
        offboard["Total"] -= 1
    if promo is not None and promo in offboard:
        offboard[promo] += 1
        offboard["Total"] += 1
    if piece == WHITE_KING or piece == BLACK_KING:
        king_tracker[piece] = (r1, c1)
    placed_piece = promo if promo is not None else piece
    if placed_piece in tracker: 
        tracker[placed_piece].remove((r2, c2))
    if captured in tracker: 
        tracker[captured].add((r2, c2))
    if piece in tracker: 
        tracker[piece].add((r1, c1))
        
    board[r1][c1] = piece
    board[r2][c2] = captured
    
def format_move(piece, r1, c1, r2, c2, promo):
    src = idx_to_cell(r1,c1)
    dst = idx_to_cell(r2,c2)
    if promo is not None:
        return f"{piece}:{src}->{dst}={promo}"
    else:
        return f"{piece}:{src}->{dst}"

# ---------------------------------------------------------------------------
# Search and Move Logic
# ---------------------------------------------------------------------------
#incremental score simply modifies the score based on the move made
def get_incremental_score(board, move, current_score):
    piece, r1, c1, r2, c2, promo = move
    piece_placed = promo if promo is not None else piece
    piece_at_src = board[r1][c1]
    captured = board[r2][c2]
    new_score = current_score
    new_score -= PIECE_VALUES[piece_at_src]
    new_score -= PSTS[piece_at_src][r1][c1]
    if captured != EMPTY:
        new_score -= PIECE_VALUES[captured]
        new_score -= PSTS[captured][r2][c2]
    new_score += PIECE_VALUES[piece_placed]
    new_score += PSTS[piece_placed][r2][c2]
    return new_score
#To prevent horizon effect
def quiescence_search(board, alpha, beta, color, current_score, offboard, king,tracker):
    stand_pat = current_score * color
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat
    moves = get_all_moves(board, color == 1, offboard, king, True)
    capture_moves = score_moves(board, moves)
    for move in capture_moves:
        new_score = get_incremental_score(board, move, current_score)
        captured, _ = apply_move(board, *move, offboard, king, 0,tracker)
        score = -quiescence_search(board, -beta, -alpha, -color, new_score, offboard, king,tracker)
        unapply_move(board, *move, captured, offboard, king,tracker)
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha
#to speed up alpha beta pruning
def score_moves(board: np.ndarray, moves: list) -> list:
    scored_moves = []
    for move in moves:
        piece, src_r, src_c, dst_r, dst_c, promo = move
        score = 0        
        target_piece = board[dst_r, dst_c]
        if target_piece != EMPTY:
            victim_val = abs(PIECE_VALUES.get(target_piece, 0))
            aggressor_val = abs(PIECE_VALUES.get(piece, 0))
            score = (10 * victim_val) - aggressor_val 
        if promo is not None:
            score = 9000
        scored_moves.append((score, move))
    scored_moves.sort(key=lambda x: x[0], reverse=True) 
    return [m[1] for m in scored_moves]

class TimeoutException(Exception):
    pass
#to check time after a certain interval
nodes_visited = 0

def check_time():
    global nodes_visited
    nodes_visited += 1
    if nodes_visited & 2047 == 0:
        if time.perf_counter() - start_time_of_search > time_limit:
            raise TimeoutException()
#main search function implemented with Negamax logic
def Search(board, depth, alpha, beta, color, current_score, offboard, king, current_hash, tracker):
    check_time()
    original_alpha = alpha
    tt_entry = transposition_table.get(current_hash)
    tt_move = None
    if tt_entry is not None:
        tt_depth, tt_score, tt_flag, tt_move = tt_entry
        if tt_depth >= depth:
            if tt_flag == TT_EXACT:
                return tt_score
            if tt_flag == TT_ALPHA and tt_score <= alpha:
                return alpha
            if tt_flag == TT_BETA and tt_score >= beta:
                return beta
    moves = score_moves(board, get_all_moves(board, color == 1, offboard, king))
    if not moves:
        king_piece = WHITE_KING if color == 1 else BLACK_KING
        kr, kc = king[king_piece]
        if king_under_attack(board, kr, kc, color == 1):
            return -100000 - depth
        return 0
    if depth == 0:
        complex_bonus = evaluate_complex(board, tracker, offboard, king)
        cornering_bonus = Cornering(current_score + complex_bonus, king,offboard)
        final_score = current_score + complex_bonus+cornering_bonus
        return quiescence_search(board, alpha, beta, color, final_score, offboard, king, tracker)
    if tt_move in moves:
        moves.remove(tt_move)
        moves.insert(0, tt_move)
    best_move = None
    for move in moves:
        new_score = get_incremental_score(board, move, current_score)
        captured, next_hash = apply_move(board, *move, offboard, king, current_hash,tracker)
        evaluation = -Search(board, depth - 1, -beta, -alpha, -color, new_score, offboard, king, next_hash, tracker)
        unapply_move(board, *move, captured, offboard, king,tracker)
        if evaluation >= beta:
            transposition_table[current_hash] = (depth, beta, TT_BETA, move)
            return beta
        if evaluation > alpha:
            alpha = evaluation
            best_move = move            
    if best_move is None:
        transposition_table[current_hash] = (depth, original_alpha, TT_ALPHA, None)
    else:
        transposition_table[current_hash] = (depth, alpha, TT_EXACT, best_move)
    return alpha
#to find offboard pieces
def get_offboard_pieces(board):
    counts = PIECE_LIMITS.copy()
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            piece = board[r][c]
            if piece in counts:
                counts[piece] -= 1
    return counts
#Main I/O function
def get_best_move(board: np.ndarray, playing_white: bool = True) -> Optional[str]:
    global remaining_time, start_time_of_search, time_limit
    
    start_time_of_search = time.perf_counter()
    time_limit = remaining_time / 20
    transposition_table.clear()     
    offboard_tracker = get_offboard_pieces(board)
    offboard_tracker["Total"] = sum(offboard_tracker.values())
    if not is_endgame(offboard_tracker):
        PSTS[1] = dynamic_pawn_pst(board, PST1, True)
        PSTS[6] = dynamic_pawn_pst(board, PST6, False)
    else:
        PSTS[5] = PST15
        PSTS[10] = PST20
    wk = np.where(board == WHITE_KING)
    bk = np.where(board == BLACK_KING)
    king_tracker = {WHITE_KING: (wk[0][0], wk[1][0]),
                    BLACK_KING: (bk[0][0], bk[1][0])}
    piece_tracker = get_piece_tracker(board)
    root_score = evaluate(board)
    current_hash = get_initial_hash(board, playing_white)    
    best_move = None
    current_color = 1 if playing_white else -1
    all_moves = score_moves(board, get_all_moves(board, playing_white, offboard_tracker, king_tracker))
    if len(all_moves) == 1:
        return format_move(*all_moves[0])
    if not all_moves:
        return None
    #Iterative deepening
    try:
        for depth in range(1, 20):
            current_best_at_depth = None
            max_eval = -float('inf')
            alpha = -float('inf')
            beta = float('inf')
            for move in all_moves:
                new_score = get_incremental_score(board, move, root_score)
                captured, next_hash = apply_move(board, *move, offboard_tracker, king_tracker, current_hash,piece_tracker)                   
                score = -Search(board, depth - 1, -beta, -alpha, -current_color, new_score, offboard_tracker, king_tracker, next_hash,piece_tracker)
                unapply_move(board, *move, captured, offboard_tracker, king_tracker,piece_tracker)
                if score > max_eval:
                    max_eval = score
                    current_best_at_depth = move
                alpha = max(alpha, score)
            if current_best_at_depth:
                best_move = current_best_at_depth
                all_moves.remove(best_move)
                all_moves = [best_move] + all_moves
            if max_eval > 90000:
                break
            elapsed = time.perf_counter() - start_time_of_search
            if elapsed > (time_limit * 0.6):
                break
    except TimeoutException:
        pass
    elapsed = time.perf_counter() - start_time_of_search
    remaining_time -= elapsed
    return format_move(*best_move) if best_move else None

if __name__ == "__main__":
    # Example: standard-ish starting position on a 6x6 board
    # White pieces on rows 4-5, Black pieces on rows 0-1
    initial_board = np.array([
        [ 2,  3,  4,  5,  3,  2],   # Row 1 (A1–F1) — White back rank
        [ 1,  1,  1,  1,  1,  1],   # Row 2         — White pawns
        [ 0,  0,  0,  0,  0,  0],   # Row 3
        [ 0,  0,  0,  0,  0,  0],   # Row 4
        [ 6,  6,  6,  6,  6,  6],   # Row 5         — Black pawns
        [ 7,  8,  9, 10,  8,  7],   # Row 6 (A6–F6) — Black back rank
    ], dtype=int)
    
    print("Board:\n", initial_board)
    move = get_best_move(initial_board, playing_white=True)
    print("Best move for White:", move)
