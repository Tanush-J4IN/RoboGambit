import game
import numpy as np
import requests
import argparse
import serial
import time
import perception
from perception import BoardPerception


import json

# Board geometry — must match perception.py
_COL_MAP     = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5}

# Arm Z heights in mm — tune these to your physical setup
Z_HOVER = 200   # safe travel height above the board
Z_PICK  =  20   # surface level to engage the piece

ARM_SPD = 0.3   # speed for T=104 command
T_ANGLE = 3.14  # wrist angle (rad) — straight down

# Off-board discard zone for captured/promoted-out pieces (world mm)
DISCARD_X, DISCARD_Y = -150, 0

# Reserve positions for promotion pieces (world mm) — tune to your layout
RESERVE_POS = {
    4: (-150, -100),   # White Queen  reserve
    3: (-150, -200),   # White Bishop reserve
    2: (-150, -300),   # White Knight reserve
    9:  (450, -100),   # Black Queen  reserve
    8:  (450, -200),   # Black Bishop reserve
    7:  (450, -300),   # Black Knight reserve
}



BOARD = np.zeros((6, 6), dtype=int)
# Serial port for the arm
ser = serial.Serial("COM4", baudrate=115200, dsrdtr=None)
ser.setRTS(False)
ser.setDTR(False)
# Serial port for the Solenoid
ser2 = serial.Serial('COM3', 115200) 
POSES = {}
vision_system = BoardPerception()

def get_board_state() -> np.ndarray:
    """Use the perception module to get the current board state."""
    # add code to update BOARD using perception.board
    global BOARD, POSES
    latest_board, latest_poses = vision_system.get_latest_state()
    if latest_board is not None:
        BOARD = latest_board
        POSES = latest_poses
    return BOARD

def move(playing_white) -> str:
    """Determine the best move using the game module."""
    return game.get_best_move(get_board_state(),playing_white)


def movetocmd(move: str) -> list:
    """
    Convert a move string into an ordered list of robot steps.

    Each element is either:
      - A JSON string  → pass to send_cmd()
      - "PICK"         → call pick()
      - "PLACE"        → call place()

    """

    # ── Internal helpers ──────────────────────────────────────────────────────

    def cell_to_world(cell: str):
        """'B3' → (wx, wy) in mm, using the same formula as perception.py."""
        col = _COL_MAP[cell[0].upper()]
        row = int(cell[1]) - 1
        wx = perception.TOP_LEFT_X - (row * perception.SQUARE_SIZE + perception.SQUARE_SIZE / 2)
        wy = perception.TOP_LEFT_Y - (col * perception.SQUARE_SIZE + perception.SQUARE_SIZE / 2)
        return wx, wy

    def arm_goto(x, y, z):
        """Build a Waveshare T=104 (CMD_XYZT_GOAL_CTRL) JSON command."""
        return json.dumps({
            "T": 104,
            "x": round(x, 1),
            "y": round(y, 1),
            "z": round(z, 1),
            "t": T_ANGLE,
            "spd": ARM_SPD
        })

    def pick_from(wx, wy):
        """Steps to hover → descend → PICK → raise at world position."""
        return [
            arm_goto(wx, wy, Z_HOVER),   # hover above cell
            arm_goto(wx, wy, Z_PICK),    # descend to piece
            "PICK",                       # electromagnet ON
            arm_goto(wx, wy, Z_HOVER),   # raise back up
        ]

    def place_at(wx, wy):
        """Steps to hover → descend → PLACE → raise at world position."""
        return [
            arm_goto(wx, wy, Z_HOVER),   # hover above target
            arm_goto(wx, wy, Z_PICK),    # descend to surface
            "PLACE",                      # electromagnet OFF
            arm_goto(wx, wy, Z_HOVER),   # raise back up
        ]

    # ── Parse move string ─────────────────────────────────────────────────────
    move_str = move
    promo_id = None

    if ":" in move_str:
        _, move_str = move_str.split(":", 1)       # strip "piece_id:"

    if "=" in move_str:
        move_str, promo_str = move_str.split("=", 1)
        promo_id = int(promo_str)                  # piece to promote to

    src_cell, dst_cell = move_str.split("->")
    src_x, src_y = cell_to_world(src_cell)
    dst_x, dst_y = cell_to_world(dst_cell)

    # ── Check for capture (enemy piece sitting at destination?) ───────────────
    board = get_board_state()
    dst_row = int(dst_cell[1]) - 1
    dst_col = _COL_MAP[dst_cell[0].upper()]
    is_capture = board[dst_row][dst_col] != 0

    steps = []

    # ── 1. Capture: clear the destination square first ────────────────────────
    if is_capture:
        steps += pick_from(dst_x, dst_y)
        steps += place_at(DISCARD_X, DISCARD_Y)

    # ── 2. Main move: lift our piece from source, set it at destination ───────
    steps += pick_from(src_x, src_y)
    steps += place_at(dst_x, dst_y)

    # ── 3. Promotion: swap pawn for the promoted piece ────────────────────────
    if promo_id is not None:
        res_x, res_y = RESERVE_POS[promo_id]

        # Remove pawn from promotion square → discard
        steps += pick_from(dst_x, dst_y)
        steps += place_at(DISCARD_X, DISCARD_Y)

        # Bring promoted piece from reserve → promotion square
        steps += pick_from(res_x, res_y)
        steps += place_at(dst_x, dst_y)

    return steps


def pick():
    """Activate the electromagnet to grip a piece."""
    ser2.write(b'1')



def place():
    """Deactivate the electromagnet to release a piece."""
    ser2.write(b'0')



def send_cmd(command: str):
    """Send a single JSON command to the robot arm via HTTP."""
    print(f"Sending command: {command}")
    ser.write(command.encode() + b'\n')

COL_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F']
LOG_FILE = "game_log.txt"

def _idx_to_cell(row: int, col: int) -> str:
    return f"{COL_LETTERS[col]}{row + 1}"


def log_move(prev_board: np.ndarray, curr_board: np.ndarray, log_file: str = LOG_FILE):
    vacated  = []   # had a piece, now empty  → this is the source square
    arrived  = []   # was empty, now has piece → destination (no capture)
    replaced = []   # had one piece, now different piece → capture destination

    for r in range(6):
        for c in range(6):
            pv, cv = int(prev_board[r][c]), int(curr_board[r][c])
            if pv == cv:
                continue
            if pv != 0 and cv == 0:
                vacated.append((r, c, pv))
            elif pv == 0 and cv != 0:
                arrived.append((r, c, cv))
            elif pv != 0 and cv != 0:
                replaced.append((r, c, pv, cv))

    if not vacated:
        return 

    # For perception glitches, confirming source of moves
    src_r, src_c, moved_piece = vacated[0]
    src_cell = _idx_to_cell(src_r, src_c)

    promo_id = None

    if arrived:                                # piece landed on an empty square
        dst_r, dst_c, dst_piece = arrived[0]
        if dst_piece != moved_piece:           # different piece ID → promotion
            promo_id = dst_piece
    elif replaced:                             # piece landed on an occupied square → capture
        dst_r, dst_c, _, dst_piece = replaced[0]
        if dst_piece != moved_piece:           # different piece ID → capture + promotion (rare)
            promo_id = dst_piece
    else:
        return  # Can't determine destination; skip

    dst_cell = _idx_to_cell(dst_r, dst_c)

    
    move_str = f"{moved_piece}:{src_cell}->{dst_cell}"
    if promo_id is not None:
        move_str += f"={promo_id}"

    with open(log_file, 'a') as f:
        f.write(move_str + '\n')
    return (moved_piece, src_r, src_c, dst_r, dst_c, promo_id)


def log_result(result: str, log_file: str = LOG_FILE):
    with open(log_file, 'a') as f:
        f.write(f"Result: {result}\n\n")

def check_legal(prev_board, move_tuple):
    if move_tuple is None: return False
    moved_piece = move_tuple[0]
    is_white_move = game.is_white(moved_piece)
    offboard = game.get_offboard_pieces(prev_board)
    offboard["Total"] = sum(offboard.values())
    wk = np.where(prev_board == game.WHITE_KING)
    bk = np.where(prev_board == game.BLACK_KING)
    king = {game.WHITE_KING: (int(wk[0][0]), int(wk[1][0])),
            game.BLACK_KING: (int(bk[0][0]), int(bk[1][0]))}
    return move_tuple in game.get_all_moves(prev_board, is_white_move, offboard, king)

def get_stable_board_state(required_frames=10):
    """
    Waits for the board to be identical for 'required_frames' 
    consecutive successful reads.
    """
    confidence = 0
    last_detected_board = None
    
    while confidence < required_frames:
        current_board = get_board_state() 
        
        if last_detected_board is not None and np.array_equal(current_board, last_detected_board):
            confidence += 1
        else:
            confidence = 0
            last_detected_board = current_board
        time.sleep(0.05) 
        
    return last_detected_board

   
if __name__ == "__main__":
    color=input("Which color is the bot playing?(w/b): ")
    game.remaining_time=int(input("Enter the Time control(10/15): "))*60
    playing_white=(color=='w')
    if playing_white==True: t=0
    else: t=-1
    BOARD= get_stable_board_state()
    while True:
        curr = get_stable_board_state()
        if t%4==0:
            send_cmd(json.dumps({'T':100}))
            best_move = move(playing_white)
            if best_move is None:
                print("No moves available — game over.")
                log_result(input("Enter result"),'rglog.txt')
                break
            print(f"Best move: {best_move}")
            for step in movetocmd(best_move):
                if step == "PICK":  pick()
                elif step == "PLACE": place()
                else:send_cmd(step)
                time.sleep(0.25)
            send_cmd(json.dumps({'T':100}))
            BOARD=get_stable_board_state()
            t+=1
        elif not np.array_equal(curr, BOARD):
            move_tuple = log_move(BOARD, curr, 'rglog.txt')
            if t%4 != 0 and not check_legal(BOARD, move_tuple):
                with open('rglog.txt', 'a') as f:
                    f.write("previous move was illegal\n")
                continue
            elif t%4==2:
                start=time.time()
            elif t%4==3 and t>0:
                game.remaining_time-=time.time()-start#to account for human player's move and robotic movement delay
            BOARD=get_stable_board_state()
            t+=1
        time.sleep(0.1)   # brief pause before sensing the next board state
