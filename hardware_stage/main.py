import game
import numpy as np
import requests
import argparse
import serial
import time
import perception
from perception import BoardPerception
import json
import os

# ─── LINUX OS COMPATIBILITY FIX ───────────────────────────────────────────────
original_apply_move = game.apply_move

def safe_apply_move(board, piece, r1, c1, r2, c2, promo, offboard, king_tracker, current_hash, tracker):
    if type(current_hash) is int and current_hash == 0:
        current_hash = np.uint64(0)
    return original_apply_move(board, piece, r1, c1, r2, c2, promo, offboard, king_tracker, current_hash, tracker)

game.apply_move = safe_apply_move
# ──────────────────────────────────────────────────────────────────────────────

# ── DYNAMIC CALIBRATION LOADER ──
CALIB = {
    "TOP_LEFT_X": 133.0,
    "TOP_LEFT_Y": -119.4,
    "SQUARE_SIZE_X": 71.0,
    "SQUARE_SIZE_Y": 59.0
}
if os.path.exists("robot_calib.json"):
    with open("robot_calib.json", "r") as f:
        CALIB = json.load(f)
        print("Loaded physical robot calibration from robot_calib.json ✓")
else:
    print("No robot_calib.json found. Using fallback coordinates.")
# ────────────────────────────────

# Board geometry
_COL_MAP     = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5}

# Arm Z heights
Z_HOVER = 200   # safe travel height above the board
Z_PICK  = -80   # surface level to engage the piece

ARM_SPD = 0.3   # speed for T=104 command
T_ANGLE = 3.14  # wrist angle (rad) — straight down

# Off-board discard zone for captured/promoted-out pieces (world mm)
DISCARD_X, DISCARD_Y = -150, 0

# Reserve positions for promotion pieces (world mm)
RESERVE_POS = {
    4: (-150, -100),   # White Queen  reserve
    3: (-150, -200),   # White Bishop reserve
    2: (-150, -300),   # White Knight reserve
    9:  (450, -100),   # Black Queen  reserve
    8:  (450, -200),   # Black Bishop reserve
    7:  (450, -300),   # Black Knight reserve
}

BOARD = np.zeros((6, 6), dtype=int)
# Serial ports
ser = serial.Serial("/dev/ttyUSB1", baudrate=115200, dsrdtr=None)
ser.setRTS(False)
ser.setDTR(False)
ser2 = serial.Serial('/dev/ttyUSB0', 115200) 

POSES = {}
vision_system = BoardPerception()

def get_board_state() -> np.ndarray:
    """Use the perception module to get the current board state."""
    global BOARD, POSES
    latest_board, latest_poses = vision_system.get_latest_state()
    if latest_board is not None:
        BOARD = latest_board
        POSES = latest_poses
    return BOARD

def move(playing_white) -> str:
    """Determine the best move using the game module."""
    return game.get_best_move(get_board_state(), playing_white)

def movetocmd(move_str: str) -> list:
    def cell_to_world(cell: str):
        """Translates the 6x6 grid into physical Robot coordinates."""
        row_idx = ord(cell[0].upper()) - ord('A')
        col_idx = int(cell[1]) - 1
        
        # Pulls directly from your CALIB dictionary
        wx = CALIB["TOP_LEFT_X"] + (row_idx * CALIB["SQUARE_SIZE_X"])
        wy = CALIB["TOP_LEFT_Y"] + (col_idx * CALIB["SQUARE_SIZE_Y"])
        return wx, wy

    def arm_goto(x, y, z):
        return json.dumps({
            "T": 104,
            "x": round(x, 1),
            "y": round(y, 1),
            "z": round(z, 1),
            "t": T_ANGLE,
            "spd": ARM_SPD
        })

    def pick_from(wx, wy):
        return [
            arm_goto(wx, wy, Z_HOVER),
            arm_goto(wx, wy, Z_PICK),
            "PICK",
            arm_goto(wx, wy, Z_HOVER),
        ]

    def place_at(wx, wy):
        return [
            arm_goto(wx, wy, Z_HOVER),
            arm_goto(wx, wy, Z_PICK),
            "PLACE",
            arm_goto(wx, wy, Z_HOVER),
        ]

    promo_id = None
    if ":" in move_str:
        _, move_str = move_str.split(":", 1)

    if "=" in move_str:
        move_str, promo_str = move_str.split("=", 1)
        promo_id = int(promo_str)

    src_cell, dst_cell = move_str.split("->")
    src_x, src_y = cell_to_world(src_cell)
    dst_x, dst_y = cell_to_world(dst_cell)

    board = get_board_state()
    dst_row = int(dst_cell[1]) - 1
    dst_col = _COL_MAP[dst_cell[0].upper()]
    is_capture = board[dst_row][dst_col] != 0

    steps = []

    if is_capture:
        steps += pick_from(dst_x, dst_y)
        steps += place_at(DISCARD_X, DISCARD_Y)

    steps += pick_from(src_x, src_y)
    steps += place_at(dst_x, dst_y)

    if promo_id is not None:
        res_x, res_y = RESERVE_POS[promo_id]
        steps += pick_from(dst_x, dst_y)
        steps += place_at(DISCARD_X, DISCARD_Y)
        steps += pick_from(res_x, res_y)
        steps += place_at(dst_x, dst_y)

    return steps

def pick():
    ser2.write(b'1')

def place():
    ser2.write(b'0')

def send_cmd(command: str):
    print(f"Sending command: {command}")
    ser.write(command.encode() + b'\n')

COL_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F']
LOG_FILE = "game_log.txt"

def _idx_to_cell(row: int, col: int) -> str:
    return f"{COL_LETTERS[col]}{row + 1}"

def log_move(prev_board: np.ndarray, curr_board: np.ndarray, log_file: str = LOG_FILE):
    vacated  = []
    arrived  = []
    replaced = []

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

    src_r, src_c, moved_piece = vacated[0]
    src_cell = _idx_to_cell(src_r, src_c)
    promo_id = None

    if arrived:
        dst_r, dst_c, dst_piece = arrived[0]
        if dst_piece != moved_piece:
            promo_id = dst_piece
    elif replaced:
        dst_r, dst_c, _, dst_piece = replaced[0]
        if dst_piece != moved_piece:
            promo_id = dst_piece
    else:
        return

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
    confidence = 0
    last_detected_board = None
    
    while confidence < required_frames:
        current_board = get_board_state() 
        
        if vision_system.H_matrix is not None and np.any(current_board != 0):
            # Check for BOTH Kings before allowing the game to proceed
            if np.any(current_board == game.WHITE_KING) and np.any(current_board == game.BLACK_KING):
                if last_detected_board is not None and np.array_equal(current_board, last_detected_board):
                    confidence += 1
                else:
                    confidence = 0
                    last_detected_board = current_board.copy()
            else:
                confidence = 0
                print("Waiting for both Kings to be clearly visible...", end="\r")
        else:
            confidence = 0
            if vision_system.H_matrix is None:
                print("Waiting for all 4 corners to lock Homography...", end="\r")
        
        time.sleep(0.05) 
        
    print("\nValid stable board acquired!")
    return last_detected_board

if __name__ == "__main__":
    color = input("Which color is the bot playing?(w/b): ")
    game.remaining_time = int(input("Enter the Time control(10/15): ")) * 60
    playing_white = (color == 'w')
    if playing_white == True: t = 0
    else: t = -1
    
    try:
        BOARD = get_stable_board_state()
        while True:
            curr = get_stable_board_state()
            if t % 4 == 0:
                send_cmd(json.dumps({'T':100}))
                print('Calculating next move...')
                best_move = move(playing_white)
                if best_move is None:
                    print("No moves available — game over.")
                    log_result(input("Enter result: "), 'rglog.txt')
                    break
                print(f"Best move: {best_move}")
                
                for step in movetocmd(best_move):
                    if step == "PICK":  
                        pick()
                    elif step == "PLACE": 
                        place()
                    else: 
                        send_cmd(step)
                    time.sleep(1.0) # Ensures the arm has time to finish moving
                    
                send_cmd(json.dumps({'T':100}))
                BOARD = get_stable_board_state()
                t += 1
                
            elif not np.array_equal(curr, BOARD):
                move_tuple = log_move(BOARD, curr, 'rglog.txt')
                if t % 4 != 0 and not check_legal(BOARD, move_tuple):
                    with open('rglog.txt', 'a') as f:
                        f.write("previous move was illegal\n")
                    continue
                elif t % 4 == 2:
                    start = time.time()
                elif t % 4 == 3 and t > 0:
                    game.remaining_time -= time.time() - start
                BOARD = get_stable_board_state()
                t += 1
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nShutting down safely...")
    finally:
        vision_system.cleanup()
        ser.close()
        ser2.close()
        print("Hardware disconnected.")
