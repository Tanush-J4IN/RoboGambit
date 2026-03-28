import math
import cv2
import cv2.aruco as aruco
import numpy as np
import socket
import struct

# ── Configuration ─────────────────────────────────────────────────────────────
SERVER_IP   = '10.194.7.31' 
SERVER_PORT = 9999

CAMERA_MATRIX = np.array([
    [343.49, 0,      320.0],
    [0,      457.99, 240.0],
    [0,      0,      1.0]
], dtype=np.float32)
DIST_COEFFS = np.zeros((1, 5), dtype=np.float32)

CORNER_WORLD = {
    21: (212.5,  212.5),
    22: (212.5, -212.5),
    23: (-212.5, -212.5),
    24: (-212.5,  212.5),
}

# RESTORED VISION MATH (DO NOT PUT ROBOT CALIBRATION HERE)
SQUARE_SIZE = 60
TOP_LEFT_X  = 180
TOP_LEFT_Y  = 180
BOARD_SIZE  = 6
PIECE_IDS   = set(range(1, 11))

class BoardPerception:
    def __init__(self, connect_socket=True):
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        params     = aruco.DetectorParameters()
        params.cornerRefinementMethod      = aruco.CORNER_REFINE_SUBPIX
        params.adaptiveThreshWinSizeMin    = 3
        params.adaptiveThreshWinSizeMax    = 35
        params.adaptiveThreshWinSizeStep   = 10
        params.minMarkerPerimeterRate      = 0.01
        params.maxMarkerPerimeterRate      = 4.0
        params.polygonalApproxAccuracyRate = 0.03
        params.minCornerDistanceRate       = 0.05
        params.minDistanceToBorder         = 1
        self.detector = aruco.ArucoDetector(aruco_dict, params)

        self.H_matrix      = None
        self.corner_pixels = {}
        
        if connect_socket:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((SERVER_IP, SERVER_PORT))
            self.payload_size = struct.calcsize("Q")
            self.data_buffer  = b""
        else:
            self.client_socket = None

    def _recv_frame(self):
        while len(self.data_buffer) < self.payload_size:
            packet = self.client_socket.recv(4096)
            if not packet: return None
            self.data_buffer += packet
        packed_msg_size = self.data_buffer[:self.payload_size]
        self.data_buffer = self.data_buffer[self.payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]
        while len(self.data_buffer) < msg_size:
            packet = self.client_socket.recv(4096)
            if not packet: return None
            self.data_buffer += packet
        frame_data = self.data_buffer[:msg_size]
        self.data_buffer = self.data_buffer[msg_size:]
        return cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)

    def _pixel_to_world(self, px, py):
        pt = cv2.perspectiveTransform(np.array([[[px, py]]], dtype=np.float32), self.H_matrix)
        return float(pt[0][0][0]), float(pt[0][0][1])

    def _world_to_cell(self, wx, wy):
        best_row, best_col, min_dist = None, None, float('inf')
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                cx = TOP_LEFT_X - (col * SQUARE_SIZE + SQUARE_SIZE / 2)
                cy = TOP_LEFT_Y - (row * SQUARE_SIZE + SQUARE_SIZE / 2)
                d  = math.hypot(wx - cx, wy - cy)
                if d < min_dist:
                    min_dist, best_row, best_col = d, row, col
        return best_row, best_col

    def get_latest_state(self):
        if not self.client_socket: return None, None
        frame = self._recv_frame()
        if frame is None: return None, None
        return self.get_latest_state_from_frame(frame)

    def get_latest_state_from_frame(self, frame):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        poses = {} 

        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                if mid in CORNER_WORLD:
                    self.corner_pixels[mid] = np.mean(corners[i][0], axis=0)
            if self.H_matrix is None and len(self.corner_pixels) == 4:
                pixel_pts = np.array([self.corner_pixels[m] for m in [21, 22, 23, 24]], dtype=np.float32)
                world_pts = np.array([CORNER_WORLD[m]  for m in [21, 22, 23, 24]], dtype=np.float32)
                self.H_matrix, _ = cv2.findHomography(pixel_pts, world_pts)
            if self.H_matrix is not None:
                for i, mid in enumerate(ids.flatten()):
                    if mid not in PIECE_IDS: continue
                    c = corners[i][0]
                    px, py = float(np.mean(c[:, 0])), float(np.mean(c[:, 1]))
                    wx, wy = self._pixel_to_world(px, py)
                    poses[mid] = (wx, wy)
                    row, col = self._world_to_cell(wx, wy)
                    if row is not None:
                        board[row][col] = mid
        return board, poses

    def cleanup(self):
        if self.client_socket: self.client_socket.close()
