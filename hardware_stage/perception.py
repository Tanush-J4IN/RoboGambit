import math
import cv2
import cv2.aruco as aruco
import numpy as np
import socket
import struct

# ── Configuration ─────────────────────────────────────────────────────────────
SERVER_IP   = '192.168.4.2' 
SERVER_PORT = 9992

# Updated for 640x480 resolution stream
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
SQUARE_SIZE = 60
TOP_LEFT_X  = 180
TOP_LEFT_Y  = 180
BOARD_SIZE  = 6
PIECE_IDS   = set(range(1, 11))

class BoardPerception:
    def __init__(self, connect_socket=True):
        """Initialize ArUco detector and connect to the camera stream."""
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
        
        # Conditionally connect to socket
        if connect_socket:
            print(f"Connecting to {SERVER_IP}:{SERVER_PORT} ...")
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((SERVER_IP, SERVER_PORT))
            print("Connected ✓")
            self.payload_size = struct.calcsize("Q")
            self.data_buffer  = b""
        else:
            self.client_socket = None

    def _recv_frame(self):
        """Internal helper to read one frame from the socket."""
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

        frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        return frame

    def _pixel_to_world(self, px, py):
        """Converts pixel coordinates to real-world coordinates using Homography."""
        pt = cv2.perspectiveTransform(np.array([[[px, py]]], dtype=np.float32), self.H_matrix)
        return float(pt[0][0][0]), float(pt[0][0][1])

    def _world_to_cell(self, wx, wy):
        """Maps physical world coordinates to the 6x6 grid."""
        best_row, best_col, min_dist = None, None, float('inf')
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                # FIXED: col is X-axis, row is Y-axis
                cx = TOP_LEFT_X - (col * SQUARE_SIZE + SQUARE_SIZE / 2)
                cy = TOP_LEFT_Y - (row * SQUARE_SIZE + SQUARE_SIZE / 2)
                d  = math.hypot(wx - cx, wy - cy)
                if d < min_dist:
                    min_dist, best_row, best_col = d, row, col
        return best_row, best_col

    def get_latest_state(self):
        """Reads a live frame from the socket and returns the board state."""
        if not self.client_socket:
            print("[ERROR] Cannot fetch live frame: Socket not connected.")
            return None, None
            
        frame = self._recv_frame()
        if frame is None:
            return None, None
            
        return self.get_latest_state_from_frame(frame)

    def get_latest_state_from_frame(self, frame):
        """Processes a single provided frame to extract board state and poses."""
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        poses = {} 

        if ids is not None:
            # Draw green boxes for the debug image
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Update corners for homography
            for i, mid in enumerate(ids.flatten()):
                if mid in CORNER_WORLD:
                    self.corner_pixels[mid] = np.mean(corners[i][0], axis=0)

            # Lock homography if all 4 corners are found
            if self.H_matrix is None and len(self.corner_pixels) == 4:
                pixel_pts = np.array([self.corner_pixels[m] for m in [21, 22, 23, 24]], dtype=np.float32)
                world_pts = np.array([CORNER_WORLD[m]  for m in [21, 22, 23, 24]], dtype=np.float32)
                self.H_matrix, _ = cv2.findHomography(pixel_pts, world_pts)
                print("Homography locked ✓")

            # Extract poses and build board
            if self.H_matrix is not None:
                for i, mid in enumerate(ids.flatten()):
                    if mid not in PIECE_IDS:
                        continue
                    
                    c = corners[i][0]
                    px, py = float(np.mean(c[:, 0])), float(np.mean(c[:, 1]))
                    
                    wx, wy = self._pixel_to_world(px, py)
                    poses[mid] = (wx, wy)
                    
                    row, col = self._world_to_cell(wx, wy)
                    if row is not None:
                        if board[row][col] != 0:
                            print(f"\n[WARNING] PIECE OVERLAP! ID {mid} and ID {board[row][col]} are both snapping to cell [{row}][{col}]. Center them!")
                        board[row][col] = mid

        return board, poses

    def cleanup(self):
        """Close the socket connection."""
        if self.client_socket:
            self.client_socket.close()
