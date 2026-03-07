# RoboGambit – Task 2  
## ArUco-Based Board State Estimation

In this task you will implement a **vision-based perception pipeline** that reconstructs the current board state from an overhead image of the RoboGambit arena.

Your goal is to detect **ArUco markers**, estimate the pose of the board, and infer the **6×6 board configuration**.

The perception pipeline should detect game pieces and determine their corresponding positions on the board.

---

# Files Provided

You will receive the following files:

```
task2/
│
├── perception.py          # Starter code
├── requirements.txt       # Python dependencies
│
├── input/
│   ├── board_1.png
│   ├── board_2.png
│   ├── board_3.png
│   └── ...
│
└── README.md              # Instructions
```

---

# Setup Instructions

## 1. Install Python

Ensure that **Python 3.8 or later** is installed.

Check your Python installation using:

```bash
python --version
```

---

## 2. Install Required Packages

Navigate to the task folder and install the required dependencies:

```bash
pip install -r requirements.txt
```

The required libraries are:

- **opencv-contrib-python** (for ArUco detection)
- **numpy**

---

# Running the Starter Code

Run the perception script by passing an image file as input.

Example:

```bash
python perception.py input/board_1.png
```

Two windows will appear:

### 1. Detected Markers

Displays the input image with detected ArUco markers.

### 2. Game Board

Displays the reconstructed **6×6 board state**.

---

# Your Task

You must implement the perception pipeline by completing the missing sections in the starter code.

The following components must be implemented.

---

# 1. Marker Detection

Detect ArUco markers present in the image.

Hint:

```python
corners, ids, rejected = detector.detectMarkers(gray_image)
```

---

# 2. Identify Corner Markers

The arena contains **four reference markers** used to estimate the board pose.

Corner marker IDs:

```
21
22
23
24
```

You must extract the **pixel coordinates of these markers** and map them to their known **world coordinates**. 

World coordinates are provided in the starter code.

---

# 3. Compute Homography

Use the detected corner markers to compute the transformation between **image coordinates** and **world coordinates**.

Hint:

```python
cv2.findHomography()
```

The homography matrix allows conversion of marker positions from pixel space into board coordinates.

---

# 4. Pixel → World Coordinate Conversion

Implement the `pixel_to_world()` function.

This function converts pixel coordinates into world coordinates using the homography matrix.

Hint:

```python
cv2.perspectiveTransform()
```

---

# 5. Board Reconstruction

For each detected **piece marker**:

1. Compute the **center pixel coordinate**
2. Convert it to **world coordinates**
3. Map the world coordinate to the **closest board square**

Call:

```python
place_piece_on_board()
```

to update the board array.

---

# Marker IDs

## Corner Markers (Reference Markers)

Used for board pose estimation.

```
21
22
23
24
```

## Game Piece Markers

Each game piece has an ArUco marker.

```
1 – 10
```

These correspond to the piece identifiers used in the board state.

---

# Board Representation

The board is represented as a **6×6 NumPy array**.

Example:

```
[[0 0 0 0 0 0]
 [0 1 0 0 6 0]
 [0 0 0 0 0 0]
 [0 0 0 7 0 0]
 [0 0 0 0 0 0]
 [0 0 0 0 0 0]]
```

Where:

```
0 = empty square
1–10 = piece IDs
```

Each cell corresponds to a board square.

---

# Functions You Must Implement

You are expected to complete the following functions in `perception.py`.

### process_image()

Main perception pipeline.

Tasks include:

- Detect ArUco markers
- Identify corner markers
- Compute homography
- Detect piece markers
- Convert pixel coordinates to world coordinates
- Place pieces on the board

---

### pixel_to_world()

Convert pixel coordinates into world coordinates using the homography matrix.

---

### place_piece_on_board()

Place detected pieces into the correct board square.

---

# Functions You Must NOT Modify

The following functions are provided by the organisers and must **not be modified**:

```
prepare_image()
visualize_board()
main()
```

Camera calibration parameters are also fixed and must not be changed.

---

# Evaluation

Your code will be evaluated on **multiple unseen board images**.

Your perception pipeline must:

- detect markers correctly
- estimate board pose
- reconstruct the board configuration accurately

Hardcoding board states or marker positions will result in disqualification.

---

# Submission

Submit the following file:

```
perception.py
```

Your program must run using:

```bash
python perception.py image.png
```

without requiring any modification.

---

# Debugging Tips

While developing your solution:

- Verify marker detection visually
- Print detected marker IDs and coordinates
- Check homography by transforming known points

Using visualization during development is highly recommended.

---

# Good Luck

Design your perception pipeline carefully and ensure it works reliably across different board configurations.