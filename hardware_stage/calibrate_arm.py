import serial
import json
import os

# ── Configuration ──
ARM_PORT = "/dev/ttyUSB1"  # Update this if Linux assigns a different port tomorrow
BAUD_RATE = 115200
Z_HOVER = 200              # Safe travel height

def send_goto(ser, x, y, z):
    cmd = json.dumps({"T": 104, "x": round(x, 1), "y": round(y, 1), "z": round(z, 1), "t": 3.14, "spd": 0.3})
    ser.write(cmd.encode() + b'\n')

def get_point(ser, point_name, start_x, start_y):
    print(f"\n--- Calibrating {point_name} ---")
    curr_x, curr_y = start_x, start_y
    send_goto(ser, curr_x, curr_y, Z_HOVER)
    
    while True:
        val = input(f"[{point_name}] Enter 'X Y' (e.g. '150 -100'), or type 'save' to lock this point: ").strip().lower()
        if val == 'save':
            return curr_x, curr_y
        try:
            parts = val.split()
            if len(parts) == 2:
                curr_x, curr_y = float(parts[0]), float(parts[1])
                send_goto(ser, curr_x, curr_y, Z_HOVER)
            else:
                print("Invalid input. Provide two numbers separated by a space.")
        except ValueError:
            print("Invalid input. Use numbers.")

def main():
    try:
        ser = serial.Serial(ARM_PORT, baudrate=BAUD_RATE)
    except Exception as e:
        print(f"[ERROR] Failed to connect to {ARM_PORT}: {e}")
        return
        
    print("=== ROBOGAMBIT ARM CALIBRATION TOOL ===")
    print("You will jog the arm to the exact center of A1, and then F6.")
    
    # Defaults start near your last known good coordinates
    a1_x, a1_y = get_point(ser, "Square A1 (Top-Left)", 133.0, -119.0)
    f6_x, f6_y = get_point(ser, "Square F6 (Bottom-Right)", 488.0, 175.0)
    
    # ── Calculate Math ──
    # Grid is 6x6, meaning there are 5 square "gaps" between A1 and F6
    sq_x = (f6_x - a1_x) / 5.0
    sq_y = (f6_y - a1_y) / 5.0
    
    calib_data = {
        "TOP_LEFT_X": round(a1_x, 2),
        "TOP_LEFT_Y": round(a1_y, 2),
        "SQUARE_SIZE_X": round(sq_x, 2),
        "SQUARE_SIZE_Y": round(sq_y, 2)
    }
    
    with open("robot_calib.json", "w") as f:
        json.dump(calib_data, f, indent=4)
        
    print("\n=== CALIBRATION SUCCESSFUL ===")
    print(json.dumps(calib_data, indent=4))
    print("Saved to robot_calib.json. main.py will now use these values automatically.")
    ser.close()

if __name__ == "__main__":
    main()