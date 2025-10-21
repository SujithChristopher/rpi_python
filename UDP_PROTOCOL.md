# UDP Communication Protocol for stream_optimize_v2.py

## Overview

This document describes the UDP communication protocol between the Python ArUco tracking system and Godot. The v2 version provides explicit control over reference frame capture and state management.

## Connection Details

- **IP**: localhost (127.0.0.1)
- **Port**: 8000
- **Protocol**: UDP
- **Message Format**: ASCII text (commands) / Binary float32 array (responses)

## State Machine

The system operates in four distinct states:

1. **IDLE** (0)
   - Initial state
   - No reference frame captured
   - Waiting for `CAPTURE_REF` command

2. **REFERENCE_CAPTURED** (1)
   - Reference frame has been saved
   - Ready to start tracking
   - Can accept `START_TRACK` or `USER:` commands

3. **TRACKING** (2)
   - Actively tracking and sending coordinates
   - Coordinates are relative to reference frame
   - No data recording

4. **RECORDING** (3)
   - Tracking AND recording to CSV file
   - Associated with a hospital/user ID
   - Data saved to disk

## Commands (Godot → Python)

All commands are sent as ASCII bytes.

### Core Commands

| Command | Description | Valid States | Next State |
|---------|-------------|--------------|------------|
| `CAPTURE_REF` | Capture current pose as reference frame | IDLE, REFERENCE_CAPTURED | REFERENCE_CAPTURED |
| `RESET_REF` | Clear reference frame and return to IDLE | Any | IDLE |
| `START_TRACK` | Start tracking (requires reference) | REFERENCE_CAPTURED | TRACKING |
| `STOP_TRACK` | Stop tracking | TRACKING, RECORDING | REFERENCE_CAPTURED |
| `STATUS` | Request current state | Any | (no change) |
| `STOP` | Stop everything and exit program | Any | Exit |

### User/Recording Commands

| Command | Format | Description | Valid States | Next State |
|---------|--------|-------------|--------------|------------|
| `USER:<id>` | `USER:patient123` | Set hospital ID and start recording | REFERENCE_CAPTURED, TRACKING | RECORDING |
| `CHANGE:<id>` | `CHANGE:patient456` | Change to new user/close previous file | RECORDING | RECORDING |

### Example Command Sequence

```python
# Godot sends commands as bytes
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
python_address = ("localhost", 8000)

# 1. Capture reference frame when patient is in starting position
sock.sendto(b"CAPTURE_REF", python_address)

# 2. Start tracking
sock.sendto(b"START_TRACK", python_address)

# 3. Start recording with patient ID
sock.sendto(b"USER:patient_001", python_address)

# 4. Stop recording but keep tracking
sock.sendto(b"STOP_TRACK", python_address)

# 5. Reset for next patient
sock.sendto(b"RESET_REF", python_address)
```

## Responses (Python → Godot)

Python sends binary data as float32 array:

```
[state_code, x, y, z]
```

### State Codes

| Code | State | Meaning |
|------|-------|---------|
| 0.0 | IDLE | No reference frame, waiting |
| 1.0 | REFERENCE_CAPTURED | Reference set, ready to track |
| 2.0 | TRACKING | Actively tracking |
| 3.0 | RECORDING | Tracking and recording |
| -1.0 | ERROR | Error occurred (check console) |
| -99.0 | STOP | Shutdown signal |

### Receiving Data in Godot (GDScript)

```gdscript
extends Node

var udp := PacketPeerUDP.new()
var python_port := 8000

func _ready():
    # Listen for responses from Python
    udp.bind(python_port + 1)  # Use different port for receiving

func send_command(command: String):
    var packet = command.to_utf8_buffer()
    udp.set_dest_address("127.0.0.1", python_port)
    udp.put_packet(packet)

func _process(delta):
    if udp.get_available_packet_count() > 0:
        var bytes = udp.get_packet()
        var data = parse_float_array(bytes)

        var state_code = data[0]
        var position = Vector3(data[1], data[2], data[3])

        match state_code:
            0.0:
                print("Python is IDLE")
            1.0:
                print("Reference captured")
            2.0:
                print("Tracking: ", position)
            3.0:
                print("Recording: ", position)
                update_player_position(position)
            -1.0:
                print("ERROR from Python")
            -99.0:
                print("Python stopped")

func parse_float_array(bytes: PackedByteArray) -> Array:
    var result = []
    for i in range(0, bytes.size(), 4):
        result.append(bytes.decode_float(i))
    return result
```

## Workflow Examples

### Basic Tracking Workflow

1. **System starts** → State: IDLE
2. **Position patient at starting point**
3. **Send `CAPTURE_REF`** → State: REFERENCE_CAPTURED
4. **Send `START_TRACK`** → State: TRACKING
   - Python now sends coordinates continuously
5. **Send `STOP_TRACK`** when done → State: REFERENCE_CAPTURED
6. **Send `RESET_REF`** to prepare for next session → State: IDLE

### Recording Workflow

1. **System starts** → State: IDLE
2. **Send `CAPTURE_REF`** → State: REFERENCE_CAPTURED
3. **Send `USER:patient_id`** → State: RECORDING
   - Python sends coordinates AND saves to CSV
   - File: `~/Documents/NOARK/data/patient_id/Session-YYYY-MM-DD/MovementData/timestamp_data.csv`
4. **Send `STOP_TRACK`** when done → State: REFERENCE_CAPTURED
   - CSV file closed automatically

### Multi-Patient Session

```python
# Patient 1
send("CAPTURE_REF")
send("USER:patient_001")
# ... exercise ...
send("STOP_TRACK")

# Change to Patient 2 (uses same reference frame)
send("CHANGE:patient_002")
# ... exercise ...
send("STOP_TRACK")

# Reset for new session with different starting position
send("RESET_REF")
```

## Error Handling

### Common Errors

1. **"Cannot start tracking without reference frame"**
   - Cause: Tried `START_TRACK` or `USER:` before `CAPTURE_REF`
   - Solution: Send `CAPTURE_REF` first

2. **"No markers detected, cannot capture reference frame"**
   - Cause: No ArUco markers visible when `CAPTURE_REF` was sent
   - Solution: Ensure markers are visible in camera view

3. **"Lost connection to Godot, exiting..."**
   - Cause: No UDP messages received for 5 seconds
   - Solution: Send periodic heartbeat (e.g., `STATUS`) every 2-3 seconds

### Heartbeat Pattern

To prevent timeout, send periodic status checks:

```gdscript
var heartbeat_timer := 0.0

func _process(delta):
    heartbeat_timer += delta
    if heartbeat_timer > 2.0:  # Every 2 seconds
        send_command("STATUS")
        heartbeat_timer = 0.0
```

## Visual Feedback

The Python application displays status on the video window:

- **Red text "IDLE - Waiting for CAPTURE_REF"** → Need to capture reference
- **Green text "REF CAPTURED - Ready to track"** → Reference set, ready
- **Green text "TRACKING"** → Actively tracking
- **Green text "RECORDING"** → Recording data to file

## Data Format

### CSV Recording Format

When in RECORDING state, data is saved as:

```csv
Time,X,Y,Z
21/10/2025 14:23:45,0.123,-0.045,0.678
21/10/2025 14:23:45,0.125,-0.044,0.679
...
```

### Coordinate System

- **Origin**: Centroid of marker offsets at reference frame
- **Axes**: Based on first detected marker's orientation at reference
- **Units**: Meters
- **Filtering**: Exponential moving average (alpha=0.4)

## Configuration

Key parameters in `Config` class ([stream_optimize_v2.py:14-28](stream_optimize_v2.py#L14-L28)):

```python
FRAME_SIZE = (1200, 800)       # Camera resolution
MARKER_LENGTH = 0.05           # Marker size in meters
UDP_IP = "localhost"
UDP_PORT = 8000
DEFAULT_IDS = [4, 8, 12, 14, 20]  # Expected marker IDs
ALPHA = 0.4                    # Filter smoothing (0=no filter, 1=no smoothing)
```

## Differences from Original Version

### stream_optimize.py (Original)
- Reference frame captured automatically on first detection
- No explicit state management
- Limited command set
- Recording starts immediately with USER command
- No visual status feedback

### stream_optimize_v2.py (New)
- ✅ Explicit reference frame capture via `CAPTURE_REF`
- ✅ Clear state machine (IDLE → REF_CAPTURED → TRACKING → RECORDING)
- ✅ More commands for fine-grained control
- ✅ Separate tracking and recording states
- ✅ Visual status overlay on video
- ✅ Better error handling and feedback
- ✅ Proper CSV file closing
- ✅ Status query command
- ✅ Can reset reference frame without restarting

## Troubleshooting

### Python not receiving commands
- Check firewall settings
- Verify port 8000 is not in use: `netstat -an | findstr 8000`
- Ensure both Python and Godot use same IP/port

### Coordinates seem wrong
- Verify reference frame was captured at correct position
- Check marker IDs match `Config.DEFAULT_IDS`
- Verify marker offsets in `Config.MARKER_OFFSETS` are correct for your setup

### Recording file not created
- Check directory exists: `~/Documents/NOARK/data/`
- Verify write permissions
- Look for error messages in console

## Testing Without Godot

You can test the Python application using netcat or a simple Python script:

```python
import socket
import time

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
addr = ("localhost", 8000)

# Send commands
time.sleep(1)
sock.sendto(b"STATUS", addr)
time.sleep(1)
sock.sendto(b"CAPTURE_REF", addr)
time.sleep(1)
sock.sendto(b"START_TRACK", addr)
time.sleep(5)
sock.sendto(b"STOP", addr)
```
