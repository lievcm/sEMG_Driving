# EMG Steering Host

Delsys Trigno Avanti (6× EMG) -> steering angle -> Logitech G29/G920 (closed-loop constant-force feedback via SDL2).

## Setup (Windows, 64-bit Python)
1. `pip install -r requirements.txt`
2. Copy `DelsysAPI.dll` to `resources/DelsysAPI/resources/` (or set the folder in *Settings > Application settings*).
3. Enter the Delsys **key** and **license** in the same dialog.
4. Install Logitech G HUB and set the wheel range there (enter the same value as *Wheel range*).
   No Logitech SDK DLL is needed. The wheel is auto-detected; set *Device index* only if it isn't.
   If the G29 isn't detected, set its PS3/PS4 switch to PS3.
5. `python main.py`, then **EMG > Connect EMG** and **Wheel > Connect G29 / G920**.

No hardware? Set *EMG source* to `simulated` to try the GUI and pipeline.

## Configuration
`config/app_config.json` and `config/user_config.json` are loaded at start-up if present (defaults otherwise).
Copy the `*.example.json` files to get started, or use *Save* / *Save As…* / *Load…* in the settings dialogs.
Weights (`n`, `w` per channel) apply immediately; band-power settings rebuild the filters live;
Delsys changes and wheel device-index / axis changes need a reconnect.

## Signal chain
per channel: causal sliding-window FFT band metric -> trailing moving average (`avg_window_s`)
-> `clip(n * value, -1, 1)` -> arm = `clip(sum(w * that), -1, 1)` -> `(L + R) / 2`
-> `* lock_to_lock_deg / 2` degrees -> clamp (`max_target_deg`), speed- and acceleration-limit
(`max_slew_deg_per_s`, `max_accel_deg_per_s2`; the wheel brakes in time instead of overshooting)
-> PD controller (`kp`, `kd`) on the measured wheel angle -> constant force, capped at `max_force_pct`.
If no new EMG target arrives for `target_timeout_s`, the wheel is centred.

## Wheel control
The wheel is driven through SDL2 (PySDL2), so it works on Windows (G HUB driver), Linux
(in-kernel driver; needs read/write access to the wheel's `/dev/input/event*`) and macOS.
- **On connect** the motor ramps up a small force until the wheel moves 3°, to learn which way
  positive force turns it (this differs between drivers). Keep hands off the wheel for that second.
- **Every cycle** (`loop_hz`) the thread reads the wheel angle and sets one constant-force effect
  to `kp × (commanded − measured) + kd × (commanded − measured velocity)`, clamped to `max_force_pct`.
  The wheel's own centring spring is switched off so it doesn't fight the controller.
- **Safety:** if the wheel is more than `safety_max_error_deg` from the commanded angle (grabbed,
  or *Wheel range* doesn't match G HUB), the motor is released and the wheel disconnects.
  Disconnecting or closing the app also zeroes the force.

## Streaming the steering angle to another computer
The steering angle can be sent over Ethernet (or any IP network) to a second computer.

**On the other computer** copy `client/steering_client.py` (standard library only, Python 3.8+) and run

    python steering_client.py                  # TCP, port 5005
    python steering_client.py --protocol udp   # or UDP
    python steering_client.py --port 6000 --print-hz 0     # other port / print every message

It listens, waits for the host and prints lines such as
`16:33:42.652  #152  angle -19.92°  norm -0.044  L -0.045  R -0.044`
(`--print-hz`, default 20, limits the lines per second; `[lost N]` is shown if UDP packets are missing,
`[EMG stale …]` if the host's EMG is not streaming). Allow the port through that computer's firewall.

**On the EMG host** open *Settings > Application settings > Network streaming*, enter the other
computer's IP address (or name), the port and the same protocol, then *Save* / *Apply*.
Use *Network > Connect to remote computer* / *Disconnect*. The status bar shows the state.
- **TCP** (default, reliable, `TCP_NODELAY`): if the first connection fails an error is shown; once
  connected, a lost link is retried every *reconnect interval* until you disconnect.
- **UDP**: lowest latency, no connection; datagrams are sent even if nobody is listening.
- *Send rate* (default 100 Hz) applies immediately; protocol / host / port need a reconnect.
  EMG and wheel do not have to be connected, and streaming does not slow the EMG thread down.
- Direct cable between two PCs: give both a static IP in the same subnet (e.g. 192.168.10.1 / .2).

Message format (one JSON object per line; one per datagram for UDP):
`{"v":1,"seq":42,"t":1767000000.123,"angle_deg":-12.3,"angle_norm":-0.027,"arm_left":-0.05,"arm_right":0.0,"age_s":0.004}`
`angle_deg` is the EMG steering angle (±`lock_to_lock_deg`/2, *not* limited by the wheel's `max_target_deg`);
`age_s` is the time since the EMG last published (the angle is 0 when the EMG is disconnected).

## Files
`main.py` entry | `main_window.py` GUI | `settings_dialogs.py` config dialogs | `config.py` configs + JSON
`emg_worker.py` EMG thread | `band_power.py` streaming band metric | `steering.py` angle math + shared state
`delsys_client.py` AeroPy client | `sources.py` Delsys adapter + simulator | `wheel_controller.py` G29 thread (SDL2)
`network_stream.py` network sender thread | `client/steering_client.py` receiver for the other computer
`tests/test_core.py` tests (`python tests/test_core.py`)
