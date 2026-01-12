# utils/traffic_signal.py

def get_signal_state(frame_count):
    """
    TEMPORARY signal simulator
    Replace later with ML model
    """
    if frame_count % 300 < 150:
        return "RED"
    return "GREEN"
