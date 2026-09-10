from common.config import NUM_CHANNELS, SAMPLE_RATE, SEQ_LEN
from common.states import STATES, STATE_TO_INDEX

def test_config_values_present():
    assert NUM_CHANNELS == 8
    assert SAMPLE_RATE == 128
    assert SEQ_LEN == 5

def test_states_indexed_correctly():
    assert STATES == ["Focused", "Overloaded", "Confused", "Fatigued", "Engaged"]
    assert STATE_TO_INDEX["Focused"] == 0
    assert STATE_TO_INDEX["Engaged"] == 4
