from src.preprocessing.pitch_family import is_cutter, map_pitch_family


def test_four_seam_and_sinker_are_fastball():
    assert map_pitch_family("FF") == "fastball"
    assert map_pitch_family("SI") == "fastball"
    assert map_pitch_family("FT") == "fastball"
    assert map_pitch_family("FA") == "fastball"


def test_cutter_is_fastball_family_but_flagged():
    assert map_pitch_family("FC") == "fastball"
    assert is_cutter("FC") is True
    assert is_cutter("FF") is False


def test_named_breaking_and_offspeed_codes_are_breaking():
    for code in ["SL", "ST", "CU", "KC", "CS", "CH", "FS", "FO", "SC", "KN"]:
        assert map_pitch_family(code) == "breaking"


def test_unlisted_codes_fall_back_to_breaking_not_silently_dropped():
    assert map_pitch_family("EP") == "breaking"
    assert map_pitch_family("SV") == "breaking"


def test_lowercase_input_is_normalized():
    assert map_pitch_family("ff") == "fastball"


def test_missing_pitch_type_maps_to_none():
    assert map_pitch_family(None) is None
    assert map_pitch_family(float("nan")) is None
