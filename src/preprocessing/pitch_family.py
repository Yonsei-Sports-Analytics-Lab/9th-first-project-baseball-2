"""Map Statcast pitch_type codes to a binary fastball/breaking family.

Statcast's own pitch_type taxonomy has ~15 codes and grows over time
(e.g. "ST" and "SV" were added after 2022). The project only needs two
broad families, so instead of enumerating every non-fastball code, only
the fastball set is enumerated and everything else (named or not) is
"breaking" -- this keeps the split exhaustive with no silent "other"
bucket if Statcast introduces a new code.
"""

FASTBALL_CODES = {"FF", "SI", "FT", "FA"}
CUTTER_CODE = "FC"


def _is_missing(pitch_type: str | float | None) -> bool:
    return pitch_type is None or (isinstance(pitch_type, float) and pitch_type != pitch_type)


def is_cutter(pitch_type: str | float | None) -> bool:
    if _is_missing(pitch_type):
        return False
    return pitch_type.upper() == CUTTER_CODE


def map_pitch_family(pitch_type: str | float | None) -> str | None:
    if _is_missing(pitch_type):
        return None
    code = pitch_type.upper()
    if code in FASTBALL_CODES or code == CUTTER_CODE:
        return "fastball"
    return "breaking"
