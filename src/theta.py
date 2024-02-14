from enum import StrEnum, auto

class Theta(StrEnum):
    """Enum for the two parameters we want to fit"""
    n_0 = auto()
    gamma = auto()
