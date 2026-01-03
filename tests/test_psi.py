import pytest
import numpy as np
from src.monitoring.psi import calculate_psi

def test_calculate_psi():
    expected = np.random.normal(0, 1, 1000)
    actual = np.random.normal(0, 1, 1000)
    psi = calculate_psi(expected, actual)
    assert isinstance(psi, float)
    assert psi >= 0