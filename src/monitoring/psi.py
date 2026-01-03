import numpy as np
import pandas as pd

def calculate_psi(expected, actual, bins=10):
    """
    Calculate Population Stability Index (PSI) for data drift.
    expected: baseline distribution
    actual: current distribution
    """
    # Bin the data
    breakpoints = np.linspace(0, 1, bins + 1)
    expected_bins = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_bins = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # Avoid division by zero
    expected_bins = np.where(expected_bins == 0, 0.0001, expected_bins)
    actual_bins = np.where(actual_bins == 0, 0.0001, actual_bins)

    psi = np.sum((actual_bins - expected_bins) * np.log(actual_bins / expected_bins))
    return psi