import pandas as pd
from .psi import calculate_psi
from .ks_monitor import calculate_ks

def generate_drift_report(baseline_data, current_data, baseline_preds, current_preds):
    """
    Generate data and model drift report.
    """
    report = {}

    # Data drift: PSI for each feature
    for col in baseline_data.columns:
        if col in current_data.columns:
            psi = calculate_psi(baseline_data[col], current_data[col])
            report[f'PSI_{col}'] = psi

    # Model drift: KS
    ks_baseline = calculate_ks(baseline_data['default'], baseline_preds)
    ks_current = calculate_ks(current_data['default'], current_preds)
    report['KS_baseline'] = ks_baseline
    report['KS_current'] = ks_current
    report['KS_drift'] = abs(ks_current - ks_baseline)

    # Save report
    pd.DataFrame([report]).to_csv('reports/drift_report.csv', index=False)
    return report