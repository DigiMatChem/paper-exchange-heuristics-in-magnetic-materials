import numpy as np
from scipy.stats import distributions


def compute_ks_weighted(afm_occus, fm_occus, alternative="two-sided"):
    """Compute KS test of samples with non-integer occurrences with sample weights
    based on stackoverflow answers by Luca Jokull and Tim Williams.
    https://stackoverflow.com/questions/40044375/how-to-calculate-the-kolmogorov-
    smirnov-statistic-between-two-weighted-samples/55664242#55664242
    Main difference: sample sizes n1 and n2 do *not* correspond to len(data1) and len(data2),
    but to sum of occurrences.
    :return d, p
    """
    data1 = np.array([ls[1] for ls in afm_occus])
    wei1 = np.array([ls[2] for ls in afm_occus])
    data2 = np.array([ls[1] for ls in fm_occus])
    wei2 = np.array([ls[2] for ls in fm_occus])

    ix1 = np.argsort(data1)
    ix2 = np.argsort(data2)
    data1 = data1[ix1]
    data2 = data2[ix2]
    wei1 = wei1[ix1]
    wei2 = wei2[ix2]
    data = np.concatenate([data1, data2])
    cwei1 = np.hstack([0, np.cumsum(wei1) / sum(wei1)])
    cwei2 = np.hstack([0, np.cumsum(wei2) / sum(wei2)])
    cdf1we = cwei1[np.searchsorted(data1, data, side='right')]
    cdf2we = cwei2[np.searchsorted(data2, data, side='right')]
    d = np.max(np.abs(cdf1we - cdf2we))
    # calculate p-value
    # n1 = data1.shape[0]  # original implementation
    # n2 = data2.shape[0]
    n1 = wei1.sum()
    n2 = wei2.sum()
    m, n = sorted([float(n1), float(n2)], reverse=True)
    en = m * n / (m + n)
    if alternative == 'two-sided':
        prob = distributions.kstwo.sf(d, np.round(en))
    else:
        z = np.sqrt(en) * d
        # Use Hodges' suggested approximation Eqn 5.3
        # Requires m to be the larger of (n1, n2)
        expt = -2 * z ** 2 - 2 * z * (m + 2 * n) / np.sqrt(m * n * (m + n)) / 3.0
        prob = np.exp(expt)
    return d, prob
