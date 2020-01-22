# Definition of read depth ratio

import vireoSNP
import numpy as np
from scipy.special import logsumexp, digamma, betaln

def Poisson_PDF(X, phi, log_out=True):
    """
    Probability density function for Poisson distribution

    X: count matrix
    phi: parameters for each element
    """
    return None


def RDR_logLik(DP, ID_prob, CNV_prob, CNV_link, cell_factors, gene_factors):
    """
    Log likelihood for read depth ratio

    Parameters
    ----------
    DP: gene-by-cell array of expression count
    ID_prob: cell-by-clone array of assignment probability
    CNV_prob: gene-by-clone-by-n_CNV_states array for probability that a gene 
              in a cell is in a CNV state
    CNV_link: the link function that a CNV states for the expression levels
    """
    n_state = len(CNV_link) # number of CNV states

    # n_gene-by-n_cell-by-n_state
    _cnv_prob_log = np.tensordot(CNV_prob, ID_prob, axes=[1, 1])

    # n_gene-by-n_cell-by-n_state
    _log_like_arr = np.zeros((_cnv_prob_log.shape.append(n_state)))
    for i in range(n_state):
        phi = CNV_link[i] * np.tensordot(gene_factors, cell_factors)
        _log_like_arr[:, :, i] = Poisson_PDF(DP, phi)

    _log_like = logsumexp(_log_like_arr, axis=2)
    return _log_like

    

