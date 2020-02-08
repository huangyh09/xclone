import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

FullNormal = tfd.MultivariateNormalFullCovariance

from scipy.special import binom

def get_binom_coeff(AD, DP, max_val=700):
    """Get the binomial coefficients
    """
    # Since binom can't give log value, the maximum value in 64bit is 
    # around e**700, close to binom(1000, 500)

    binom_coeff = np.log(binom(DP.astype(np.int64), AD.astype(np.int64)))
    binom_coeff[binom_coeff > max_val] = max_val 
    binom_coeff = binom_coeff.astype(np.float32)

    return binom_coeff


class XCloneVB():
    """
    A Bayesian Binomial mixture model for CNV clonal lineage reconstruction.
    
    Parameters
    ----------
    Nb : int > 0
        Number of blocks, similar as genes.
    Nc : int > 0
        Number of cells.
    Nk : int > 0
        Number of clones in the cell population
    cnv_states : numpy.array (Ns, 2)
        The CNV states, paternal copy numbers and maternal copy numbers
    """
    def __init__(self, Nb, Nc, Nk, cnv_states=[[0, 2], [1, 1], [2, 0]]):
        # Initialize
        self.cnv_states = np.array(cnv_states)
        Ns = self.cnv_states.shape[0] # Number of CNV states

        self.Ns = Ns
        self.Nb = Nb
        self.Nc = Nc
        self.Nk = Nk
        
        # Variational distribution variables for cell assignment
        self.cell_logit = tf.Variable(tf.random.uniform((Nc, Nk), -0.1, 0.1))
        
        # Variational distribution variables for CNV states
        self.CNV_logit  = tf.Variable(tf.random.uniform((Nb, Nk, Ns), -1, 1))
        
        # Variational distribution variables for allelic ratio
        self.theta_s1 = tf.Variable(tf.random.uniform((Ns, ), 1, 2))
        self.theta_s2 = tf.Variable(tf.random.uniform((Ns, ), 1, 2))
        
        # Variational distribution variables for depth ratio
        # self.RDR_mu = tf.Variable(tf.random.uniform((self.Ns), -2., 2.))
        self.RDR_mu = tf.random.uniform((self.Ns, ), -2., 2.)
        self.RDR_cov = None

        self.set_prior()
        
    def set_GP_kernal(self, l1=0.1, l2=1.0):
        """Set the hyperparameters for GP prior on depth ratio"""
        RDR_cov = np.zeros((self.Ns, self.Ns), dtype=np.float32)
        for i in range(self.Ns):
            for j in range(self.Ns):
                diff = self.cnv_states[i, :] - self.cnv_states[j, :]
                RDR_cov[i, j] = l1 * np.exp(-0.5 * np.sum(diff**2) / l2)
        self.RDR_cov = RDR_cov
        
        
    def set_prior(self, theta_prior=None, gamma_prior=None, Y_prior=None, 
                  Z_prior=None):
        """Set prior ditributions
        """
        # Prior distributions for the allelic ratio
        if theta_prior is None:
            self.theta_prior = tfd.Beta(self.cnv_states[:, 0] + 0.01, 
                                        self.cnv_states[:, 1] + 0.01)
        else:
            self.theta_prior = theta_prior
            
        # Prior distributions for the depth ratio
        if gamma_prior is None:
            if self.RDR_cov is None:
                self.set_GP_kernal()
            self.gamma_prior = FullNormal(loc = self.cnv_states.sum(axis=1), 
                                          covariance_matrix = self.RDR_cov)
        else:
            self.gamma_prior = gamma_prior
            
        # Prior distributions for CNV state weights
        if Y_prior is None:
            self.Y_prior = tfd.Multinomial(total_count=1,
                        probs=tf.ones((self.Nb, self.Nk, self.Ns)) / self.Ns)
        else:
            self.Y_prior = Y_prior
            
        # Prior distributions for cell assignment weights
        if Z_prior is None:
            self.Z_prior = tfd.Multinomial(total_count=1,
                        probs=tf.ones((self.Nc, self.Nk)) / self.Nk)
        else:
            self.Z_prior = Z_prior
        

    @property
    def theta(self):
        """Variational posterior for ASE ratio"""
        # return tfd.Beta(tf.math.exp(self.theta_s1), tf.math.exp(self.theta_s2))
        return tfd.Beta(self.theta_s1, self.theta_s2)

    @property
    def gamma(self):
        """Variational posterior for distribution mean"""
        return FullNormal(self.RDR_mu, self.RDR_cov)

    @property
    def Z(self):
        """Variational posterior for cell assignment"""
        return tfd.Multinomial(total_count=1, logits=self.cell_logit)
    
    @property
    def Y(self):
        """Variational posterior for CNV state"""
        return tfd.Multinomial(total_count=1, logits=self.CNV_logit)
    
    
    @property
    def KLsum(self):
        """Sum of KL divergences between posteriors and priors"""
        kl_theta = tf.reduce_sum(tfd.kl_divergence(self.theta, self.theta_prior))
        kl_gamma = tf.reduce_sum(tfd.kl_divergence(self.gamma, self.gamma_prior))
        kl_Y = tf.reduce_sum(self.Y.mean() * 
                             tf.math.log(self.Y.mean() / self.Y_prior.mean()))
        kl_Z = tf.reduce_sum(self.Z.mean() * 
                             tf.math.log(self.Z.mean() / self.Z_prior.mean()))
        
        return kl_theta + kl_Y + kl_Z # + kl_gamma
    
        
    def logLik(self, AD, DP, binom_coeff, sampling=False, size=10):
        """
        Compute marginalised log likelihood for full data sets
        
        Parameters
        ----------
        AD : tf.Tensor, (n_blocks, n_cells)
            Counts for alternative allele for each block in each cell
        DP : tf.Tensor, (n_blocks, n_cells)
            Counts for total depth (two alleles) for each block in each cell
        sampling : bool
            Whether to sample from the variational posterior
            distributions (if True, the default), or just use the
            mean of the variational distributions (if False).
            
        Returns
        -------
        log_likelihood : tf.Tensor
            Log likelihood for all data, e.g., E_p(log_lik)
        """
        # Variational distributions (Nb, Nc, Nk, Ns)
        AD = tf.reshape(AD, (self.Nb, self.Nc, 1)) # depth of A allele
        DP = tf.reshape(DP, (self.Nb, self.Nc, 1)) # depth of A & B allele
        BD = DP - AD
        binom_coeff = tf.reshape(binom_coeff, (self.Nb, self.Nc, 1))

        ## marginalise element CNV state probability
        # _prob = tf.tensordot(self.Z.mean(), self.Y.mean(), axes=[1, 1])
        # _prob = tf.transpose(_prob, [0, 2, 1]) #(Nb, Ns, Nc) --> (Nb, Nc, Ns)
        _Z = tf.reshape(self.Z.mean(), (1, self.Nc, self.Nk, 1))
        _Y = tf.reshape(self.Y.mean(), (self.Nb, 1, self.Nk, self.Ns))
        _prob = tf.reduce_sum(_Z * _Y, axis=2)

        ## marginalise allelic ratio mess
        _theta_ss = self.theta_s1 + self.theta_s2
        _digamma1 = tf.reshape(tf.math.digamma(self.theta_s1), (1, 1, self.Ns))
        _digamma2 = tf.reshape(tf.math.digamma(self.theta_s2), (1, 1, self.Ns))
        _digamma3 = tf.reshape(tf.math.digamma(_theta_ss), (1, 1, self.Ns))
        
        _log_lik_ASE = _prob * (AD * _digamma1 + 
                                BD * _digamma2 - 
                                DP * _digamma3 + binom_coeff)
        
        # _theta1 = tf.reshape(tf.math.log(self.theta.mean()), (1, 1, self.Ns))
        # _theta2 = tf.reshape(tf.math.log(1 - self.theta.mean()), (1, 1, self.Ns))
        # _log_lik_ASE = _prob * (binom_coeff + AD * _theta1 + BD * _theta2)

        ## marginalise depth ratio loglikelihood
        _log_lik_RDR = tf.zeros([1, ])

        ## summarised mass logLik 
        return tf.reduce_sum(_log_lik_ASE) + tf.reduce_sum(_log_lik_RDR)
    
    def fit(self, AD, DP, num_steps=200, 
            optimizer=None, learn_rate=0.05, **kwargs):
        """Fit the model's parameters"""
        if optimizer is None:
            optimizer = tf.optimizers.Adam(learning_rate=learn_rate)
            
        binom_coeff = get_binom_coeff(AD, DP)
            
        loss_fn = lambda: (self.KLsum - 
                           self.logLik(AD, DP, binom_coeff, **kwargs))
        
        losses = tfp.math.minimize(loss_fn, 
                                   num_steps=num_steps, 
                                   optimizer=optimizer)
        return losses
