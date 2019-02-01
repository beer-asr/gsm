'''This module implements a Bayesian Generalized Subspace Model.

The generative model of the GSM can be summarized as the following 
sequence:

      prior 
        |
        | sample
        v
    latent variable (h)
        : 
        :
        v
    likelihood  <-- params = f( W^T h + b )
        |
        |  sample
        v
    observation

'''

__all__ = ['create_latent_posteriors', 'GSM']

from collections import namedtuple
from dataclasses import dataclass
import math
import torch
from .dists import NormalDiagonalCovariance
from .dists import NormalWishart
from .dists import NormalWishartStdParams


########################################################################
# Affine Transform:  W^T h + b
# "W" and "b" have a Normal (with diagonal covariance matrix) prior.


# Parametererization of the Normal distribution. We use the log variance
# instead of the variance so we don't have any constraints during the 
# S.G.D. 
class _MeanLogDiagCov(torch.nn.Module):

    def __init__(self, mean, log_diag_cov):
        super().__init__()
        if mean.requires_grad:
            self.register_parameter('mean', torch.nn.Parameter(mean))
        else:
            self.register_buffer('mean', mean)
        
        if log_diag_cov.requires_grad:
            self.register_parameter('log_diag_cov', 
                                    torch.nn.Parameter(log_diag_cov))
        else:
            self.register_buffer('log_diag_cov', log_diag_cov)

    @property
    def diag_cov(self):
        return self.log_diag_cov.exp() 

    def __hash__(self):
        return hash(id(self))
    
    # For some unknown reason it is necessary to make the 
    # respresentation unique per object otherwise pytorch is not able to
    # properly find the parameters ???? >:-( ????
    def __repr__(self):
        return str(id(self))


class AffineTransform(torch.nn.Module):
    '''Affine Transformation: y = W^T x + b.

    Attributes:
        in_dim (int): input dimension
        out_dim (int): output dimension

    '''

    def __init__(self, in_dim, out_dim, prefix=''):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Bias prior/posterior.
        self.prior_bias = NormalDiagonalCovariance(
            _MeanLogDiagCov(
                mean=torch.zeros(out_dim, requires_grad=False), 
                log_diag_cov=torch.zeros(out_dim, requires_grad=False),
            )
        )
        self.posterior_bias = NormalDiagonalCovariance(
            _MeanLogDiagCov(
                mean=torch.zeros(out_dim, requires_grad=True), 
                log_diag_cov=torch.zeros(out_dim, requires_grad=True),
            )
        )

        # Weights prior/posterior.
        self.prior_weights = NormalDiagonalCovariance(
            _MeanLogDiagCov(
                mean=torch.zeros(in_dim * out_dim, requires_grad=False), 
                log_diag_cov=torch.zeros(in_dim * out_dim, requires_grad=False),
            )
        )
        self.posterior_weights = NormalDiagonalCovariance(
            _MeanLogDiagCov(
                mean=torch.zeros(in_dim * out_dim, requires_grad=True), 
                log_diag_cov=torch.zeros(in_dim * out_dim, requires_grad=True),
            )
        )

    def forward(self, X, nsamples=5):
        s_b = self.posterior_bias.sample(nsamples)
        s_W = self.posterior_weights.sample(nsamples).reshape(-1, self.in_dim, 
                                                              self.out_dim)
        res = torch.matmul(X[None], s_W) + s_b[:, None, :]

        # For coherence with other components we reorganize the 3d 
        # tensor to have:
        #   - 1st dimension: number of input data points (i.e. len(X))
        #   - 2nd dimension: number samples to estimate the expectation
        #   - 3rd dimension: dimension of the latent space.
        return res.permute(1, 0, 2)

    def kl_div_posterior_prior(self):
        'KL-divergence between the current posterior and the prior.'
        return self.posterior_weights.kl_div(self.prior_weights) \
               + self.posterior_bias.kl_div(self.prior_bias)


########################################################################
# Variational Latent Posterior, a Normal (with diagonal covariance 
# matrix) for each latent variable. We use the same parameterization 
# as the AffineTransform's prior.

def create_latent_posteriors(n, dim):
    '''Create a list of latent posteriors initialized to be the standard 
    Normal.

    Parameters:
        n (int): Number of posterior distributions to create.
        dim (int): Dimension of the support of the posterior.
    
    Returns:
        list of ``NormalDiagonalCovariance``

    '''
    return [
        NormalDiagonalCovariance(
            _MeanLogDiagCov(
                mean=torch.zeros(dim, requires_grad=True), 
                log_diag_cov=torch.zeros(dim, requires_grad=True),
            )
        ) for i in range(n)
    ]

def sample_from_latent_posteriors(posteriors, nsamples, compute_llh=False):
    '''Samples from a list of latent posterior distributions

    Parameters:
        posteriors (list of ``NormalDiagonalCovariance``): The 
            posteriors to sample from.
        nsamples (int): Number of samples to draw per posterior.
        compute_llh (boolean): If true compute the log-likelihood of the
            sample w.r.t. the sampling distribution.

    Returns:
        ``torch.Tensor[len(posteriors), nsamples, dim]``: Samples.
        ``torch.Tensor[len(posteriors), nsamples]``: list of 
            log-likelihood (if "compute_llh=True")

    '''
    samples = torch.cat([post.sample(nsamples)[None] for post in posteriors])
    if compute_llh:
        llh = torch.cat([post(s)[None] for post, s in zip(posteriors, samples)])
        return samples, llh
    return samples


########################################################################
# GSM latent variable prior is a multivariate Normal pdf with a 
# Normal-Wishart hyper-prior over the mean and precision matrix 
# parameters.
 
class NormalPrior(torch.nn.Module):
    '''Multivariate Normal model serving as the prior for the GSM's 
    latent variable.

    Attributes:
        prior: Normal-Wishart prior
        posterior: Normal-Wishart posterior
        
    '''
    
    def __init__(self, prior, posterior):
        super().__init__()
        self.prior = prior
        self.posterior = posterior

    def _sufficient_stats(self, X):
        return torch.cat([
            X, 
            -.5 * (X[:, :, None] * X[:, None, :]).reshape(len(X), -1),
            -.5 * torch.ones(len(X), 1, dtype=X.dtype, device=X.device),
            .5 * torch.ones(len(X), 1, dtype=X.dtype, device=X.device),
        ], dim=-1) 

    # Expected log-likelihood of the model w.r.t. to the posterior over 
    # the parameters of the model.
    def forward(self, X):
        nparams = self.posterior.expected_sufficient_stats()
        s_stats = self._sufficient_stats(X)
        return s_stats @ nparams - .5 * X.shape[-1] * math.log(2 * math.pi)

    def kl_div_posterior_prior(self):
        'KL-divergence between the current posterior and the prior.'
        return self.posterior.kl_div(self.prior) 
        

########################################################################
# GSM implementation.

# Build a default prior using the standard parameterization of the 
# Normal-Wishart.
def _default_prior(in_dim):
    params = NormalWishartStdParams(
        mean=torch.zeros(in_dim, requires_grad=False),
        scale=torch.tensor(1., requires_grad=False),
        scale_matrix=torch.eye(in_dim, requires_grad=False),
        dof=torch.tensor(float(in_dim), requires_grad=False)
    )
    hyper_prior = NormalWishart(params)
    params = NormalWishartStdParams(
        mean=torch.zeros(in_dim, requires_grad=False),
        scale=torch.tensor(1., requires_grad=False),
        scale_matrix=torch.eye(in_dim, requires_grad=False),
        dof=torch.tensor(float(in_dim), requires_grad=False)
    )
    hyper_posterior = NormalWishart(params)
    return NormalPrior(hyper_prior, hyper_posterior)
    

class GSM(torch.nn.Module):
    'Generalized Subspace Model.'

    def __init__(self, in_dim, out_dim, llh_func, prior=None):
        super().__init__()
        self.llh_func = llh_func
        self.trans = AffineTransform(in_dim, out_dim)
        self.prior = prior if prior is not None else _default_prior(in_dim)

    def forward(self, X, latent_posts, nsamples_latents=1, nsamples_params=1):
        s_h, s_llh = sample_from_latent_posteriors(latent_posts, 
                                                   nsamples_latents, 
                                                   compute_llh=True)
        s_entropy = -s_llh.mean(dim=-1)
        s_h = s_h.reshape(-1, s_h.shape[-1])
        s_xentropy = -self.prior(s_h).reshape(len(X), 
                                              max(1, nsamples_latents)).mean(dim=-1)
        params = self.trans(s_h, nsamples=nsamples_params)
        params = params.reshape(len(X), max(nsamples_latents * nsamples_params, 1), -1)
        s_kl_div = s_xentropy - s_entropy
        return self.llh_func(params, X).mean(dim=-1), s_kl_div

    def sample_params(self, latent_posts, nsamples_latents=1, nsamples_params=1):
        s_h = sample_from_latent_posteriors(latent_posts, nsamples_latents)
        s_h = s_h.reshape(-1, s_h.shape[-1])
        samples = self.trans(s_h, nsamples=nsamples_params)
        return samples.reshape(len(latent_posts), 
                               max(nsamples_latents * nsamples_params, 1), -1)

    def kl_div_posterior_prior(self):
        return self.trans.kl_div_posterior_prior() \
               + self.prior.kl_div_posterior_prior()

    def update_prior(self, latents):
        'Update the prior to maximize the ELBO.'
        latents_exp_stats = torch.cat([
            latent.expected_sufficient_stats(store_full_cov=True)[None]
            for latent in latents
        ])
        new_nparams = self.prior.prior.natural_parameters() \
            + latents_exp_stats.sum(dim=0).detach()
        self.prior.posterior.params = \
            NormalWishartStdParams.from_natural_parameters(new_nparams)
