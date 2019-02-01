'''This module implements basic probability distributions 
(or density functions) used by the GSM model as prior/posterior.

'''

__all__ = ['Distribution']

import abc
from dataclasses import dataclass
import math
import torch


# Error raised when the user is providing parameters object with a
#  missing attribute. 
class MissingParameterAttribute(Exception): pass

# Error raised when a "Distribution" subclass does not define the 
# parameters of the distribution.
class UndefinedParameters(Exception): pass


# Check if a parameter object has a specific attribute.
def _check_params_have_attr(params, attrname):
    if not hasattr(params, attrname):
        raise MissingParameterAttribute(
                    f'Parameters have no "{attrname}" attribute')


class Distribution(torch.nn.Module, metaclass=abc.ABCMeta):
    'Abstract base class for distribution/pdf.'

    # Sucbclasses need to define the parameters of the distribution
    # in a dictionary stored in a class variable named 
    # "_std_params_def". For example:
    #_std_params_def = {
    #
    #      +---------------------------- Parameter's name which will be 
    #      |                             a read-only attribute of the 
    #      |                             subclass.
    #      |
    #      |             +-------------- Documentation of the parameter
    #      |             |               which will be converted into
    #      |             |               the docstring of the attribute.
    #      v             v
    #    'mean': 'Mean parameter.', 
    #    'var': 'Variance parameter.'
    #}

    def __init_subclass__(cls):
        if not hasattr(cls, '_std_params_def'):
            raise UndefinedParameters('Parameters of the distribution are ' \
                                      'undefined. You need to specify the ' \
                                      'field "_std_params_def" in your ' \
                                      'class definition.')


    def __init__(self, params):
        super().__init__()
        self.params = params
        for param_name, param_doc in self._std_params_def.items():
            _check_params_have_attr(params, param_name)
            getter = lambda self, name=param_name: getattr(self.params, name)
            setattr(self.__class__, param_name, property(getter, doc=param_doc))

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            for param_name in self._std_params_def:
                param0 = getattr(self, param_name)
                param1 = getattr(other, param_name)
                if param0.shape != param1.shape: 
                    # Check if the dimension matches.
                    return False
                elif not torch.allclose(param0, param1):
                    # Check if the parameters are the same.
                    return False
            else:
                return True
        return NotImplemented
    
    def __hash__(self):
        return hash(id(self))
    #    params_value = (getattr(self, param_name) 
    #                    for param_name in self._std_params_def)
    #    return hash(params_value)


class NormalDiagonalCovariance(Distribution):
    'Normal pdf with diagonal covariance matrix.'

    _std_params_def = {
        'mean': 'Mean parameter.',
        'diag_cov': 'Diagonal of the covariance matrix.',
    }

    @property
    def dim(self):
        'Dimension of the support.'
        return len(self.mean)

    def expected_sufficient_stats(self, store_full_cov=False):
        '''Expected sufficient statistics given the current 
        parameterization. If `store_full_cov` is `True` the returned
        tensor will be padded with 0 as if the full covariance was 
        stored.

        Note: The dimension indicated below are only valid for the 
        case when `store_full_cov` is `False`.

        For the random variable X (D-dimensional vector) the sufficient statistics of
        the Normal with diagonal covariance matrix are given: 

                                    +-- Dimension
                                    |
        stats = (                   v  
            x,                  <-- D
            -.5 * x^2,          <-- D
            -.5,                <-- 1
            .5                  <-- 1
        )

        The constant terms (the two last dimensions) are added to 
        match the parameterization of the conjugate prior (Normal-Gamma
        or Normal-Wishart).

        For the standard parameters (m=mean, s=diag_cov) 
        expectation of the sufficient statistics is given by:

                                                    +-- Dimension 
                                                    | 
        exp_stats = (                               v
            m,                                  <-- D   
            -.5 * (s + m^2),                    <-- D
            -.5,                                <-- 1
            .5                                  <-- 1
        )

        '''
        stats2 = -.5 * (self.params.diag_cov + self.params.mean**2)
        dtype, device = self.params.mean.dtype, self.params.mean.device
        return torch.cat([
            self.params.mean,
            stats2 if not store_full_cov else stats2.diag().reshape(-1),
            torch.tensor(-.5, dtype=dtype, device=device).reshape(1),
            torch.tensor(.5, dtype=dtype, device=device).reshape(1)
        ])
    
    # Log-likelihood of the data given the current parameters. 
    def forward(self, X):
        quad = (X - self.mean)**2
        diag_prec = 1. / self.diag_cov
        dim = X.shape[-1]
        bmeasure = -.5 * dim * math.log(2 * math.pi)
        lnorm = -.5 * self.diag_cov.log().sum()
        return -.5 * torch.sum(quad * diag_prec, dim=-1) + lnorm + bmeasure

    def sample(self, nsamples):
        '''Draw random values using the "reparameterization trick".
        
        Parameters:
            nsamples (int): number of samples to draw.
        
        Returns:
            ``torch.Tensor[nsamples, D]``: A matrix of `nsamples` of 
            `D` dimensional values.
            
        '''
        return self.mean + self.diag_cov.sqrt() * torch.randn(nsamples, self.dim)

    def kl_div(self, other):
        '''KL divergence between two ``NormalDiagonalCovariance``
        
        See the wikipedia definition:
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback%E2%80%93Leibler_divergence
        
        Parameters:
            other (``NormalDiagonalCovariance``): pdf to compare.
            
        Returns:
            float: value of the KL divergence.
            
        '''
        if self.__class__ is not other.__class__ or self.dim != other.dim:
            return NotImplemented
        mean_diff = other.mean - self.mean
        diag_cov0, diag_cov1 = self.diag_cov, other.diag_cov
        log_det_cov0, log_det_cov1 = diag_cov0.log().sum(), \
                                     diag_cov1.log().sum()
        tr_prec_cov = (diag_cov0 / diag_cov1).sum()
        
        return .5 * (tr_prec_cov + (1. / diag_cov1) @ (mean_diff ** 2) \
               - self.dim + log_det_cov1 - log_det_cov0)


########################################################################
# Normal-Wishart pdf.

@dataclass(init=False, eq=False, unsafe_hash=True)
class NormalWishartStdParams(torch.nn.Module):
    'Standard parameterization of the Normal-Wishart pdf.'

    mean: torch.Tensor
    scale: torch.Tensor
    scale_matrix: torch.Tensor
    dof: torch.Tensor

    def __init__(self, mean, scale, scale_matrix, dof):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('scale', scale)
        self.register_buffer('scale_matrix', scale_matrix) 
        self.register_buffer('dof', dof)
    
    @classmethod
    def from_natural_parameters(cls, natural_params):
        # First we recover the dimension of the mean parameters (D). 
        # Since the dimension of the natural parameters of the 
        # Normal-Wishart is: 
        #       l = len(natural_params) 
        #       D^2 + D + 2 = l 
        # we can find D by looking for the positive root of the above 
        # polynomial which is given by:
        #       D = .5 * (-1 + sqrt(1 + 4 * l))   
        dim = int(.5 * (-1 + math.sqrt(1 + 4 * len(natural_params))))

        np1 = natural_params[:dim]
        np2 = natural_params[dim: dim * (dim + 1)]
        np3 = natural_params[-2]
        np4 = natural_params[-1]
        scale = -2 * np3
        mean = np1 / scale
        scale_matrix = (-2 * np2.reshape(dim, dim) \
                        - scale * torch.ger(mean, mean)).inverse()
        dof = 2 * np4 + dim

        return cls(mean, scale, scale_matrix, dof)

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return torch.allclose(self.mean, other.mean) \
                and torch.allclose(self.scale, other.scale) \
                and torch.allclose(self.scale_matrix, other.scale_matrix) \
                and torch.allclose(self.dof, other.dof)
        return NotImplemented

class NormalWishart(Distribution):
    'Normal-Wishart pdf.'

    _std_params_def = {
        'mean': 'Mean of the Normal pdf.',
        'scale': 'scale of the precision of  the Normal pdf.',
        'scale_matrix': 'Scale matrix of the Wishart pdf.',
        'dof': 'Number degree of freedom of the Wishart pdf.',
    }

    @property
    def dim(self):
        'Dimension of the support of the Normal pdf.'
        return len(self.mean)
    
    def natural_parameters(self):
        '''Natural form of the current parameterization. For the 
        standard parameters (m=mean, k=scale, W=W, v=dof) the natural
        parameterization is given by:

                                            +-- Dimension 
                                            | 
        nparams = (                         v
            k * m      ,                <-- D      
            -.5 * W^{-1} + k * m * m^T, <-- D^2
            -.5 * k,                    <-- 1
            .5 * (v - D)                <-- 1
        )

        Note: "D" is the dimension of "m"

        Returns:
            ``torch.Tensor[D + D^2 + 2]`` 
        
        '''
        return torch.cat([
            self.scale * self.mean,
            -.5 * (self.scale_matrix.inverse() \
                + self.scale * torch.ger(self.mean, self.mean)).reshape(-1),
            -.5 * self.scale.reshape(1),
            .5 * (self.dof - self.dim).reshape(1)
        ])

    def expected_sufficient_stats(self):
        '''Expected sufficient statistics given the current 
        parameterization. 

        For the random variable mu (vector), S (positive definite 
        matrix) the sufficient statistics of the Normal-Wishart are 
        given by: 

                                    +-- Dimension
                                    |
        stats = (                   v  
            S * mu,             <-- D
            S,                  <-- D^2
            tr(S * mu * mu^T),  <-- 1
            ln |S|              <-- 1
        )

        For the standard parameters (m=mean, k=scale, W=W, v=dof) 
        expecation of the sufficient statistics is given by:

                                                    +-- Dimension 
                                                    | 
        exp_stats = (                               v
            v * W * m,                          <-- D   
            v * W,                              <-- D^2
            (D/k) + tr(v * W * m * m^T),        <-- 1
            ( \sum_i psi(.5 * (v + 1 - i)) ) \  
             + D * ln 2 + ln |W|                <-- 1
        )

        Note: "tr" is the trace operator, "D" is the dimenion of "m" 
            and "psi" is the "digamma" function.

        '''
        idxs = torch.arange(0, self.dim, dtype=self.mean.dtype,
                            device=self.mean.device)
        L = torch.cholesky(self.scale_matrix, upper=False)
        logdet = torch.log(L.diag()).sum()
        mean_quad = torch.ger(self.mean, self.mean)
        exp_prec = self.dof * self.scale_matrix
        return torch.cat([
           exp_prec @ self.mean,
            exp_prec.reshape(-1),
            ((self.dim / self.scale) \
                + (exp_prec @ mean_quad).trace()).reshape(1),
            (torch.digamma(.5 * (self.dof + 1 - idxs)).sum() \
                + self.dim * math.log(2) + logdet).reshape(1)
        ])

    def log_norm(self):
        'Log-normalization constant given the current parameterization.'  
        idxs = torch.arange(0, self.dim, dtype=self.mean.dtype,
                            device=self.mean.device)
        L = torch.cholesky(self.scale_matrix, upper=False)
        logdet = 2 * torch.log(L.diag()).sum()
        return .5 * self.dof * logdet + .5 * self.dof * self.dim * math.log(2) \
               + .25 * self.dim * (self.dim - 1) * math.log(math.pi) \
               + torch.lgamma(.5 * (self.dof + 1 - idxs)).sum() \
               + .5 * self.dim * torch.log(self.scale) \
               + .5 * self.dim * math.log(2 * math.pi)

    def kl_div(self, other):
        'KL-divergence between two Normal-Wishart densities.'
        nparams0 = self.natural_parameters()
        nparams1 = other.natural_parameters()
        exp_stats = self.expected_sufficient_stats()
        lnorm0 = self.log_norm()
        lnorm1 = other.log_norm()
        return lnorm1 - lnorm0 - exp_stats @ (nparams1 - nparams0) 
