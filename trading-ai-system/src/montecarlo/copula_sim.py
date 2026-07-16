"""Empirical-marginal Gaussian-copula simulation with a latent market factor."""
import numpy as np
from scipy.special import ndtr


def simulate(returns,n,horizon,rng,market_corr=.4):
 history=np.sort(np.asarray(returns.dropna(),dtype=float));
 if not len(history):return np.zeros(n)
 market=rng.normal(size=(n,horizon));idiosyncratic=rng.normal(size=(n,horizon));latent=float(market_corr)*market+np.sqrt(max(1-float(market_corr)**2,0))*idiosyncratic;uniform=np.clip(ndtr(latent),1e-6,1-1e-6);indices=np.minimum((uniform*len(history)).astype(int),len(history)-1);draws=history[indices];return np.prod(1+draws,axis=1)-1
