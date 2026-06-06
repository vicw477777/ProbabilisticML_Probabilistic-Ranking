# Metropolis-Hastings MCMC algorithm for sampling skills in the probit rank model
# -gtc 20/09/2025
import numpy as np
from scipy.stats import norm
from tqdm import tqdm

def MH_sample(games, num_players, num_its):

    # pre-process data:
    # array of games for each player, X[i] = [(other_player, outcome), ...]
    X = [[] for _ in range(num_players)] 
    for a, (i,j) in enumerate(games):
        X[i].append((j, +1))  # player i beat player j
        X[j].append((i, -1))  # player j lost to payer i
    for i in range(num_players):
        X[i] = np.array(X[i])

    # array that will contain skill samples
    skill_samples = np.zeros((num_players, num_its))

    w = np.zeros(num_players)  # skill for each player

    proposal_sigma = 1.0 #add
    accepted_count = 0
    total_count = 0

    for itr in tqdm(range(num_its)):
        for i in range(num_players):
            j, outcome = X[i].T

            # current local log-prob 
            #lp1 = norm.logpdf(w[i]) + np.sum(norm.logcdf(outcome*(w[i]-w[j])))
            # 1. Calculate current local log-prob 
            # Prior: N(0,1) -> logpdf(w[i])
            # Likelihood: Product of Probit likelihoods -> sum of logcdf
            # Note: w[j] are the current skills of opponents (fixed for this step)
            diff = w[i] - w[j]
            log_prior = norm.logpdf(w[i])
            log_like = np.sum(norm.logcdf(outcome * diff))
            lp1 = log_prior + log_like

            # 2. Proposed new skill and log-prob (TODO filled)
            # Gaussian Random Walk Proposal
            w_prop = w[i] + np.random.normal(0, proposal_sigma)
            
            diff_prop = w_prop - w[j]
            log_prior_prop = norm.logpdf(w_prop)
            log_like_prop = np.sum(norm.logcdf(outcome * diff_prop))
            lp2 = log_prior_prop + log_like_prop

            # 3. Accept or reject move (TODO filled)
            # Metropolis ratio in log domain: log(alpha) = lp2 - lp1
            # (Proposal is symmetric, so q(x'|x) cancels out)
            if np.log(np.random.rand()) < (lp2 - lp1):
              w[i] = w_prop # Accept the move
              accepted_count += 1
            
            total_count += 1

        skill_samples[:, itr] = w
    
    print(f"MCMC Acceptance Rate: {accepted_count / total_count:.2%}")
    return skill_samples

