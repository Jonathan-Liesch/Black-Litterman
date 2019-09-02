import numpy as np

def get_mkt_weights(stock_price_vec, shares_outstanding_vec):
    MCs = stock_price_vec * shares_outstanding_vec
    total_MC = np.sum(MCs)
    return MCs/total_MC

def get_Pi(lam, sigma, mkt_weights):
    return lam * np.matmul(sigma, mkt_weights)

def get_new_return_vec(stock_price_vec, shares_outstanding_vec, Sigma, lam, tau, P, Q, Omega):
    assert tau > 0, "tau <= 0"

    w_mkt = get_mkt_weights(stock_price_vec, shares_outstanding_vec)
    Pi = get_Pi(lam, Sigma, w_mkt)

    inv_tSigma = np.linalg.inv(tau*Sigma)
    
    if Omega.ndim == 1:
        inv_Omega = 1/Omega
        inv_left = inv_tSigma + np.matmul(P, inv_Omega * P) # should be np.matmul(inv_Omega, P)
        left = np.linalg.inv(inv_left)
        right = np.matmul(inv_tSigma, Pi) + P*inv_Omega*Q # should be np.matmul(P, np.matmul(inv_Omega, Q))
        return np.matmul(left, right)
    else:
        inv_Omega = np.linalg.inv(Omega)
        inv_left = inv_tSigma + np.matmul(P,np.matmul(inv_Omega, P))
        left = np.linalg.inv(inv_left)
        right = np.matmul(inv_tSigma, Pi) + np.matmul(P, np.matmul(inv_Omega, Q))
        return np.matmul(left, right)

def get_Markowitz_Portfolio(return_vec, Sigma, desired_return):
    """
    min wSw st wR = r, w1 = 1
    """
    y = np.array([1, desired_return])
    one = np.array([1] * len(return_vec))
    inv_Sigma = np.linalg.inv(Sigma)

    a = np.matmul(one, np.matmul(inv_Sigma, one))
    b = np.matmul(return_vec, np.matmul(inv_Sigma, one))
    c = np.matmul(one, np.matmul(inv_Sigma, return_vec))
    d = np.matmul(return_vec, np.matmul(inv_Sigma, return_vec))

    A = np.array([[a,b],[c,d]])
    inv_A = np.linalg.inv(A)

    lagrangian_multipliers = np.matmul(inv_A, y)

    weights = lagrangian_multipliers[0] * np.matmul(inv_Sigma, one) + \
        lagrangian_multipliers[1] * np.matmul(inv_Sigma, return_vec)

    return weights

def get_Markowitz_Portfolio_Vec(return_vec, Sigma, desired_return):
    weights = get_Markowitz_Portfolio(return_vec, Sigma, desired_return)
    mean = np.matmul(return_vec, weights)
    std = np.matmul(weights, np.matmul(Sigma, weights))**0.5
    return std, mean

###############
#    Main
###############

sp_vec = np.array([2, 2])
so_vec = np.array([10000, 10000])
Sigma = np.array([[0.025, 0],[0, 0.025]]) # covariance of excess returns
lam = 5
P = np.array([1, -1])
Q = np.array([0.02])
Omega = np.array([0.25])
tau = 1

new_return_vec = get_new_return_vec(sp_vec, so_vec,Sigma,lam,tau,P,Q,Omega)
print(new_return_vec)
from matplotlib import pyplot as plt

plt.style.use('seaborn')

desired_returns = np.linspace(min(new_return_vec), max(new_return_vec), 100)

for desired_return in desired_returns:
    std, mean = get_Markowitz_Portfolio_Vec(new_return_vec, Sigma, desired_return)
    plt.scatter(std, mean, c='teal')

plt.show()
