import numpy as np


mean_vec = np.array([0.05, 0.05]) ##Interestingly, this isn't needed

Sigma = np.array([[0.025, 0],[0, 0.025]]) # Supposed to be of excess returns

market_caps = np.array([100, 100])
total_MC = np.sum(market_caps)
market_cap_weights = np.array([])
for index in market_caps:
    market_cap_weights = np.append(market_cap_weights, index/total_MC)

lam = 3.07
Pi = lam *np.matmul(Sigma, market_cap_weights)


P = np.array([1, -1])
Q = np.array([0.02])
Omega = np.array([0.25])
inv_Omega = 1/ Omega # should be np.linalg.inv(Omega) for more than 1 view
t = 1

inv_tSigma = np.linalg.inv(t*Sigma)

inv_left = inv_tSigma + np.matmul(P, inv_Omega * P) # should be np.matmul(inv_Omega, P)
left = np.linalg.inv(inv_left)

right = np.matmul(inv_tSigma, Pi) + P*inv_Omega*Q # should be np.matmul(P, np.matmul(inv_Omega, Q))

print(left)
print(right)
new_return = np.matmul(left, right)
print(new_return)