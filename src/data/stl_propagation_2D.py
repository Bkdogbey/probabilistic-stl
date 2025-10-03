import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, solve
from scipy.stats import norm

# Dynamic System Parameters
n = 2  # State dimension: [y, z]
A = np.array([[0.05, 0.2], [0.3, 0.04]])  # Dynamics matrix (stable with coupling)
B = np.array([[1.0],
              [0.6]])
G = np.eye(2) * 0.4
W = np.eye(2) * 0.2

# Initial State
mu0 = np.array([53.0, 55.0])          # initial mean 
P0  = np.diag([6.0, 5.0])             # initial covariance

t = np.linspace(0.0, 10.0, 80)        # time grid

def u_func(tt: float) -> float:
    """Scalar control input u(t)."""
    return -0.5
def propagate_gaussian(A, B, G, W, mu0, P0, u_func, t):
    # Propagate the belief through linear SDE dX = (A X + B u) dt + G dW
    # A: (n,n), B: (n,m), G: (n,p), W: (p,p)
    # mu0: (n,), P0: (n,n)  
    
    n = A.shape[0]                             # state dimension
    mean_trace = np.zeros((len(t), n))        # mean trajectory
    cov_trace  = np.zeros((len(t), n, n))     # covariance trajectory
    mean_trace[0] = mu0                       # initial mean
    cov_trace[0]  = P0                        # initial covariance

    I = np.eye(n)                             # identity matrix
    for k in range(1, len(t)):                # for each time step
        dt = float(t[k] - t[k-1])             # time step
        u = u_func(t[k-1])                    # control input at previous time

        Phi = expm(A * dt)                    # state transition matrix
        # Gamma: effect of control over dt; solve A * X = (Phi - I) @ B  => X = A^{-1} ((Phi-I) B)
        Gamma = solve(A, np.dot((Phi - I), B))

        # support scalar or vector control
        if np.isscalar(u):
            u_vec = np.full((B.shape[1],), float(u))
        else:
            u_vec = np.asarray(u, dtype=float)

        Qd = np.dot(G, np.dot(W, G.T)) * dt                # discrete-time process noise covariance

        mean_trace[k] = np.dot(Phi, mean_trace[k-1]) + np.dot(Gamma, u_vec)   # mean update
        cov_trace[k]  = np.dot(Phi, np.dot(cov_trace[k-1], Phi.T)) + Qd      # covariance update

    return mean_trace, cov_trace

mean_trace, cov_trace = propagate_gaussian(A, B, G, W, mu0, P0, u_func, t)


# Bounds Calculation
def bounds_calculation(mean_trace, var_trace, t, min_height=50, start_time=0, end_time=20, complete_trace=True):
    mask = (t >= start_time) & (t <= end_time)
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        return 0.0, 1.0                    # no time in the interval
    probs = []                              # P(z(t) >= min_height) at each time in [start_time, end_time] 
    for i in idxs:
        m = float(mean_trace[i, 1])       # mean height (z)
        v = float(var_trace[i])           # variance of height (z)
        if v <= 0:
            p = 1.0 if m >= min_height else 0.0
        else:
            std = np.sqrt(v)
            p = 1.0 - norm.cdf((min_height - m) / std)
        probs.append(p)                     # Update probability list
    min_prob = float(np.min(probs))  # Choosing the minimum probability over the interval due the G operator
    return min_prob, min_prob           # return the bounds

mean_trace, cov_trace = propagate_gaussian(A, B, G, W, mu0, P0, u_func, t)      
lower_bound, upper_bound = bounds_calculation(mean_trace, cov_trace[:, 1, 1], t, min_height=50, start_time=0, end_time=20, complete_trace=True)
print(f"Stori Bounds for G[0,10] (z >= 50): [{lower_bound:.3f}, {upper_bound:.3f}]")

# Plotting the mean and sigma interval
plt.figure(figsize=(10, 5))
std = np.sqrt(cov_trace[:,1,1])
plt.plot(t, mean_trace[:,1], label='Mean Height', color='blue') 
plt.fill_between(t, mean_trace[:,1] - std, mean_trace[:,1] + std, color='blue', alpha=0.2, label='1-sigma Interval')
plt.axhline(50, color='red', linestyle='--', label='Height Threshold (50)')
plt.title('Height (z) Over Time with 1-sigma Interval')         
plt.xlabel('Time')
plt.ylabel('Height (z)')    
plt.legend()
plt.grid()
plt.show()
