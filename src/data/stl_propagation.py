import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# --------- System Dynamics ------------#
a = 0.1  # state
b = 1.0  # input
g = 0.5  # Stochastic noise
q = 0.1  # process noise covariance

mu = 53  # mean height
P = 5  # initial height variance

t = np.linspace(0, 10, 50)  # time from 0 to 30 seconds as given by stl


def u_func(t):
    """Control input function u(t)."""
    return -0.5  # constant downward velocity


# We begin propagating the belief trajectory
def propagate(a, b, g, q, mu, P, u, t):
    """Propagate the belief state (mu, P) through one time step."""
    mean_trace = np.zeros(len(t))
    var_trace = np.zeros(len(t))

    mean_trace[0] = mu
    var_trace[0] = P
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        u = u_func(t[i - 1])  # control input at time t[i-1]

        Phi = np.exp(a * dt)
        int_u = dt * b * u  # integral of b*u from t[i-1] to t[i]
        mean_trace[i] = Phi * mean_trace[i - 1] + int_u

        # Variance update
        var_trace[i] = (Phi**2) * var_trace[i - 1] + (g**2) * dt + q * dt
    return mean_trace, var_trace


mean_trace, var_trace = propagate(a, b, g, q, mu, P, u_func, t)


# Bounds Calculation
def compute_bounds(
    mean_trace,
    var_trace,
    t,
    min_height=50,
    start_time=0,
    end_time=10,
    complete_trace=True,
):
    mask = (t >= start_time) & (t <= end_time)
    idxs = np.where(mask)[0]
    if np.sum(mask) == 0:
        return 0.0, 1.0  # no time in the interval

    probs = []  # P(z(t) >= min_height) at each time in [start_time, end_time]
    for i in idxs:
        m = mean_trace[i]
        v = var_trace[i]
        if v <= 0:
            p = 1.0 if m >= min_height else 0.0
        else:
            std = np.sqrt(v)
            # P(Z >= h) = 1 - Phi((h - m)/std)
            p = 1.0 - norm.cdf((min_height - m) / std)
        probs.append(p)

    min_prob = float(
        np.min(probs)
    )  # Choosing the minimum probability over the interval due the G operator
    # Here we don't compute a separate loose upper bound; set both to min_prob.
    return min_prob, min_prob  # return the bounds


# Plots
# Plot the mean and sigma interval
plt.figure(figsize=(12, 6))
plt.plot(t, mean_trace, label="Mean Height", color="blue")
plt.fill_between(
    t,
    mean_trace - np.sqrt(var_trace),
    mean_trace + np.sqrt(var_trace),
    color="blue",
    alpha=0.2,
    label="1-sigma Interval",
)
plt.axhline(50, color="red", linestyle="--", label="Threshold Height = 50m")
plt.title("Height Trajectory with 1-sigma Confidence Interval")
plt.xlabel("Time [s]")
plt.ylabel("Height [m]")
plt.legend()
plt.grid()
plt.show()
lb, ub = compute_bounds(
    mean_trace,
    var_trace,
    t,
    min_height=50,
    start_time=0,
    end_time=10,
    complete_trace=True,
)
print(f"Stori Bounds for G[0,10] (z >= 50): [{lb:.3f}, {ub:.3f}]")  # Computed bounds
