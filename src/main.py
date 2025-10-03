import matplotlib.pyplot as plt
import numpy as np
import yaml
from data.stl_propagation import compute_bounds, propagate, u_func
from utils import skip_run

# The configuration file
config_path = "configs/config.yml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run("run", "stl_propagation") as check, check():
    # --------- System Dynamics ------------#
    a = 0.1  # state
    b = 1.0  # input
    g = 0.5  # Stochastic noise
    q = 0.1  # process noise covariance

    mu = 45  # mean height
    P = 5  # initial height variance

    t = np.linspace(0, 5, 30)  # time from 0 to 30 seconds as given by stl
    mean_trace, var_trace = propagate(a, b, g, q, mu, P, u_func, t)

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
    print(
        f"Stori Bounds for G[0,10] (z >= 50): [{lb:.3f}, {ub:.3f}]"
    )  # Computed bounds
