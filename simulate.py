from sklearn import linear_model
import numpy as np 
import os 
import argparse 



def min_lasso(X, y, theta):
    # returns a matrix of d x num_knots (max = dim) coefficients
    coefs = linear_model.lars_path(X, y, 
                                   method='lasso', 
                                   max_iter=np.Inf)[2]
    return compute_min_path(X, y, coefs, theta)

def compute_min_path(X, y, coefs, theta):
    min_dist, num_coefs = np.Inf, coefs.shape[1]
    # compute minimum along end points: 
    for i in range(num_coefs):
        min_dist = min(min_dist, 
                       np.linalg.norm(coefs[:, i] - theta) ** 2)
    # compute minimum within intervals
    for i in range(1, num_coefs):
        min_dist = min(min_dist, 
                       min_interval(theta, coefs[:, i-1], coefs[:, i]))
    return min_dist

def min_interval(target, start, stop):
    """
    Computes the global minimum of (sqrt of)
    ||target - x_t||_2^2 over t in R 
    Returns value if attained at t \in [0, 1]
    Returns np.Inf otherwise. 
    x_t = start + t*(stop - start). 
    ||start + t*(stop - start) - target||^2 = 
    ||start - target||^2 + 2t <(stop - start), start-target> 
    + t^2 ||stop - start||^2. 
    Let c = ||start - target||^2, b = 2<(stop - start), start-target> , 
    and a = ||stop - start||^2. 
    Then minimum of a t^2 + b t + c is at 
    t_opt = -b/2a 
    """
    a = np.linalg.norm(stop - start) ** 2 
    b = 2 * np.dot(stop-start, start-target)
    c = np.linalg.norm(start-target) ** 2 
    t_opt = -b/(2*a)
    if 0 <= t_opt <= 1:
        return a*np.square(t_opt) + b * t_opt + c
    else:
        return np.Inf

def stols_error(X, y, theta, lambda_val):
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    stols = ols/np.abs(ols) * np.maximum(np.abs(ols) - lambda_val, 0)
    return np.linalg.norm(stols - theta) ** 2

def make_instance(n, p):
    alpha, B = np.sqrt(n), np.sqrt(n)
    k = int(np.floor(n/2))
    # construct design matrix 
    X = np.zeros((n, n))
    for j in range(k):
        X[j, j] = np.sqrt(n * alpha / B)
    for j in range(k, n):
        X[j, j] = np.sqrt(n * 1.0 / B)
    # construct parameter vector 
    theta = np.zeros((n,))
    if p == 0.0: 
        theta[k] = 2 * np.sqrt(B * alpha / n)
        lamb_opt = np.sqrt(2 * B * (1 + np.log(n))/n)
    else:
        theta[k] = 1.0
        lamb_opt = np.sqrt((2/np.sqrt(n)) * (1 + (1 - p/4) * np.log(n)))
    # optimal threshold for stols 
    return X, theta, lamb_opt

def run_sample_size(n, log_dir, trial, p):
    X, theta, lambda_opt = make_instance(n, p)
    y = X @ theta + np.random.randn(n)
    best_lasso = min_lasso(X, y, theta)
    stols_err = stols_error(X, y, theta, lambda_opt)
    with open(os.path.join(log_dir, 
                           "result_n_{}_trial_{}.log".format(n, trial)), "w") as f:
        f.write("Lasso: {}\n".format(best_lasso))
        f.write("STOLS: {}".format(stols_err))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Lasso and STOLS simulation')
    parser.add_argument('sample_size', type=int,
                        help='Sample size (required)')
    parser.add_argument('trial', type=int, 
                        help='Trial number (required)')
    parser.add_argument('p', type=float, 
                        help='Order p of l_p constriant')
    parser.add_argument('path', type=str, help='path to save files')
    args = parser.parse_args()
    curr_log_dir = os.path.join(args.path, "results_p_{}".format(args.p))
    if not os.path.isdir(curr_log_dir):
        os.mkdir(curr_log_dir) 
    run_sample_size(n=args.sample_size,
                    log_dir=curr_log_dir, 
                    trial=args.trial, 
                    p=args.p)
    
    
