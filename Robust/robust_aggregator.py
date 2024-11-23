import numpy as np
import torch

# Put any functions you write for Question 1 here. 
def max_var_and_direction(covar_mat): # Q 1.1
    """
    Compute the maximum variance and the direction of maximum variance of the
    given covariance matrix.
    Args:
        covar_mat (np.ndarray): covariance matrix of gradients.
    Returns:
        max_variance (float): maximum variance of the covariance matrix.
        max_var_direction (np.ndarray): direction of maximum variance.
    """
    eigen_result = np.linalg.eig(covar_mat)
    max_variance = np.max(np.real(eigen_result.eigenvalues))
    max_var_idx = np.argmax(np.real(eigen_result.eigenvalues))
    max_var_direction = eigen_result.eigenvectors[:, max_var_idx]
    return max_variance, max_var_direction

def find_outlier(grads, max_var_direction): # Q 1.2
    """
    Find the most possible outlier gradient in the given gradients w.r.t. the
    maximum variance direction.
    Args:
        grads (torch.Tensor): gradients of the model.
        max_var_direction (np.ndarray): direction of maximum variance.
    Returns:
        outlier_idx (int): index of the outlier gradient.
        abs_projected_distances (np.ndarray): projected distances of each gradient
        from the mean gradient w.r.t. the maximum variance direction.
    """
    mean_grad = np.mean(grads.numpy(), axis=0)
    delta_grads = grads.numpy() - mean_grad
    abs_projected_distances = np.abs(np.dot(delta_grads, max_var_direction))
    outlier_idx = np.argmax(abs_projected_distances)
    return outlier_idx, abs_projected_distances[outlier_idx]

def remove_outlier(grads, outlier_idx): # Q 1.3
    """
    Remove the outlier gradient from the given gradients and calculate the
    covariance matrix of the pruned gradients.
    Args:
        grads (torch.Tensor): gradients of the model.
        outlier_idx (int): index of the outlier gradient.
    Returns:
        pruned_grads (torch.Tensor): pruned gradients.
        pruned_cov_mat (np.ndarray): covariance matrix of the pruned gradients.
    """
    pruned_grads = torch.cat((grads[:outlier_idx], grads[outlier_idx + 1:]))
    pruned_cov_mat = np.cov(pruned_grads.numpy(), rowvar=False)
    return pruned_grads, pruned_cov_mat

def main():
    grads = torch.load("all_gradients.pt").reshape(-1, 1536)
    covar_mat = np.cov(grads, rowvar=False)

    # Question 1.1
    # compute:
    # maximum variance, i.e. maximum eigenvalue of cov matrix, and direction 
    # of maximum variance, i.e. eigenvector corresponding to maximum eigenvalue.
    max_var, max_var_direction = max_var_and_direction(covar_mat)
    print(f"Maximum variance: {max_var}")
    # Maximum variance: 19781804.514221925
    print(f"Direction of maximum variance: {max_var_direction}")
    # Direction of maximum variance: [-0.02526271+0.j -0.01008863+0.j
    # -0.03705509+0.j ...  0.03702794+0.j 0.0112057 +0.j -0.00446688+0.j]

    # Question 1.2
    # compute projected distance of each gradient from the mean gradient w.r.t. 
    # the maximum variance direction. Find most possible outlier gradient.
    outlier_idx, outlier_grad = find_outlier(grads, max_var_direction)
    print(f"Index of outlier: {outlier_idx}")
    # Index of outlier: 872
    print(f"Projected distance of outlier: {outlier_grad}")
    # Projected distance of outlier: 10922.767278341198
    print(f"outlier gradient: {grads[outlier_idx]}")
    # outlier gradient: tensor([-0.0000,  0.0249, -1.1044,  ...,  1.0991, 
    # -0.2681,  0.7061])

    # Question 1.3
    # Remove the outlier from grads, calculate the change of maximum variance
    grads, pruned_covar_mat = remove_outlier(grads, outlier_idx)
    max_var_pruned, _ = max_var_and_direction(pruned_covar_mat)
    print(f"Pruned maximum variance: {max_var_pruned}")
    # Pruned maximum variance: 19717253.461455297
    print(f"Change of maximum variance: {max_var_pruned - max_var}")
    # Change of maximum variance: -64551.05276662856

    # Question 1.4
    # detect the outliers until the maximum variance is less than the threshold
    k, benign_spec_norm_of_covmat = 9, 39725 # given
    threshold = k * benign_spec_norm_of_covmat
    num_outliers = 1
    while max_var_pruned > threshold:
        outlier_idx, _ = find_outlier(grads, max_var_direction)
        grads, pruned_covar_mat = remove_outlier(grads, outlier_idx)
        max_var_pruned, _ = max_var_and_direction(pruned_covar_mat)
        print(f"pruned maximum variance: {max_var_pruned}, \
              Number of outliers: {num_outliers}")
        num_outliers += 1
    # number of outliers detected: 555, percentage of outliers: 82.5%


# For question 2, it would be helpful to define a function that returns the set 
# of pruned gradients
def robust_aggregator(gradients):
    # gradients.shape = (batch_size, dimension=1536)
    # Run pruning procedure
    pruned_gradients = gradients
    return pruned_gradients


if __name__ == "__main__":
    main()