import torch

torch.manual_seed(0)


def generate_dataset(n_samples=500):
    covariance = torch.tensor([[1, 0.25], [0.25, 1]])
    n_samples = torch.tensor(n_samples)
    mean_class_one = torch.tensor([5, 10], dtype=torch.float32)
    mean_class_two = torch.tensor([0, 5], dtype=torch.float32)
    dist_class_one = torch.distributions.MultivariateNormal(
        loc=mean_class_one, covariance_matrix=covariance
    )
    class_one_features = dist_class_one.rsample(sample_shape=(n_samples,))
    dist_class_two = torch.distributions.MultivariateNormal(
        loc=mean_class_two, covariance_matrix=covariance
    )
    class_two_features = dist_class_two.rsample(sample_shape=(n_samples,))
    features = torch.vstack((class_one_features, class_two_features))
    labels = torch.hstack((-torch.ones(n_samples), torch.ones(n_samples))).reshape(
        (-1, 1)
    )
    return features, labels
