from torch import nn
import torch
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from self_train import accuracy

device = "cuda" if torch.cuda.is_available() else "cpu"


class CustomDataset(Dataset):
    def __init__(self, n_samples=500):
        """
        Custom dataset for logistic regression.

        :param n_samples: Number of samples per class (default: 500)
        """

        # Generate dataset
        self.features, self.labels = self.generate_dataset(n_samples)

    def generate_dataset(self, n_samples):
        """Generates synthetic data for logistic regression classification."""
        torch.manual_seed(0)  # Ensures reproducibility
        covariance = torch.tensor([[1, 0.25], [0.25, 1]], dtype=torch.float32)

        # Define means for two classes
        mean_class_one = torch.tensor([5, 10], dtype=torch.float32)
        mean_class_two = torch.tensor([0, 5], dtype=torch.float32)

        # Generate Gaussian distributed samples
        dist_class_one = torch.distributions.MultivariateNormal(
            mean_class_one, covariance
        )
        dist_class_two = torch.distributions.MultivariateNormal(
            mean_class_two, covariance
        )

        class_one_features = dist_class_one.sample((n_samples,))
        class_two_features = dist_class_two.sample((n_samples,))

        # Stack features and labels
        features = torch.vstack((class_one_features, class_two_features))
        labels = torch.hstack((torch.zeros(n_samples), torch.ones(n_samples))).reshape(
            (-1, 1)
        )

        return features.to(device), labels.to(device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def accuracy(predictions, labels):
    predicted_labels = torch.where(predictions > 0.5, 1, 0)
    return (predicted_labels == labels).sum() / labels.shape[0]


class LogisticRegression(nn.Module):
    def __init__(self, feature_dim: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, 1, bias=True)

    def forward(self, X):
        return self.linear(X)


def train_logistic_regression(
    n_samples=500, max_iter=100, learning_rate=1e-1, batch_size=32
):
    dataset = CustomDataset(n_samples=n_samples)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    # DataLoader for batch processing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = LogisticRegression(feature_dim=dataset.features.shape[1]).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_over_iter = []
    accurary_over_iter = []

    for epoch in (p := tqdm(range(max_iter))):
        p.set_description(f"epoch={epoch+1}")
        model.train()
        total_loss = 0
        for features, labels in train_loader:
            optimizer.zero_grad()
            logits = model(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_over_iter.append(total_loss / len(train_loader))

        # compute validation accuracy
        model.eval()
        with torch.no_grad():
            total_acc = 0
            for features, labels in val_loader:
                predictions = torch.sigmoid(model(features))
                total_acc += accuracy(predictions, labels).item()
            accurary_over_iter.append(total_acc / len(val_loader))

    return model, [loss_over_iter, accurary_over_iter]


if __name__ == "__main__":
    print(f"Running on {device=}")
    model, [loss_over_iter, accuray_over_iter] = train_logistic_regression(
        max_iter=100, learning_rate=0.1
    )
    print(
        f"Model(w={model.linear.weight.detach().numpy()}, b={model.linear.bias.detach().numpy()})"
    )
    print(f"Validation Accuracy={accuray_over_iter[-1]}")
