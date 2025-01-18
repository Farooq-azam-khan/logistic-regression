from torch import nn
import torch
from tqdm import tqdm
from make_dataset import generate_dataset


def accuracy(predictions, labels):
    predicted_labels = torch.where(predictions > 0.5, 1, 0)
    return (predicted_labels == labels).sum() / labels.shape[0]


class LogisticRegression(nn.Module):
    def __init__(self, feature_dim: int = 2) -> None:
        super().__init__()
        self.w = torch.zeros((1, feature_dim), requires_grad=True)
        self.b = torch.tensor(0.0, requires_grad=True)

    def forward(self, X):
        return nn.functional.sigmoid(X @ self.w.T + self.b)


def train_logistic_regression(features, labels, max_iter=100, learning_rate=1e-1):
    model = LogisticRegression(features.shape[1])
    loss_fn = nn.BCELoss()
    loss_over_iter = []
    accurary_over_iter = []

    for _ in tqdm(range(max_iter)):
        predictions = model(features)
        loss = loss_fn(predictions, labels)
        loss.backward()
        loss_over_iter.append(loss.item())

        model.w.data -= learning_rate * model.w.grad.data
        model.b.data -= learning_rate * model.b.grad.data
        acc = accuracy(predictions, labels)
        accurary_over_iter.append(acc.item())
        model.w.grad.data.zero_()
        model.b.grad.data.zero_()

    return model, [loss_over_iter, accurary_over_iter]


if __name__ == "__main__":
    features, labels = generate_dataset()
    labels = (labels + 1) / 2  # for binary cross entropy function
    model, [loss_over_iter, accuray_over_iter] = train_logistic_regression(
        features, labels, max_iter=100, learning_rate=0.1
    )
    print(f"Model(w={model.w.detach().numpy()}, b={model.b.item()})")
    print(f"Accuracy={accuray_over_iter[-1]}")
