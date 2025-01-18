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
        # self.w = torch.zeros((1, feature_dim), requires_grad=True)
        # self.b = torch.tensor(0.0, requires_grad=True)
        self.linear = nn.Linear(feature_dim, 1, bias=True)

    def forward(self, X):
        # return nn.functional.sigmoid(X @ self.w.T + self.b)
        # return nn.functional.sigmoid(self.linear(X))
        return self.linear(X)


def train_logistic_regression(features, labels, max_iter=100, learning_rate=1e-1):
    model = LogisticRegression(features.shape[1])
    # loss_fn = nn.BCELoss()
    loss_fn = nn.BCEWithLogitsLoss()  # numerically stable loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_over_iter = []
    accurary_over_iter = []

    for _ in tqdm(range(max_iter)):
        optimizer.zero_grad()
        logits = model(features)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        loss_over_iter.append(loss.item())

        with torch.no_grad():
            acc = accuracy(torch.sigmoid(logits), labels)
            accurary_over_iter.append(acc.item())

    return model, [loss_over_iter, accurary_over_iter]


if __name__ == "__main__":
    features, labels = generate_dataset()
    labels = (labels + 1) / 2  # for binary cross entropy function
    model, [loss_over_iter, accuray_over_iter] = train_logistic_regression(
        features, labels, max_iter=100, learning_rate=0.1
    )
    print(
        f"Model(w={model.linear.weight.detach().numpy()}, b={model.linear.bias.detach().numpy()})"
    )
    print(f"Accuracy={accuray_over_iter[-1]}")
