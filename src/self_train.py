import torch
from tqdm import tqdm
from make_dataset import generate_dataset

import matplotlib.pyplot as plt


def warn_exp(z):
    ans = torch.exp(z)
    if torch.any(torch.isinf(ans)):
        print("Overflow detected: result is infinity.")
    return ans


def sigmoid_stable(z):
    mask = z >= 0
    result = torch.zeros_like(z)
    result[mask] = 1 / (1 + warn_exp(-z[mask]))
    result[~mask] = warn_exp(z[~mask]) / (1 + warn_exp(z[~mask]))

    return result


def warn_log(z):
    ans = torch.log(z)
    if torch.any(torch.isinf(ans)):
        print("Overflow detected: result is infinity.")
    return ans


ep = 10e-10


def lls(predictions):
    return -warn_log(torch.clamp(predictions, ep, 1 - ep)).sum()


def gradient(X, y, w, b):
    _, d = X.shape
    wgrad = torch.zeros((d, 1))
    bgrad = torch.tensor(0.0)

    linear_preds = X @ w + b
    activation = sigmoid_stable(-y * linear_preds)

    # weights gradients
    d_dw = activation * y * X
    wgrad = -1 * torch.sum(d_dw, dim=0, keepdim=True).T

    # bias gradient
    bgrad = -1 * torch.sum(activation * y)

    return wgrad, bgrad


def accuracy(predictions, labels):
    predicted_labels = torch.where(predictions > 0.5, 1, -1)
    return (predicted_labels == labels).sum() / labels.shape[0]


def y_pred(X, w, b=0):
    result = sigmoid_stable(X @ w + b)
    return result


def train_logistic_regression(features, labels, max_iter=100, learning_rate=1e-1):
    w = torch.zeros((features.shape[1], 1))
    b = torch.tensor(0.0)
    loss_over_iter = []
    accurary_over_iter = []

    for _ in tqdm(range(max_iter)):
        wgrad, bgrad = gradient(features, labels, w, b)

        w -= learning_rate * wgrad
        b -= learning_rate * bgrad
        linear_preds = features @ w + b
        activation = sigmoid_stable(linear_preds * labels)

        nll = lls(activation).item()
        avg_nll = nll / labels.shape[0]
        loss_over_iter.append(avg_nll)
        acc = accuracy(y_pred(features, w, b), labels).item()
        accurary_over_iter.append(acc)
    return w, b, [loss_over_iter, accurary_over_iter]


def analyze_model_preds(features, labels, w, b, plot=True):
    predictions = y_pred(features, w=w, b=b)
    predicted_labels = torch.where(predictions > 0.5, 1, -1)
    if plot:
        plt.figure(figsize=(9, 6))
        plt.scatter(features[:, 0], features[:, 1], c=predicted_labels, alpha=0.6)
        plt.title("Binary labeled data in 2D", size=15)
        plt.xlabel("Feature 1", size=13)
        plt.ylabel("Feature 2", size=13)
        plt.grid()
    acc = accuracy(predictions, labels)
    print(f"Accuracy {acc:.4f} (want close to 1)")
    log_loss_sum = lls(predictions)
    print(f"Log Loss Sum: {log_loss_sum:.4f} (want close to 0)")
    print(
        f"Log Loss Average: {log_loss_sum/predictions.shape[0]:.4f} (want close to 0)"
    )


def plot_training_history(loss_over_iter, accuray_over_iter):
    fig, ax1 = plt.subplots()
    ax1.plot(loss_over_iter, color="tab:red")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Average Log Loss")

    ax2 = ax1.twinx()
    ax2.plot(accuray_over_iter, color="tab:blue")
    ax2.set_ylabel("Accuracy")


if __name__ == "__main__":
    features, labels = generate_dataset()
    w, b, [loss_over_iter, accuray_over_iter] = train_logistic_regression(
        features, labels, max_iter=100, learning_rate=0.1
    )
    print(f"Model Params(w={w.flatten()} {b=})")
    analyze_model_preds(features, labels, w, b, plot=False)
