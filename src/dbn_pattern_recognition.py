"""A small Deep Belief Network for MNIST digit pattern recognition.

The feature extractor is pretrained greedily with stacked Bernoulli RBMs.
A supervised linear classifier is then trained on the learned representation.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

try:
    from torchvision import datasets, transforms
except Exception:  # pragma: no cover - compatibility fallback for broken torchvision installs
    class _FallbackToTensor:
        def __call__(self, image):
            array = np.asarray(image, dtype=np.float32)
            if array.ndim == 2:
                array = array.reshape(1, 28, 28)
            elif array.ndim == 3 and array.shape[0] == 1:
                array = array.reshape(1, 28, 28)
            elif array.ndim == 3 and array.shape[-1] == 1:
                array = array[..., 0].reshape(1, 28, 28)
            else:
                raise ValueError(f"Unsupported image shape: {array.shape}")
            return torch.from_numpy(array / 255.0)

    class _FallbackTransforms:
        ToTensor = _FallbackToTensor

    transforms = _FallbackTransforms()

    class _FallbackMNIST(torch.utils.data.Dataset):
        def __init__(self, root, train=True, download=True, transform=None):
            del download
            self.root = Path(root)
            self.root.mkdir(parents=True, exist_ok=True)
            self.transform = transform
            data, targets = fetch_openml("mnist_784", version=1, as_frame=False, cache=True, data_home=str(self.root), parser="auto", return_X_y=True)
            data = data.astype(np.float32) / 255.0
            targets = targets.astype(np.int64)
            if train:
                indices = np.arange(60000)
            else:
                indices = np.arange(60000, 70000)
            self.images = data[indices].reshape(-1, 28, 28)
            self.labels = targets[indices]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            image = self.images[index]
            label = int(self.labels[index])
            if self.transform is not None:
                image = self.transform(image)
            return image, label

    class _FallbackDatasets:
        class MNIST(_FallbackMNIST):
            pass

    datasets = _FallbackDatasets()

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "dataset"

class RBM(nn.Module):
    """Bernoulli Restricted Boltzmann Machine trained with CD-k."""
    def __init__(self, n_visible: int, n_hidden: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

    def sample_hidden(self, visible):
        probabilities = torch.sigmoid(visible @ self.W + self.h_bias)
        return probabilities, torch.bernoulli(probabilities)

    def sample_visible(self, hidden):
        probabilities = torch.sigmoid(hidden @ self.W.T + self.v_bias)
        return probabilities, torch.bernoulli(probabilities)

    def contrastive_divergence(self, batch, lr=0.01, k=1):
        v0 = batch
        _, h0 = self.sample_hidden(v0)
        vk, hk = v0, h0
        for _ in range(k):
            _, vk = self.sample_visible(hk)
            _, hk = self.sample_hidden(vk)
        with torch.no_grad():
            batch_size = max(1, batch.size(0))
            positive = v0.T @ h0
            negative = vk.T @ hk
            self.W.add_(lr * (positive - negative) / batch_size)
            self.v_bias.add_(lr * (v0 - vk).mean(dim=0))
            self.h_bias.add_(lr * (h0 - hk).mean(dim=0))
        return torch.mean((v0 - vk) ** 2).item()

    def hidden_probabilities(self, visible):
        return torch.sigmoid(visible @ self.W + self.h_bias)

class DBNClassifier(nn.Module):
    """Stacked RBM feature extractor with a supervised classification head."""
    def __init__(self, layer_sizes=(256, 128), n_classes=10):
        super().__init__()
        self.layer_sizes = tuple(layer_sizes)
        self.rbms = nn.ModuleList()
        visible = 784
        for hidden in self.layer_sizes:
            self.rbms.append(RBM(visible, hidden))
            visible = hidden
        self.classifier = nn.Linear(visible, n_classes)

    def encode(self, x):
        x = x.view(x.size(0), -1)
        for rbm in self.rbms:
            x = rbm.hidden_probabilities(x)
        return x

    def forward(self, x):
        return self.classifier(self.encode(x))

    def greedy_pretrain(self, train_loader, device, epochs=3, lr=0.01, cd_k=1, limit_batches=None):
        for layer_index, rbm in enumerate(self.rbms):
            print(f"\nPretraining RBM {layer_index + 1}/{len(self.rbms)}")
            for epoch in range(epochs):
                losses, batches = [], 0
                for images, _ in train_loader:
                    x = images.to(device).view(images.size(0), -1)
                    with torch.no_grad():
                        for previous in self.rbms[:layer_index]:
                            x = previous.hidden_probabilities(x)
                        x = torch.bernoulli(x)
                    losses.append(rbm.contrastive_divergence(x, lr=lr, k=cd_k))
                    batches += 1
                    if limit_batches and batches >= limit_batches:
                        break
                print(f"RBM {layer_index + 1}, epoch {epoch + 1}/{epochs}, reconstruction MSE={np.mean(losses):.4f}")


def get_loaders(batch_size=128, data_dir=None):
    data_dir = str(data_dir or (DATA_DIR / "mnist"))
    transform = transforms.ToTensor()
    train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    return (DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(test, batch_size=batch_size, shuffle=False))


def train_classifier(model, train_loader, device, epochs=5, lr=0.001, limit_batches=None):
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    for epoch in range(epochs):
        total_loss, correct, total, batches = 0.0, 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                features = model.encode(images)
            logits = model.classifier(features.detach())
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
            batches += 1
            if limit_batches and batches >= limit_batches:
                break
        print(f"Classifier epoch {epoch + 1}/{epochs}: loss={total_loss/max(1,batches):.4f}, train accuracy={100*correct/max(1,total):.2f}%")


def evaluate(model, loader, device):
    model.eval(); y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(1).cpu().numpy()
            y_pred.extend(predictions.tolist()); y_true.extend(labels.numpy().tolist())
    return np.asarray(y_true), np.asarray(y_pred)


def save_visuals(model, test_loader, device):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    images, labels = next(iter(test_loader))
    with torch.no_grad():
        probs = torch.softmax(model(images.to(device)), dim=1).cpu().numpy()
    fig, axes = plt.subplots(2, 5, figsize=(11, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i, 0], cmap="gray")
        ax.set_title(f"True: {labels[i].item()} | Pred: {probs[i].argmax()}")
        ax.axis("off")
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "classification_results.png", dpi=160); plt.close(fig)

    # Display first-layer hidden-unit weights as learned pattern templates.
    weights = model.rbms[0].W.detach().cpu().T.numpy()
    count = min(25, weights.shape[0])
    fig, axes = plt.subplots(5, 5, figsize=(7, 7))
    for i, ax in enumerate(axes.flat):
        if i < count:
            ax.imshow(weights[i].reshape(28, 28), cmap="coolwarm")
        ax.axis("off")
    fig.suptitle("First RBM: learned hidden-unit weights")
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "learned_patterns.png", dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rbm-epochs", type=int, default=3)
    parser.add_argument("--classifier-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--rbm-hidden", type=int, default=256)
    parser.add_argument("--second-hidden", type=int, default=128)
    parser.add_argument("--limit-batches", type=int, default=0, help="Use a small number of batches for a quick test; 0 means full dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    train_loader, test_loader = get_loaders(args.batch_size)
    model = DBNClassifier((args.rbm_hidden, args.second_hidden)).to(device)
    limit = args.limit_batches or None
    model.greedy_pretrain(train_loader, device, epochs=args.rbm_epochs, limit_batches=limit)
    train_classifier(model, train_loader, device, epochs=args.classifier_epochs, limit_batches=limit)
    y_true, y_pred = evaluate(model, test_loader, device)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nTest accuracy: {accuracy*100:.2f}%")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "layer_sizes": model.layer_sizes, "n_classes": 10}, MODEL_DIR / "dbn_mnist.pt")
    report = {"test_accuracy": float(accuracy), "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0)}
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=list(range(10))).plot(ax=ax, cmap="Blues", colorbar=False)
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=160); plt.close(fig)
    save_visuals(model, test_loader, device)
    print(f"Saved model to {MODEL_DIR / 'dbn_mnist.pt'} and visual results to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
