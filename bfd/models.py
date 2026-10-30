"""Modeller. Ortak arayuz: fit(X_ham, y) ve predict_proba(X_ham) -> (n, 10).

Her model kendi oznitelik cikarimini yapar; olcekleme YALNIZCA egitim
bolmesinde kestirilir.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import features as F


class _FeatureModel:
    feature_fn = staticmethod(F.full_features)
    scale = True

    def __init__(self, n_classes=10, seed=0):
        self.n_classes = n_classes
        self.seed = seed
        self.scaler = None
        self.clf = self._make()

    def _make(self):
        raise NotImplementedError

    def fit(self, X, y):
        Z = self.feature_fn(X)
        if self.scale:
            self.scaler = StandardScaler().fit(Z)
            Z = self.scaler.transform(Z)
        self.clf.fit(Z, y)
        return self

    def predict_proba(self, X):
        Z = self.feature_fn(X)
        if self.scale and self.scaler is not None:
            Z = self.scaler.transform(Z)
        p = self.clf.predict_proba(Z)
        return self._expand(p)

    def _expand(self, p):
        """Egitimde gorulmeyen sinif varsa sutunlari tam boyuta tasir."""
        if p.shape[1] == self.n_classes:
            return p
        out = np.full((len(p), self.n_classes), 1e-12)
        for j, c in enumerate(self.clf.classes_):
            out[:, int(c)] = p[:, j]
        return out / out.sum(1, keepdims=True)


class RandomForestModel(_FeatureModel):
    """M2: zaman + frekans oznitelikleri uzerinde rastgele orman."""
    name = "M2_RF"
    feature_fn = staticmethod(F.full_features)

    def _make(self):
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=1,
                                      n_jobs=-1, random_state=self.seed)


class TrivialModel(_FeatureModel):
    """M3: ONEMSIZ taban cizgisi. Yalnizca RMS ve basiklik, derinlik 4."""
    name = "M3_TRIVIAL"
    feature_fn = staticmethod(F.trivial_features)

    def _make(self):
        return DecisionTreeClassifier(max_depth=4, random_state=self.seed)


class EnvelopeKNNModel(_FeatureModel):
    """M4: zarf spektrumu uzerinde k-NN (k=5; olasilik uretebilmek icin)."""
    name = "M4_KNN"
    feature_fn = staticmethod(lambda X: F.envelope_spectrum(X, 64))

    def _make(self):
        return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)


class CNNModel:
    """M1: 1B evrisimli sinir agi. torch gerektirir."""
    name = "M1_CNN"

    def __init__(self, n_classes=10, seed=0, epochs=30, batch=128, lr=1e-3,
                 patience=5, device=None):
        self.n_classes, self.seed = n_classes, seed
        self.epochs, self.batch, self.lr, self.patience = epochs, batch, lr, patience
        self.device = device
        self.net = None
        self.mu = self.sd = None

    def _build(self, L):
        import torch.nn as nn
        ch = [1, 16, 32, 64, 64]
        layers = []
        for i in range(4):
            layers += [nn.Conv1d(ch[i], ch[i + 1], 9, padding=4),
                       nn.BatchNorm1d(ch[i + 1]), nn.ReLU(), nn.MaxPool1d(2)]
        layers += [nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                   nn.Dropout(0.2), nn.Linear(ch[-1], self.n_classes)]
        return nn.Sequential(*layers)

    def fit(self, X, y):
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as e:  # noqa: BLE001
            raise ImportError("M1_CNN icin torch gerekli: pip install torch") from e
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mu, self.sd = X.mean(), X.std() + 1e-12
        Xn = ((X - self.mu) / self.sd).astype(np.float32)[:, None, :]

        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(y))
        n_val = max(int(0.1 * len(y)), 1)
        va, tr = idx[:n_val], idx[n_val:]
        self.net = self._build(X.shape[1]).to(dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        lossf = torch.nn.CrossEntropyLoss()
        dl = DataLoader(TensorDataset(torch.tensor(Xn[tr]), torch.tensor(y[tr])),
                        batch_size=self.batch, shuffle=True)
        Xva = torch.tensor(Xn[va]).to(dev)
        yva = torch.tensor(y[va]).to(dev)
        best, bad, best_state = np.inf, 0, None
        for _ in range(self.epochs):
            self.net.train()
            for xb, yb in dl:
                opt.zero_grad()
                loss = lossf(self.net(xb.to(dev)), yb.to(dev))
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vl = float(lossf(self.net(Xva), yva))
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        import torch
        dev = next(self.net.parameters()).device
        Xn = ((X - self.mu) / self.sd).astype(np.float32)[:, None, :]
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(Xn), 512):
                z = self.net(torch.tensor(Xn[i:i + 512]).to(dev))
                out.append(torch.softmax(z, 1).cpu().numpy())
        return np.concatenate(out)


ALL_MODELS = {m.name: m for m in [CNNModel, RandomForestModel, TrivialModel, EnvelopeKNNModel]}
FAST_MODELS = {m.name: m for m in [RandomForestModel, TrivialModel, EnvelopeKNNModel]}
