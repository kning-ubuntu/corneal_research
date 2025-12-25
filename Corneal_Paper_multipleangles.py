# -*- coding: utf-8 -*-
"""
Corneal Topography — Multi-angle (360)
Optimized Dataset + NN vs Arc-step vs Klein
(Faithful baselines + paper-ready plots)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# Calibration
# =========================================================

@dataclass
class Calib:
    f: float
    Lc: float
    Lp: np.ndarray
    rings_mm: np.ndarray


def build_calib(K=13, f=4.74, Lc=50.0, Lp_base=50.0, dome_mm=45.0):
    r_min, r_max = 4.0, 20.0
    t = np.linspace(0, 1, K)
    rings_mm = r_min + (r_max - r_min) * (t ** 0.45)
    u = rings_mm / rings_mm[-1]
    Lp = Lp_base - dome_mm * (u ** 2)
    return Calib(
        f=f,
        Lc=Lc,
        Lp=Lp.astype(np.float32),
        rings_mm=rings_mm.astype(np.float32),
    )

# =========================================================
# Corneal surface
# =========================================================

def asphere_sag(r, R, Q):
    inside = np.maximum(1 - (1 + Q) * (r ** 2) / (R ** 2), 1e-12)
    return -(r ** 2) / (R * (1 + np.sqrt(inside)))

# =========================================================
# Optics helpers
# =========================================================

def forward_project_r_to_rho(r, z, f, Lc):
    return f * r / (Lc - z + 1e-9)


def find_reflection_radius_simple(Rpk, Lp_k, Lc, R, Q, r_max, n_grid=120):
    rs = np.linspace(0, r_max, n_grid)
    best_r, best_ang = 0.0, 1e9

    for r in rs:
        z = asphere_sag(np.array([r], np.float32), R, Q)[0]
        r2 = r + 1e-4
        z2 = asphere_sag(np.array([r2], np.float32), R, Q)[0]
        dzdr = (z2 - z) / (r2 - r)

        n = np.array([-dzdr, 1.0], np.float32)
        n /= (np.linalg.norm(n) + 1e-12)

        P = np.array([r, z], np.float32)
        S = np.array([Rpk, Lp_k], np.float32)
        C = np.array([0.0, Lc], np.float32)

        s = S - P; s /= (np.linalg.norm(s) + 1e-12)
        c = C - P; c /= (np.linalg.norm(c) + 1e-12)
        b = s + c; b /= (np.linalg.norm(b) + 1e-12)

        ang = np.arccos(np.clip(np.dot(n, b), -1, 1))
        if ang < best_ang:
            best_ang, best_r = ang, r

    return float(best_r)

# =========================================================
# Optimized Dataset (fast)
# =========================================================

class TinyPlacidoDataset(torch.utils.data.Dataset):
    def __init__(self,
                 N,
                 calib,
                 A=360,
                 K=13,
                 Rr=64,
                 r_max=4.5,
                 R_range=(7.0, 9.5),
                 Q_range=(-1.2, 0.2),
                 jitter_px=0.002,
                 missing_prob=0.15,
                 missing_block_prob=0.35,
                 seed=0):

        self.N = int(N)
        self.calib = calib
        self.A = int(A)
        self.K = int(K)
        self.Rr = int(Rr)
        self.r_max = float(r_max)

        self.R_range = tuple(R_range)
        self.Q_range = tuple(Q_range)

        self.jitter_px = float(jitter_px)
        self.missing_prob = float(missing_prob)
        self.missing_block_prob = float(missing_block_prob)

        self.r_grid = np.linspace(0, self.r_max, self.Rr).astype(np.float32)
        self._rng = np.random.default_rng(int(seed))

    def __len__(self):
        return self.N

    def _make_missing_mask(self):
        mask = np.ones((self.A, self.K), dtype=np.float32)

        if self.missing_prob > 0:
            drop = self._rng.random((self.A, self.K)) < self.missing_prob
            mask[drop] = 0.0

        if self.missing_block_prob > 0:
            angles = np.where(self._rng.random(self.A) < self.missing_block_prob)[0]
            for a in angles:
                if self.K <= 3:
                    continue
                blen = int(self._rng.integers(2, max(3, self.K // 2 + 1)))
                start = int(self._rng.integers(0, self.K - blen + 1))
                mask[a, start:start + blen] = 0.0

        return mask

    def __getitem__(self, idx):

        R0 = float(self._rng.uniform(*self.R_range))
        Q0 = float(self._rng.uniform(*self.Q_range))

        z_base = asphere_sag(self.r_grid, R0, Q0).astype(np.float32)
        z_target = np.tile(z_base[None, :], (self.A, 1))

        # reflection radius once per ring
        r_star_k = np.zeros(self.K, dtype=np.float32)
        for k in range(self.K):
            r_star_k[k] = find_reflection_radius_simple(
                self.calib.rings_mm[k],
                self.calib.Lp[k],
                self.calib.Lc,
                R0, Q0,
                self.r_max
            )

        rk = np.tile(r_star_k[None, :], (self.A, 1))
        idx_k = np.clip(np.searchsorted(self.r_grid, r_star_k), 0, self.Rr - 1)
        zk = z_target[:, idx_k]

        rho = forward_project_r_to_rho(rk, zk, self.calib.f, self.calib.Lc)

        if self.jitter_px > 0:
            rho += self._rng.normal(0.0, self.jitter_px, rho.shape).astype(np.float32)

        mask = self._make_missing_mask()
        rho_obs = rho.copy()
        rho_obs[mask < 0.5] = 0.0

        return {
            "rho": torch.tensor(rho_obs, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "z_target": torch.tensor(z_target, dtype=torch.float32),
            "r_grid": torch.tensor(self.r_grid, dtype=torch.float32),
            "rk": torch.tensor(rk, dtype=torch.float32),
            "R0": R0,
            "Q0": Q0,
        }

# =========================================================
# Neural model (unchanged)
# =========================================================

class EncoderDecoderNet(nn.Module):
    def __init__(self, K, Rr):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, Rr)
        )

    def forward(self, rho):
        x = rho.unsqueeze(1)
        x = self.enc(x)
        x = x.mean(dim=-1)
        return self.dec(x)

# =========================================================
# Arc-step (ORIGINAL)
# =========================================================

def arc_step_fixed(rho, f, Lc, smooth_window=3):
    rho = np.asarray(rho, np.float32)

    if smooth_window and smooth_window > 1:
        w = int(smooth_window) | 1
        pad = w // 2
        rho_p = np.pad(rho, (pad, pad), mode="edge")
        ker = np.ones(w, np.float32) / float(w)
        rho = np.convolve(rho_p, ker, mode="valid")

    r = rho * (Lc / f)
    r = np.maximum.accumulate(r)

    dr = np.diff(r)
    R_local = np.zeros_like(r)
    for i in range(1, len(r) - 1):
        R_local[i] = 0.5 * (dr[i] + dr[i - 1])
    R_local[0] = R_local[1]
    R_local[-1] = R_local[-2]

    R_local = np.clip(R_local, 6.5, 12.0)

    z = np.zeros_like(r)
    for i in range(len(r)):
        Ri = float(R_local[i])
        z[i] = Ri - np.sqrt(max(Ri * Ri - r[i] * r[i], 1e-6))

    z = -z
    z -= z[0]
    return r.astype(np.float32), z.astype(np.float32)

# =========================================================
# Klein-like
# =========================================================

def klein_like_z(rho, r_k, smooth_window=5, eps=1e-6):
    """
    Klein-like slope integration with:
      - Option A: drop duplicated r == r_max (aperture saturation)
      - Interpolation of missing rho values (rho == 0)
    """

    rho = np.asarray(rho, np.float32).copy()
    r = np.asarray(r_k, np.float32).copy()

    # -------------------------------------------------
    # 1. Interpolate missing rho (rho == 0)
    #    Klein requires continuous physical signal
    # -------------------------------------------------
    m = rho != 0.0
    if np.sum(m) >= 2:
        x = np.arange(len(rho))
        rho[~m] = np.interp(x[~m], x[m], rho[m])
    else:
        # Too few valid rings → Klein undefined
        return r.astype(np.float32), np.zeros_like(r, dtype=np.float32)

    # -------------------------------------------------
    # 2. Option A: drop duplicated outer rings at r_max
    # -------------------------------------------------
    r_max = np.max(r)
    is_edge = np.isclose(r, r_max, atol=1e-6)

    if np.sum(is_edge) > 1:
        keep = np.ones_like(r, dtype=bool)
        edge_idx = np.where(is_edge)[0]
        keep[edge_idx[1:]] = False

        r = r[keep]
        rho = rho[keep]

    # If too few points remain, Klein is invalid
    if r.size < 3:
        return r.astype(np.float32), np.zeros_like(r, dtype=np.float32)

    # -------------------------------------------------
    # 3. Optional smoothing (your original logic)
    # -------------------------------------------------
    if smooth_window and smooth_window > 1:
        w = int(smooth_window) | 1
        pad = w // 2
        rho_p = np.pad(rho, (pad, pad), mode="edge")
        ker = np.ones(w, np.float32) / float(w)
        rho = np.convolve(rho_p, ker, mode="valid").astype(np.float32)

    # -------------------------------------------------
    # 4. Original Klein math (unchanged)
    # -------------------------------------------------
    drho_dr = np.gradient(rho, r + eps)
    kappa = drho_dr / (r + eps)

    dr = np.gradient(r)
    slope = np.cumsum(kappa * dr)
    z = np.cumsum(slope * dr)

    z = z - z[0]
    z = -z  # physical corneal convention

    return r.astype(np.float32), z.astype(np.float32)


# =========================================================
# Training
# =========================================================

def train_epoch(model, loader, optim):
    model.train()
    total = 0.0
    for b in loader:
        rho = b["rho"].to(DEVICE)       # [B,A,K]
        zt = b["z_target"].to(DEVICE)   # [B,A,Rr]

        B, A, K = rho.shape
        Rr = zt.shape[-1]

        rho = rho.reshape(B * A, K)
        zt = zt.reshape(B * A, Rr)

        pred = model(rho)
        loss = F.smooth_l1_loss(pred, zt)

        optim.zero_grad()
        loss.backward()
        optim.step()
        total += loss.item()

    return total / len(loader)

# =========================================================
# Plotting + metrics (YOUR CODE)
# =========================================================

def compute_metrics(z_pred, z_gt):
    e = (z_pred - z_gt).astype(np.float32)
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e * e)))
    return mae, rmse


def summary_print(z_gt, z_nn, z_arc, z_klein):
    mae_nn, rmse_nn = compute_metrics(z_nn, z_gt)
    mae_a, rmse_a = compute_metrics(z_arc, z_gt)
    mae_k, rmse_k = compute_metrics(z_klein, z_gt)

    print("=== Overall metrics over ALL angles and radii ===")
    print(f"NN     : MAE={mae_nn:.6f}  RMSE={rmse_nn:.6f}")
    print(f"Arc    : MAE={mae_a:.6f}  RMSE={rmse_a:.6f}")
    print(f"Klein  : MAE={mae_k:.6f}  RMSE={rmse_k:.6f}")


def plot_four_angles(r_grid, z_gt, z_nn, z_arc, z_klein, angles_deg=(45, 135, 225, 270)):
    plt.figure(figsize=(12, 9))
    for i, deg in enumerate(angles_deg, 1):
        a = int(deg) % z_gt.shape[0]
        ax = plt.subplot(2, 2, i)
        ax.plot(r_grid, z_gt[a], "k--", lw=2, label="GT")
        ax.plot(r_grid, z_nn[a], "r", lw=2, label="NN")
        ax.plot(r_grid, z_arc[a], "g:", lw=2.5, label="Arc-step")
        ax.plot(r_grid, z_klein[a], "--", color="purple", lw=2, label="Klein")
        ax.set_title(f"Angle {deg}°")
        ax.set_xlabel("r (mm)")
        ax.set_ylabel("z (mm)")
        ax.grid(True)
        if i == 1:
            ax.legend()
    plt.tight_layout()
    plt.show()


def plot_per_angle_error(z_gt, z_nn, z_arc, z_klein):
    err_nn = np.mean(np.abs(z_nn - z_gt), axis=1)
    err_arc = np.mean(np.abs(z_arc - z_gt), axis=1)
    err_k = np.mean(np.abs(z_klein - z_gt), axis=1)

    plt.figure(figsize=(11, 4))
    plt.plot(err_nn, label="NN")
    plt.plot(err_arc, label="Arc-step")
    plt.plot(err_k, label="Klein")
    plt.xlabel("angle index (0..359)")
    plt.ylabel("mean |error|")
    plt.title("Mean |error| per angle")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_heatmaps(z_gt, z_nn, z_arc, z_klein, mask, title_prefix=""):
    def show(im, ttl):
        plt.figure(figsize=(10, 4))
        plt.imshow(im.T, aspect="auto", origin="lower")
        plt.colorbar()
        plt.title(ttl)
        plt.xlabel("angle index")
        plt.ylabel("r index")
        plt.tight_layout()
        plt.show()

    show(z_gt,    f"{title_prefix}GT")
    show(mask,   f"{title_prefix}Mask")
    show(z_nn,   f"{title_prefix}NN")
    show(z_arc,  f"{title_prefix}Arc-step")
    show(z_klein,f"{title_prefix}Klein")

# =========================================================
# Evaluation
# =========================================================

def evaluate_case(model, sample, calib):
    r_grid = sample["r_grid"].numpy()
    rho = sample["rho"].numpy()
    z_gt = sample["z_target"].numpy()
    rk = sample["rk"].numpy()
    mask = sample["mask"].numpy()

    with torch.no_grad():
        z_nn = model(torch.tensor(rho, device=DEVICE)).cpu().numpy()

    A = rho.shape[0]
    z_arc = np.zeros_like(z_gt)
    z_klein = np.zeros_like(z_gt)

    for a in range(A):
        rA, zA = arc_step_fixed(rho[a], calib.f, calib.Lc)
        rK, zK = klein_like_z(rho[a], rk[a])

        z_arc[a] = np.interp(r_grid, rA, zA)
        z_klein[a] = np.interp(r_grid, rK, zK)

    return r_grid, z_gt, z_nn, z_arc, z_klein, mask

# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    A = 360
    K = 20
    Rr = 64

    calib = build_calib(K=K)

    train_ds = TinyPlacidoDataset(120, calib, A=A, K=K, Rr=Rr)
    val_ds = TinyPlacidoDataset(20, calib, A=A, K=K, Rr=Rr, jitter_px=0.0)

    loader = torch.utils.data.DataLoader(train_ds, batch_size=2, shuffle=True)

    model = EncoderDecoderNet(K, Rr).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(6):
        loss = train_epoch(model, loader, optim)
        print(f"Epoch {ep+1}: loss={loss:.6f}")

    r_grid, z_gt, z_nn, z_arc, z_klein, mask = evaluate_case(model, val_ds[3], calib)

    summary_print(z_gt, z_nn, z_arc, z_klein)
    plot_four_angles(r_grid, z_gt, z_nn, z_arc, z_klein)
    plot_per_angle_error(z_gt, z_nn, z_arc, z_klein)
    plot_heatmaps(z_gt, z_nn, z_arc, z_klein, mask, title_prefix="Validation: ")
