# -*- coding: utf-8 -*-
"""
Corneal Topography — Multi-angle (360) Neural (Polar U-Net) vs Classical
+ Physics-respecting regularization:
  - SmoothL1 data term
  - 2nd-derivative smoothness along radius r
  - 2nd-derivative smoothness along angle theta (periodic wrap)

Run: python this_file.py
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
        f=float(f),
        Lc=float(Lc),
        Lp=Lp.astype(np.float32),
        rings_mm=rings_mm.astype(np.float32),
    )

# =========================================================
# Surface model
# =========================================================

def asphere_sag(r, R, Q):
    r = np.asarray(r, dtype=np.float32)
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
        z = asphere_sag(np.array([r], dtype=np.float32), R, Q)[0]
        r2 = r + 1e-4
        z2 = asphere_sag(np.array([r2], dtype=np.float32), R, Q)[0]
        dzdr = (z2 - z) / (r2 - r)

        n = np.array([-dzdr, 1.0], dtype=np.float32)
        n /= np.linalg.norm(n) + 1e-12

        P = np.array([r, z], dtype=np.float32)
        S = np.array([Rpk, Lp_k], dtype=np.float32)
        C = np.array([0.0, Lc], dtype=np.float32)

        s = S - P; s /= (np.linalg.norm(s) + 1e-12)
        c = C - P; c /= (np.linalg.norm(c) + 1e-12)
        b = s + c; b /= (np.linalg.norm(b) + 1e-12)

        ang = np.arccos(np.clip(np.dot(n, b), -1.0, 1.0))
        if ang < best_ang:
            best_ang, best_r = ang, r
    return float(best_r)

# =========================================================
# Missing data helper
# =========================================================

def fill_missing_1d_with_linear(y, mask):
    y = np.asarray(y, np.float32)
    m = np.asarray(mask, np.float32)
    K = y.shape[0]

    if float(m.sum()) <= 0.0:
        return np.zeros_like(y, dtype=np.float32)

    idx = np.arange(K)
    obs = m > 0.5
    y_filled = y.copy()

    if int(obs.sum()) == 1:
        y_filled[:] = y[obs][0]
        return y_filled.astype(np.float32)

    y_filled[~obs] = np.interp(idx[~obs], idx[obs], y[obs]).astype(np.float32)
    return y_filled.astype(np.float32)

# =========================================================
# Dataset (360 angles per sample)
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
        R_theta = self._rng.uniform(self.R_range[0], self.R_range[1], size=(self.A,)).astype(np.float32)
        Q_theta = self._rng.uniform(self.Q_range[0], self.Q_range[1], size=(self.A,)).astype(np.float32)

        z_target = np.zeros((self.A, self.Rr), dtype=np.float32)
        for a in range(self.A):
            z_target[a] = asphere_sag(self.r_grid, float(R_theta[a]), float(Q_theta[a])).astype(np.float32)

        rk = np.zeros((self.A, self.K), dtype=np.float32)
        zk = np.zeros((self.A, self.K), dtype=np.float32)
        rho = np.zeros((self.A, self.K), dtype=np.float32)

        for a in range(self.A):
            R = float(R_theta[a])
            Q = float(Q_theta[a])
            for k in range(self.K):
                r_star = find_reflection_radius_simple(
                    float(self.calib.rings_mm[k]),
                    float(self.calib.Lp[k]),
                    float(self.calib.Lc),
                    R, Q, self.r_max,
                    n_grid=120
                )
                rk[a, k] = r_star
                zk[a, k] = asphere_sag(np.array([r_star], dtype=np.float32), R, Q)[0]
            rho[a] = forward_project_r_to_rho(rk[a], zk[a], float(self.calib.f), float(self.calib.Lc)).astype(np.float32)

        if self.jitter_px > 0:
            rho += self._rng.normal(0.0, self.jitter_px, size=rho.shape).astype(np.float32)

        mask = self._make_missing_mask()
        rho_obs = rho.copy()
        rho_obs[mask < 0.5] = 0.0

        return {
            "rho": torch.tensor(rho_obs, dtype=torch.float32),         # [A,K]
            "mask": torch.tensor(mask, dtype=torch.float32),           # [A,K]
            "z_target": torch.tensor(z_target, dtype=torch.float32),   # [A,Rr]
            "r_grid": torch.tensor(self.r_grid, dtype=torch.float32),  # [Rr]
            "rk": torch.tensor(rk, dtype=torch.float32),               # [A,K]
            "zk": torch.tensor(zk, dtype=torch.float32),               # [A,K]
            "R": torch.tensor(R_theta, dtype=torch.float32),           # [A]
            "Q": torch.tensor(Q_theta, dtype=torch.float32),           # [A]
        }

# =========================================================
# Optics diagram (one angle)
# =========================================================

def optics_diagram(sample, calib, angle_deg=45):
    A = sample["rho"].shape[0]
    a = int(angle_deg) % A

    rho = sample["rho"][a].numpy()
    mask = sample["mask"][a].numpy()
    zt = sample["z_target"][a].numpy()
    r = sample["r_grid"].numpy()
    rk = sample["rk"][a].numpy()
    zk = sample["zk"][a].numpy()
    R = float(sample["R"][a].item())
    Q = float(sample["Q"][a].item())

    ray_color = "0.75"

    plt.figure(figsize=(10, 10))
    plt.plot(r, zt, label="z(r)", lw=2)
    plt.scatter(calib.rings_mm, calib.Lp, s=35, label="Placido rings (r, Lp)")
    plt.scatter([0.0], [float(calib.Lc)], marker="x", s=60, color="k", label="Camera pinhole C")

    z_img = float(calib.Lc) + float(calib.f)
    plt.axhline(z_img, lw=1.0, alpha=0.6)
    plt.text(r[0] + 0.25, z_img + 0.05, "Image plane", fontsize=9)

    obs = mask > 0.5
    plt.scatter(-rho[obs], np.full(np.sum(obs), z_img, dtype=np.float32),
                marker="s", s=18, label="Observed image points I_k")
    plt.scatter(rk, zk, s=25, marker="^", label="Hit points P_k (r*, z*)")

    for k in range(len(rho)):
        if not obs[k]:
            continue
        r_ring, z_ring = float(calib.rings_mm[k]), float(calib.Lp[k])
        r_star, z_star = float(rk[k]), float(zk[k])
        rho_i = float(rho[k])
        plt.plot([r_ring, r_star, -rho_i],
                 [z_ring, z_star, z_img],
                 linestyle=":", lw=1.2, color=ray_color, alpha=0.9)

    plt.grid(True)
    plt.legend()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(f"Optics diagram (angle {angle_deg}°)\nR={R:.2f} mm, Q={Q:.2f}")
    plt.xlabel("r (mm) / ρ (sensor units)")
    plt.ylabel("z (mm)")
    plt.tight_layout()
    plt.show()

# =========================================================
# Arc-step baseline (per-angle)
# =========================================================

def arc_step_fixed(rho, f, Lc, smooth_window=3):
    rho = np.asarray(rho, np.float32)

    if smooth_window and smooth_window > 1:
        w = int(smooth_window) | 1
        pad = w // 2
        rho_p = np.pad(rho, (pad, pad), mode="edge")
        ker = np.ones(w, np.float32) / float(w)
        rho = np.convolve(rho_p, ker, mode="valid").astype(np.float32)

    r = rho * (Lc / f)
    r = np.maximum.accumulate(r).astype(np.float32)

    dr = np.diff(r)
    R_local = np.zeros_like(r)

    for i in range(1, len(r) - 1):
        R_local[i] = 0.5 * (dr[i] + dr[i - 1])
    R_local[0] = R_local[1]
    R_local[-1] = R_local[-2]

    R_local = np.clip(R_local, 6.5, 12.0).astype(np.float32)

    z = np.zeros_like(r)
    for i in range(len(r)):
        Ri = float(R_local[i])
        z[i] = Ri - np.sqrt(max(Ri * Ri - float(r[i]) * float(r[i]), 1e-6))

    z = -z
    z = z - z[0]
    return r.astype(np.float32), z.astype(np.float32)

# =========================================================
# Klein-like baseline (per-angle)
# =========================================================

def klein_like_z(rho, r_k, smooth_window=5):
    rho = np.asarray(rho, np.float32)
    r = np.asarray(r_k, np.float32)

    if smooth_window and smooth_window > 1:
        w = int(smooth_window) | 1
        pad = w // 2
        rho_p = np.pad(rho, (pad, pad), mode="edge")
        ker = np.ones(w, np.float32) / float(w)
        rho = np.convolve(rho_p, ker, mode="valid").astype(np.float32)

    drho_dr = np.gradient(rho, r + 1e-6)
    kappa = drho_dr / (r + 1e-6)

    dr = np.gradient(r)
    slope = np.cumsum(kappa * dr)
    z = np.cumsum(slope * dr)
    z = z - z[0]
    z = -z
    return r.astype(np.float32), z.astype(np.float32)

# =========================================================
# Neural model: Polar U-Net (Conv2d)
# =========================================================

class PolarUNet(nn.Module):
    def __init__(self, K, Rr, base=32):
        super().__init__()
        self.Rr = int(Rr)

        self.enc1 = nn.Sequential(
            nn.Conv2d(2, base, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.ReLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(base * 4, base * 4, 3, padding=1),
            nn.ReLU(),
        )

        self.mid = nn.Sequential(
            nn.Conv2d(base * 4, base * 4, 3, padding=1),
            nn.ReLU(),
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base * 4, base, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.ReLU(),
        )

        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, rho, mask):
        x = torch.stack([rho, mask], dim=1)  # [B,2,A,K]

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        m = self.mid(e3)

        d3 = self.dec3(m)
        d3 = _center_crop_like(d3, e2)
        d3 = torch.cat([d3, e2], dim=1)

        d2 = self.dec2(d3)
        d2 = _center_crop_like(d2, e1)
        d2 = torch.cat([d2, e1], dim=1)

        d1 = self.dec1(d2)

        z_k = self.out(d1).squeeze(1)  # [B,A,K]

        z = F.interpolate(
            z_k.unsqueeze(1),             # [B,1,A,K]
            size=(z_k.shape[1], self.Rr), # (A,Rr)
            mode="bilinear",
            align_corners=False
        ).squeeze(1)                     # [B,A,Rr]
        return z


def _center_crop_like(x, ref):
    _, _, H, W = x.shape
    _, _, Hr, Wr = ref.shape
    if H == Hr and W == Wr:
        return x
    top = max(0, (H - Hr) // 2)
    left = max(0, (W - Wr) // 2)
    return x[:, :, top:top + Hr, left:left + Wr]

# =========================================================
# Physics-respecting loss
# =========================================================

def second_diff_r(z):
    # z: [B,A,Rr] -> second difference along r (last axis)
    return z[:, :, 2:] - 2.0 * z[:, :, 1:-1] + z[:, :, :-2]


def second_diff_theta_periodic(z):
    # z: [B,A,Rr] -> periodic second diff along angle axis
    z_prev = torch.roll(z, shifts=1, dims=1)
    z_next = torch.roll(z, shifts=-1, dims=1)
    return z_next - 2.0 * z + z_prev


def loss_physics(pred, target, lam_r=1e-3, lam_theta=1e-3):
    data = F.smooth_l1_loss(pred, target)
    reg_r = torch.mean(second_diff_r(pred) ** 2)
    reg_t = torch.mean(second_diff_theta_periodic(pred) ** 2)
    return data + lam_r * reg_r + lam_theta * reg_t, data, reg_r, reg_t

# =========================================================
# Training
# =========================================================

def train_epoch(model, loader, optim, lam_r=1e-3, lam_theta=1e-3):
    model.train()
    loss_sum = 0.0
    for b in loader:
        rho = b["rho"].to(DEVICE)
        mask = b["mask"].to(DEVICE)
        zt = b["z_target"].to(DEVICE)

        pred = model(rho, mask)
        loss, data, rr, rt = loss_physics(pred, zt, lam_r=lam_r, lam_theta=lam_theta)

        optim.zero_grad()
        loss.backward()
        optim.step()
        loss_sum += float(loss.item())
    return loss_sum / max(1, len(loader))

# =========================================================
# Evaluation
# =========================================================

@torch.no_grad()
def predict_all(model, sample, calib, arc_smooth=3, klein_smooth=5):
    rho_t = sample["rho"].unsqueeze(0).to(DEVICE)
    mask_t = sample["mask"].unsqueeze(0).to(DEVICE)
    z_gt = sample["z_target"].numpy().astype(np.float32)
    r_grid = sample["r_grid"].numpy().astype(np.float32)
    rk = sample["rk"].numpy().astype(np.float32)

    z_nn = model(rho_t, mask_t)[0].cpu().numpy().astype(np.float32)

    A, K = sample["rho"].shape
    z_arc = np.zeros((A, len(r_grid)), dtype=np.float32)
    z_klein = np.zeros((A, len(r_grid)), dtype=np.float32)

    rho_np = sample["rho"].numpy().astype(np.float32)
    mask_np = sample["mask"].numpy().astype(np.float32)

    for a in range(A):
        rho_filled = fill_missing_1d_with_linear(rho_np[a], mask_np[a])

        rArc, zArc = arc_step_fixed(rho_filled, calib.f, calib.Lc, smooth_window=arc_smooth)
        z_arc[a] = np.interp(r_grid, rArc, zArc, left=zArc[0], right=zArc[-1]).astype(np.float32)

        rK, zK = klein_like_z(rho_filled, rk[a], smooth_window=klein_smooth)
        z_klein[a] = np.interp(r_grid, rK, zK, left=zK[0], right=zK[-1]).astype(np.float32)

    return z_gt, z_nn, z_arc, z_klein

# =========================================================
# Plotting + metrics
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
        ax.plot(r_grid, z_nn[a], "r", lw=2, label="NN (Polar U-Net)")
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
    plt.title("Mean |error| per angle (averaged over r)")
    plt.xlabel("angle index (0..359)")
    plt.ylabel("mean |z_pred - z_gt|")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_heatmaps(z_gt, z_nn, z_arc, z_klein, title_prefix=""):
    def show(im, ttl):
        plt.figure(figsize=(10, 4))
        plt.imshow(im, aspect="auto", origin="lower")
        plt.colorbar()
        plt.title(ttl)
        plt.xlabel("r index")
        plt.ylabel("angle index (0..359)")
        plt.tight_layout()
        plt.show()

    show(z_gt,    f"{title_prefix}GT  (z[angle, r])")
    show(z_nn,    f"{title_prefix}NN  (z[angle, r])")
    show(z_arc,   f"{title_prefix}Arc-step  (z[angle, r])")
    show(z_klein, f"{title_prefix}Klein  (z[angle, r])")

    show(np.abs(z_nn - z_gt),    f"{title_prefix}|NN - GT|")
    show(np.abs(z_arc - z_gt),   f"{title_prefix}|Arc - GT|")
    show(np.abs(z_klein - z_gt), f"{title_prefix}|Klein - GT|")

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

    train_ds = TinyPlacidoDataset(
        N=200, calib=calib, A=A, K=K, Rr=Rr,
        jitter_px=0.002,
        missing_prob=0.15,
        missing_block_prob=0.35,
        seed=1
    )
    val_ds = TinyPlacidoDataset(
        N=30, calib=calib, A=A, K=K, Rr=Rr,
        jitter_px=0.002,
        missing_prob=0.15,
        missing_block_prob=0.35,
        seed=999
    )

    loader = torch.utils.data.DataLoader(train_ds, batch_size=2, shuffle=True)

    model = PolarUNet(K=K, Rr=Rr, base=32).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    # regularization weights (tune if needed)
    lam_r = 2e-3
    lam_theta = 2e-3

    for ep in range(2):
        L = train_epoch(model, loader, optim, lam_r=lam_r, lam_theta=lam_theta)
        print(f"Epoch {ep+1}: loss={L:.6f}")

    sample = val_ds[3]

    # Optics diagram at one requested angle
    optics_diagram(sample, calib, angle_deg=45)

    z_gt, z_nn, z_arc, z_klein = predict_all(model, sample, calib)

    summary_print(z_gt, z_nn, z_arc, z_klein)

    r_grid = sample["r_grid"].numpy()
    plot_four_angles(r_grid, z_gt, z_nn, z_arc, z_klein, angles_deg=(45, 135, 225, 270))

    plot_per_angle_error(z_gt, z_nn, z_arc, z_klein)
    plot_heatmaps(z_gt, z_nn, z_arc, z_klein, title_prefix="Sample 3 — ")
