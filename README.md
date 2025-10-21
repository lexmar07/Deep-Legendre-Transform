# Deep Legendre Transform (DLT)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)
[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-purple.svg)](https://neurips.cc/)

> **A simple, scalable way to learn convex conjugates in high dimensions.**

DLT trains a neural network to approximate the convex conjugate $f^\ast$ of a differentiable convex function $f$, using an *implicit Fenchel identity* that supplies exact training targets—no closed‑form $f^\ast$ required.

---

## Table of Contents

* [Overview](#overview)
* [Mathematical Primer](#mathematical-primer)
* [Method (DLT) in One Look](#method-dlt-in-one-look)
* [Approximate Inverse Sampling](#approximate-inverse-sampling)
* [Certificates: A‑Posteriori Error Estimator](#certificates-a-posteriori-error-estimator)
* [Applications](#applications)
* [Results at a Glance](#results-at-a-glance)
* [Minimal Working Example (PyTorch)](#minimal-working-example-pytorch)
* [Project Structure](#project-structure)
* [Citation](#citation)
* [License](#license)
* [Contact](#contact)
* [Acknowledgments](#acknowledgments)

---

## Overview

**Deep Legendre Transform (DLT)** is a learning framework for computing convex conjugates in high dimensions.
Classic grid methods for

$f^\ast(y) = \sup_{x\in C}{\langle x,y\rangle - f(x)}$

suffer from the curse of dimensionality; while sup-smoothing methods still require costly integration loops. DLT avoids both by training on **exact targets** derived from the *implicit* Legendre–Fenchel identity:

$f^\ast(\nabla f(x)) = \langle x, \nabla f(x)\rangle - f(x)$

**Highlights**

* **Scales to high‑D:** Works with MLPs / ResNets / ICNNs / KANs; demonstrated up to $d=200$.
* **Convex outputs (optional):** Use an ICNN to guarantee convexity of the learned $g_\theta \approx f^\ast$.
* **No closed‑form dual needed:** Targets come from $f$ and $\nabla f$ only.
* **Built‑in validation:** A Monte‑Carlo estimator certifies $L^2$ approximation error of $g_\theta$ to $f^\ast$.
* **Symbolic recovery:** With KANs, DLT can rediscover exact closed‑form conjugates in low dimension.

---

## Mathematical Primer

**Legendre–Fenchel transform**

$f^\ast(y) = \sup_{x\in C}{\langle x,y\rangle - f(x)}, \quad y\in\mathbb{R}^d.$

**Legendre (gradient) form on $D=\nabla f(C)$**

$f^\ast(y) = \big\langle (\nabla f)^{-1}(y), y \big\rangle -  f\big((\nabla f)^{-1}(y)\big).$

**Implicit Fenchel identity**

$f^\ast(\nabla f(x)) = \langle x,\nabla f(x)\rangle - f(x), \quad x\in C.$

---

## Method (DLT) in One Look

Train a network $g_\theta : D \to \mathbb{R}$ (e.g., MLP / ResNet / ICNN / KAN) by minimizing

$\min_{\theta} \mathbb{E}_{X\sim \mu}  [ g^{\theta}  (\nabla f(X)) + f(X) - \langle X,\nabla f(X)\rangle ]^2$

or empirically,

$\min_{\theta}\frac{1}{n}\sum_{i=1}^{n} \left[ g^\theta(\nabla f(x_i)) + f(x_i) - \langle x_i,\nabla f(x_i)\rangle \right]^2$

**Sampling in gradient space.** When $\nabla f$ *distorts* $C$ heavily, you can sample directly on $D$ (uniform, Gaussian, localized, etc.) and map back to $C$ with an approximate inverse $\Psi_{\theta} \approx (\nabla f)^{-1}$; see [Approximate Inverse Sampling](#approximate-inverse-sampling).

**Convexity.** Choose $g_\theta$ as an **ICNN** to ensure the learned $g_\theta$ is convex (useful for downstream optimization/OT/control).

---

## Approximate Inverse Sampling

When $Y=\nabla f(X)$ with $X\sim\mu$ is highly concentrated or poorly covers $D$, training on $Y$ may be imbalanced. **Approximate inverse sampling** fixes this by learning a map
$h_\varphi : D \to C \quad\text{with}\quad \nabla f\big(h_\varphi(y)\big) \approx y,$
so we can first sample $Y\sim \nu^\dagger$ (a *desired* distribution on $D$: uniform, Gaussian, stratified, etc.), set $X=h_\varphi(Y)$, and then train DLT on $(Y,X)$.

**Using $\Psi_{\theta}$ in DLT.**

1. Sample $Y \sim \nu^\dagger$ on $D$.
2. Set $X = \Psi_{\theta}(Y)$.
3. Form targets $T(Y) = \langle X, Y\rangle - f(X)$ and train $g_\theta(Y)$ to match $T(Y)$.

* **Architectures for $\Psi_{\theta}$:**

  * MLP with spectral normalization (simple, fast).
  * Monotone/triangular flows (e.g., monotone splines per dimension) when $C$ is box‑shaped.
  * i‑ResNets / coupling‑flow blocks if strict invertibility on $D$ is desired.

**Minimal pseudocode (PyTorch‑style)**

```python
# 1) Learn approximate inverse Psi_theta
for step in range(T_inv):
    y = sample_from_nu_dagger(batch, D)     # desired coverage on D
    x_hat = Psi_theta(y)
    loss_inv = ((grad_f(x_hat) - y)**2).mean()
    loss_inv += lambda_C * barrier_C(x_hat) + lambda_lip * lipschitz_penalty(Psi_theta)
    update(theta_inv, loss_inv)

# 2) Train DLT using inverse-sampled pairs
for step in range(T_dlt):
    y = sample_from_nu_dagger(batch, D)
    x = Psi_theta(y).detach()
    target = (x * y).sum(dim=1, keepdim=True) - f(x)
    loss = ((g_theta(y) - target)**2).mean()
    update(theta, loss)
```

---

## Certificates: A‑Posteriori Error Estimator

Let $X_1,\dots,X_n$ be i.i.d. from a distribution $\mu$ on $C$, with $\nu = \mu\circ(\nabla f)^{-1}$ on $D$. Then
$\frac{1}{n}\sum_{i=1}^{n} \left[ g(\nabla f(X_i)) + f(X_i) - \langle X_i,\nabla f(X_i)\rangle \right]^2 \xrightarrow[n\to\infty]{} \|g - f^\ast\|_{L^2(D,\nu)}^2$

This provides a straightforward Monte‑Carlo certificate of $L^2$ error even when $f^\ast$ has no closed form.

---

## Applications

**Hamilton–Jacobi PDEs (Hopf formula):**

$u(x,t) = \big(g^\ast + t H\big)^\ast(x).$

DLT approximates the time‑parameterized dual $g^\ast+tH$ (or directly $(g^\ast+tH)^\ast$), and often outperforms residual‑minimizing PINNs in $L^2$ across $t$ and $d$.

**Optimal transport / WGANs:** Learn convex potentials, see here and here.

**Symbolic regression (KANs):** Recover exact expressions for $f^\ast$ (e.g., quadratic, negative log/entropy) with near‑machine‑precision residuals in low‑$d$.

---

## Results at a Glance

**DLT vs. classical grid (Lucet LLT) at $N=10$ grid points per dim (representative):**

| Dim $d$ | Classical (grid/FFT/LLT)               | DLT (ResNet/ICNN)         |
| ------: | -------------------------------------- | ------------------------- |
|     2–6 | Fast & accurate on fine grids          | Matches the error         |
|    8–10 | Time/memory explode $\mathcal{O}(N^d)$ | Trains in seconds–minutes |
|  20–200 | Infeasible                             | Can be trained to low RMSE|

**Architectures:** ResNet often gives the best approximation in high‑$d$; ICNN guarantees convexity (sometimes slightly higher error); KANs recover exact closed forms in 2D.

---

## Minimal Working Example (PyTorch)

Below is a tiny end‑to‑end demo of DLT on a **quadratic** $f(x)=\tfrac12|x|_2^2$ (so $f^\ast=f$). It shows the core training loop using the implicit identity—no closed‑form $f^\ast$ needed.

```python
# Minimal DLT demo (PyTorch) — quadratic example
# pip install torch
import torch, math

# Problem setup
d = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

# Define f and its gradient (quadratic)
def f(x):                      # x: (batch, d)
    return 0.5 * (x**2).sum(dim=1, keepdim=True)  # (batch, 1)

def grad_f(x):
    return x                   # ∇f(x) = x

# Approximator g_theta: R^d -> R (aims at f*)
g = torch.nn.Sequential(
    torch.nn.Linear(d, 128),
    torch.nn.GELU(),
    torch.nn.Linear(128, 128),
    torch.nn.GELU(),
    torch.nn.Linear(128, 1),
).to(device)

opt = torch.optim.Adam(g.parameters(), lr=1e-3)

def dlt_loss(x):
    y = grad_f(x)                          # (batch, d)
    target = (x * y).sum(dim=1, keepdim=True) - f(x)  # <x,∇f(x)> - f(x)
    pred = g(y)                             # g_theta(∇f(x))
    return ((pred - target)**2).mean()

# Training
for step in range(2000):
    x = torch.randn(4096, d, device=device)  # sample x ~ N(0,I)
    loss = dlt_loss(x)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 200 == 0:
        print(f"step {step:4d} | loss ~ L2 error^2: {loss.item():.3e}")

# A‑posteriori certificate on held‑out data
with torch.no_grad():
    x = torch.randn(8192, d, device=device)
    y = grad_f(x)
    target = (x * y).sum(dim=1, keepdim=True) - f(x)
    pred = g(y)
    mse = ((pred - target)**2).mean().sqrt().item()  # RMSE certificate
print(f"Certified RMSE on ∇f(C): {mse:.3e}")
```

> **Note:** For general $f$, replace `grad_f` with `torch.autograd.grad` on `f(x).sum()` (keeping `x.requires_grad_(True)`), or use your analytic gradient. For **convex** outputs, swap `g` for an **ICNN**.

---

## Project Structure

```
├── main_part/          # Core implementation and experiment scripts
├── appendix/           # Supplementary experiments, figures, extended tables
├── images              # Figures used in paper
├── LICENSE             # Apache 2.0
└── README.md
```

---

## Citation

If you use DLT in your research, please cite:

```bibtex
@inproceedings{minabutdinov2025deep,
  title     = {Deep Legendre Transform},
  author    = {Minabutdinov, Aleksey and Cheridito, Patrick},
  booktitle = {NeurIPS},
  year      = {2025}
}
```

---

## License

This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

---

## Contact

**Aleksey Minabutdinov** — [aminabutdinov@ethz.ch](mailto:aminabutdinov@ethz.ch) (ETH Zurich)
**Patrick Cheridito** — [patrickc@ethz.ch](mailto:patrickc@ethz.ch) (ETH Zurich)

---

## Acknowledgments

Swiss National Science Foundation — Grant No. **10003723**
