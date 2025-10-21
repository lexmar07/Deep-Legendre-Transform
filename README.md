# Deep Legendre Transform (DLT)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)
[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-purple.svg)](https://neurips.cc/)

> **A simple, scalable way to learn convex conjugates in high dimensions.**
> DLT trains a neural network to approximate the convex conjugate \(f^*\) of a differentiable convex function \(f\), using an *implicit Fenchel identity* that supplies exact training targets—no closed‑form \(f^*\) required.

---

## Table of Contents

* [Overview](#overview)
* [Mathematical Primer](#mathematical-primer)
* [Method (DLT) in One Look](#method-dlt-in-one-look)
* [Certificates: A‑Posteriori Error Estimator](#certificates-a-posteriori-error-estimator)
* [Applications](#applications)
* [Results at a Glance](#results-at-a-glance)
* [Minimal Working Example (PyTorch)](#minimal-working-example-pytorch)
* [Project Structure](#project-structure)
* [Citation](#citation)
* [License](#license)
* [Contact](#contact)
* [Acknowledgments](#acknowledgments)
* [FAQ](#faq)

---

## Overview

**Deep Legendre Transform (DLT)** is a learning framework for computing convex conjugates in high dimension.
Classic grid methods for \(f^*(y) = \sup_{x\in C}{\langle x,y\rangle - f(x)}\) suffer from the curse of dimensionality; smoothing methods still require costly integration loops. DLT avoids both by training on **exact targets** derived from the *implicit* Legendre–Fenchel identity:
\[
f^*(\nabla f(x)) = \langle x, \nabla f(x)\rangle - f(x).
\]

**Highlights**

* **Scales to high‑D:** Works with MLPs/ResNets/ICNNs/KANs; demonstrated up to (d=200).
* **Convex outputs (optional):** Use an ICNN to guarantee convexity of the learned \(g_\theta \approx f^*\).
* **No closed‑form dual needed:** Targets come from (f) and (\nabla f) only.
* **Built‑in validation:** An *unbiased* Monte‑Carlo estimator certifies \(L^2\) approximation error of \(g_\theta\) to \(f^*\).
* **Symbolic recovery:** With KANs, DLT can rediscover exact closed‑form conjugates in low dimension.

---

## Mathematical Primer

* **Legendre–Fenchel transform:**
  \[
  f^*(y) = \sup_{x\in C} {\langle x,y\rangle - f(x)},\quad y\in\mathbb{R}^d.
  \]
* **Legendre (gradient) form (on (D=\nabla f(C))):**
  \[
  f^*(y)= \langle (\nabla f)^{-1}(y),, y\rangle - f\big((\nabla f)^{-1}(y)\big).
  \]
* **Implicit Fenchel identity:**
  \[
  f^*(\nabla f(x)) = \langle x, \nabla f(x)\rangle - f(x),\quad x\in C.
  \]

---

## Method (DLT) 

Train a network (g_\theta: D\to\mathbb{R}) (e.g., MLP/ResNet/ICNN/KAN) by minimizing:
[
\min_{\theta}\ \sum_{x\in\mathcal{X}*{\text{train}}}
\big[g*\theta(\nabla f(x)) + f(x) - \langle x,\nabla f(x)\rangle\big]^2.
]

**Sampling in gradient space.** When \(\nabla f\) *distorts* \(C\) heavily, learn a lightweight inverse \(\Psi_\vartheta\) of \(\nabla f\) to sample desired distributions directly on (D) (uniform, Gaussian, localized, etc.), then map back (y\mapsto x=h_\vartheta(y)) for training.

**Convexity.** Choose \(g_\theta\) as an **ICNN** to ensure the learned \(g_\theta\) is convex (helpful for downstream optimization/OT/control).

---

## Certificates: A‑Posteriori Error Estimator

Let (X_1,\dots,X_n) be i.i.d. from a distribution (\mu) on (C), with (\nu = \mu\circ(\nabla f)^{-1}) on (D). Then
[
\underbrace{\frac{1}{n}\sum_{i=1}^{n}\big[g(\nabla f(X_i)) + f(X_i) - \langle X_i,\nabla f(X_i)\rangle\big]^2}_{\text{Monte‑Carlo estimator}}
\to \ |g-f^*|^2_{L^2(D,\nu)}.
]
This yields **unbiased** validation of (L^2) error even when (f^*) is not known in closed form.

---

## Applications

* **Hamilton–Jacobi PDEs (Hopf formula):**
  [
  u(x,t) = \big(g^* + t,H\big)^*(x).
  ]
  DLT approximates the time‑parameterized dual ((g^*+tH)) or directly ((g^*+tH)^*), often outperforming residual‑minimizing PINNs in (L^2) accuracy across (t) and (d).

* **Optimal transport / WGANs:** Learn convex potentials via a fast, direct conjugation primitive.

* **Symbolic regression (KANs):** Recover exact expressions for (f^*) (e.g., quadratic, negative log, negative entropy) with near‑machine precision residuals in 2D.

---

## Results at a Glance

**DLT vs. classical grid (Lucet LLT) at (N=10) grid points per dim (representative):**

| Dim (d) | Classical (grid/FFT/LLT)       | DLT (ResNet/ICNN)                 |
| ------: | ------------------------------ | --------------------------------- | 
|     2–6 | Fast & accurate on fine grids  | Matches the error                 | 
|    8–10 | Time/memory explode ((O(N^d))) | Trains in seconds–minutes         |
|  20–200 | Infeasible                     | Trains; low RMSE with ResNet      | 


**Architectures:** ResNet often gives the best approximation in high‑D; ICNN guarantees convexity (sometimes slightly higher error); KANs recover exact closed forms in 2D.

---

## Minimal Working Example (PyTorch)

Below is a tiny end‑to‑end demo of DLT on a **quadratic** (f(x)=\tfrac12|x|_2^2) (so (f^*=f)). It shows the core training loop using the implicit identity—no closed‑form (f^*) needed.

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

> **Note:** For general (f), replace `grad_f` with `torch.autograd.grad` on `f(x).sum()` (keeping `x.requires_grad_(True)`), or use your analytic gradient. For **convex** outputs, swap `g` for an **ICNN**.

---

## Project Structure

```
├── main_part/          # Core implementation and experiment scripts
├── appendix/           # Supplementary experiments, figures, extended tables
├── images/ images2/    # Figures used in README/paper
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

Swiss National Science Foundation — Grant No. **10003723**







