# Deep Legendre Transform

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-purple.svg)](https://neurips.cc/)

## 📖 Overview

This repository contains the implementation of **Deep Legendre Transform (DLT)**, a novel deep learning algorithm for computing convex conjugates of differentiable convex functions. Our method addresses a fundamental operation in convex analysis with applications across optimization, control theory, physics, and economics.

The Legendre–Fenchel transform of a function f: C → ℝ is defined as:

f*(y) = sup{x∈C} {⟨x,y⟩ - f(x)}

While traditional numerical methods suffer from the curse of dimensionality, our neural network-based method scales efficiently to high dimensions through an implicit Fenchel formulation.

## 🔬 Key Innovations

### 1. Implicit Legendre Formulation
We leverage the identity: f*(∇f(x)) = ⟨x, ∇f(x)⟩ - f(x)

This allows training neural networks with exact target values on training points.

### 2. A Posteriori Error Estimates
Our method provides rigorous L²-approximation guarantees without knowing the true conjugate.

### 3. Scalability to High Dimensions
- Successfully tested up to d=200 dimensions
- Orders of magnitude faster than classical grid-based methods

### 4. Exact Solutions via Symbolic Regression
Using Kolmogorov–Arnold Networks (KANs), we can recover exact closed-form expressions.

## 🚀 Quick Start
```bash
git clone https://github.com/lexmar07/Deep-Legendre-Transform.git
cd Deep-Legendre-Transform
pip install -r requirements.txt
```

## 📝 Citation
```bibtex
@inproceedings{minabutdinov2025deep,
  title={Deep Legendre Transform},
  author={Minabutdinov, Aleksey and Cheridito, Patrick},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

## 🏗️ Repository Structure
```
Deep-Legendre-Transform/
├── appendix/           # Supplementary materials
├── main_part/          # Main implementation
├── README.txt          # Original readme
├── README.md          # This file
└── LICENSE            # Apache 2.0 License
```

## 📧 Contact

**Aleksey Minabutdinov** (Corresponding Author)
- Email: aminabutdinov@ethz.ch
- Center of Economic Research and RiskLab, ETH Zurich

**Patrick Cheridito**
- Email: patrickc@ethz.ch
- Department of Mathematics and RiskLab, ETH Zurich

## 🙏 Acknowledgments

Supported by Swiss National Science Foundation (Grant No. 10003723).

---
**NeurIPS 2025** | Camera-ready version accepted
