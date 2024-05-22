# Deep-Legendre-Transform
Calculates Legendre Transform using ICNNs; illustrates approximate inverse sampling
Convex conjugation plays an important role in convex duality theory.
For a subset $C \subseteq  \mathbb{R} ^d$, let us denote by
$\delta_C \colon \mathbb{R} ^d \to \mathbb{R}  \cup \{+\infty\}$ the indicator function
given by
$$
\delta_C(x) := \begin{cases} 0 & \mbox{ if } x \in C\\
+ \infty & \mbox{ if } x \notin C.
\end{cases}
$$
For a function $f \colon \mathbb{R} ^d \to \mathbb{R} \cup \{+ \infty\}$, we denote by
$f_C^* \colon  \mathbb{R}  \cup \{+ \infty\}$ the convex conjugate of $f + \delta_C$ given by
$$
f_C^*(y) := \sup_{x \in C} { \langle x , y\rangle - f(x) }.
$$

**For example,** Hamiltonian in economic applications can be computed

$$H(p) = \max_c\{ p^Tc + u(c)\},$$

where $u$ is the (concave) utility function and $c$ stands for the consumption bundle.

Since function $f$ is assumed convex, there are two possible **direct** approaches:

1. Calculate FOCs (generally, subradients) of $f$ + solve corresponding equations

2. Realize maximization procedure directly: use stochastic gradient descent to evaluate $f_C^*(y)$.

These approaches are inefficient. 


**Idea 3.** In the special case, where $f \colon C \to \R$ is a differentiable convex function 
on an open convex set $C \subseteq R^d$, one has  
$$
f^*(y) = \langle(\nabla f)^{-1}(y), y\rangle  - f((\nabla f)^{-1}(y)) 
$$
for all $y \in D = \nabla f(C).$

The right side of the equation is called Legendre transform. In fact, it follows from the Fenchel--Young inequality 
that if $f$ is differentiable and convex, $\langle x, y\rangle - f(x)$ is the same for all 
$x$ in the pre-image $(\nabla f)^{-1}(y)$. So $(\nabla f)^{-1}(y)$ 
can be substituted with any $x \in C$ satisfying $\nabla f(x) = y$.



## Simplest example:

We first consider the quadratic function $f \colon R^d \to R$ given by
$$
f(x) = \langle x , x\rangle  = x^2_1 + \dots + x^2_d.
$$
It is well known (and can easily be checked) that $f^* = f/4$. 

To approximate $f^*$ numerically, we generated $X_{test}$ by sampling 
approximately $10^4$ points $x \in R^d$ from a $d$-dimensional standard normal 
distribution $N(0, I_d)$. 

## Sampling in the gradient space from a desired distribution.

Since our goal is to approximate $f^*(y)$, in some cases, it might be beneficial to train the model on samples $\{y_i\}$ that have some desired distribution $\nu$. For example, $\nu$ can be a uniform distribution on some compact subset $K$ of $D$. However, if we sample $d$-dimensional $x$ uniformly, then the gradient $\nabla f(x)$ will (generally) not have a $d$-dimensional uniform distribution. 
The desired $x$-sample can be obtained as $\{(\nabla f)^{-1}(z)\}$, where $(\nabla f)^{-1}(z)$ is some element from the preimage, and $z$ is sampled according to $\nu$ on $K$. 
In some cases, the inverse mapping $(\nabla f)^{-1}$ exists and can be found analytically; however, generally, we have to rely on approximations.

To address this problem, we introduce the concept which we call \textit{the approximate inverse mapping}\footnote{We believe this unsupervised learning approach might be useful across various disciplines where computing inverse mappings is required.}. The idea is as follows:
Let $\Psi:\mathbb{R}^d \to \mathbb{R}^d$ be some mapping. We can learn an "autoencoder type" architecture 
$$
x \xrightarrow{\mathfrak{e}} \Psi(x) \xrightarrow{\mathfrak{d}} x,
$$
where the encoder $\mathfrak{e}$ is set equal to $\Psi$. The decoder part, $\mathfrak{d}: \psi \mapsto x$, will provide an approximate inverse to the mapping $\Psi$ if $\Psi$ is invertible, and an element from the preimage otherwise. As a result, one can sample from the distribution $\nu$ in the $\{\psi\}$ space so that $\mathfrak{d}(\psi)$ generates $\{x\}$-sample, which will (approximately) have the desired property.




## Some examples:


* Quadratic
$f(x) = \frac12 x^T P x +b^Tx+c$, where $P$ is a positive semi-definite matrix, $x\in \mathbb{R}^n.$ Then conjugate is

$$f^*(y) = \frac12(y-b)^TP^{-1}(y-b) - c$$
(if $P$ is positive-definite, otherwise use pseudoinverse $P^\dagger = (AA^T)^{-1}A $).



* Quadratic over linear
$f(x, y) =  x^T P x/y$, where $(x, y)\in \mathbb{R}^n\times \mathbb{R}_{++}$


--not known



* Negative logarithm

$f(x) = -\sum\limits_{i=1}^n \log(x_i)$

$$f^*(y) = - \sum\limits_{i=1}^n\log(-y_i) -n$$

* Negative entropy
$f(x) = \sum_i x_i\log(x_i), x\in \mathbb{R}^n_+$

$f^*(y) = \sum\limits_{i=1}^ne^{y_i}-1$

Libraries: JAX.
