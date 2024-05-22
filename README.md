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

Libraries: JAX.
