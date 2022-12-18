#### Preliminaries

We consider an episodic finite horizon Decision Process that consists of $\{\mathcal{X}_ h\}_ {h=1}^H,\mathcal{A},c$, $H,\rho,P$ where $\mathcal{X}_ h$ for $h\in[H]$ is a time-dependent $\mathcal{A}$ is a discrete action such space such that $|\mathcal{A}|=K$, $H$ is the horizon.

We assume the cost function $c:\mathcal{X}_ H\to\mathbb{R}$ is only defined at the last time step $H$, and $\rho \in\Delta(\mathcal{X}_ {h+1})$ for $h\in[H-1]$.

We assume that the cost function only depends on observations.

We assume that $\mathcal{X}_h$ is discrete. Also $|\mathcal{X}_h|$ could be easily extremely large and the algorithm that has a polynomial dependency on $|\mathcal{X}_h|$ should be considered as sample inefficient.

We assume that the cost is bounded for any sequence of observation, $c_H\leq 1$.

For a time-dependent policy $\pi:\{\pi_1,\dots,\pi_H\}$ with $\pi_{h} : \mathcal{X}_h\to\Delta(\mathcal{A})$, the value function $V_h^\pi:\mathcal{X}_h\to[0,1]$ is defined as:

$$
V_h^\pi(x_h)=\mathbb{E}[c(x_H)|a_i\sim\pi_i(\cdot|x_i),x_{i+1}\sim P_{x_i,a_i}],
$$

and state-action function $Q_ h^\pi(x_ h,a_ h)$ is defined as $Q_ h^\pi(x_ h,a_ h)=\mathbb{E}_ {x_ {h+1}\sim P_{x_ h,a_ h}}[V_ {h+1}^\pi (x_ {h+1})]$ with $V_ H^\pi(x)=c(x)$.

We denote $\mu_h^\pi$ as the distribution over $\mathcal{X}_h$ at time step $h$ following $\pi$.

Given $H$ policy classes $\{\Pi_1,\dots, \Pi_H\}$, the goal is to learn a $\pi=\{\pi_1,\dots,\pi_H\}$ with $\pi_h\in\Pi_h$, which minimizes the expected cost:

$$
J(\pi)=\mathbb{E}[c(x_H)|a_h\sum \pi_h(\cdot|x_h),x_{h+1}\sim P(\cdot|x_h,a_h)].
$$

Denote $\mathcal{F}_h\subset \{f:\mathcal{X}_h\to\mathbb{R}\}$ for $h\in[H]$.

We define a Bellman Operator $\Gamma_ h$ associated with the expert $\pi_ h^{\*}$ at time step $h$ as $\Gamma_ h:\mathcal{F}_ {h+1}\to \{ f: \mathcal{X}_ h\to\mathbb{R}\}$ where for any $x_ h\in\mathcal{X}_ h$, $f\in\mathcal{F}_ {h+1}$,

$$
(\Gamma_h f)(x_h) \triangleq \mathbb{E}_{a_h\sim\pi_h^* (\cdot|x_h),x_{h+1}\sim P_{x_h,a_h}}[f(x_{h+1})].
$$

#### Integral Probability Metrics (IPM) 

IPM is a family of distance measure on distributions: given two distributions $P_1$ and $P_2$ over $\mathcal{X}$, and a function class $\mathcal{F}$ containing real-value functions $F:\mathcal{X}\to\mathbb{R}$ and symmetric (e.g., $\forall f\in\mathcal{F}$, we have $-f\in\mathcal{F}$, IPM is defined as:

$$
\underset{f\in\mathcal{F}}{\sup} (\mathbb{E}_{x\sim P_1} [f(x)] - \mathbb{E}_{x\sim P_2} [f(x)]).
$$

We could obtain various popular distances by choosing the different classes of functions $\mathcal{F}$.

For instance IPM with $\mathcal{F}=\{f:||f||_ \infty \leq 1\}$ recovers Total Variation distance, IPM with $\mathcal{F}=\{f:||f||_ L \leq 1\}$ recovers Wasserstein distance, and IPM with $\mathcal{F}=\{f:||X||_ {\mathcal{H}} \leq 1\}$ with RKHS $\mathcal{H}$ reveals maximum mean discrepancy (MMD), which is the implemented method for the paper.
