\section{Surrogate-Guided KAN (SG-KAN)}
\label{sec:sg-kan}

SG-KAN is a gradient-free training framework for Kolmogorov-Arnold Networks (KANs) that adapts the SWIM philosophy to KANs. Instead of constructing scalar weights from data pairs as in SWIM, it uses a Gaussian Process surrogate fitted to the training data to identify informative regions of the input domain and construct edge functions directly from GP posterior evaluations over those regions.

% =========================================================
\subsection{GP Surrogate Construction}
\label{sec:gp-surrogate-construction}
% =========================================================
The first stage fits a Gaussian Process to the training dataset
$\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$. Following the notation of
Section~\ref{sec:gp-surrogate-models}, the GP prior is defined as

\begin{equation}
    f \sim \mathcal{GP}\left(\mu_0(\mathbf{x}),\, k(\mathbf{x}, \mathbf{x}')\right)
    \label{eq:gp-prior}
\end{equation}

\noindent where $\mu_0(\mathbf{x})$ is the prior mean function and
$k(\mathbf{x}, \mathbf{x}')$ is the covariance kernel. In this work, two GP model variants are used: ExactGP and SparseGP. ExactGP is used for small datasets, while SparseGP is selected for larger datasets where the inversion of the full $N \times N$ kernel matrix becomes computationally expensive. The sparse approximation introduces a set of inducing points that summarize the training data, with the number of inducing points selected via hyperparameter optimization. For both variants, a constant prior mean is used, and the covariance function is either the RBF kernel or the Mat\'{e}rn $\nu = 5/2$ kernel, selected as a hyperparameter.
Hyperparameters are optimized by maximizing the exact log marginal likelihood for ExactGP and the variational ELBO for SparseGP. After fitting, the GP is frozen and used only for inference.

\noindent Given the training data, the GP posterior at any query point $\mathbf{x}^*$ is a Gaussian
distribution with posterior mean $\mu^*(\mathbf{x}^*)$ and posterior variance
$\sigma^{*2}(\mathbf{x}^*)$, as defined in Section~\ref{sec:gp-surrogate-models}. The
posterior mean provides a smooth, noise-filtered estimate of the target function. The
posterior variance quantifies the uncertainty of the target function estimation. 

% =========================================================
\subsection{Candidate Pair Sampling}
\label{sec:candidate-pair-sampling}
% =========================================================

Following the same strategy as SWIM~\cite{Bolager2023}, a large set of $M$ candidate pairs from the training is selected. 
% The second stage draws a large set of $M$ candidate pairs from the training data, following the same strategy as SWIM~\cite{Bolager2023}. 
For each candidate $m =
1, \ldots, M$, two distinct training points $\mathbf{x}^a_m$ and $\mathbf{x}^b_m$
are sampled uniformly from $\mathcal{D}$ with replacement. To guarantee that
$\mathbf{x}^a_m \neq \mathbf{x}^b_m$, the delta trick is used: a source index
$i$ is drawn uniformly from $\{0, \ldots, N-1\}$, an offset $\delta$ is drawn
uniformly from $\{1, \ldots, N-2\}$, and the target index is set to
$(i + \delta) \bmod N$. This ensures that each pair has distinct endpoints, while pairs may repeat across candidates due to sampling with replacement.

\noindent For each candidate pair $(\mathbf{x}^a_m, \mathbf{x}^b_m)$, a set of
$T$ interior points are constructed by linear interpolation along the segment
connecting the two endpoints:

\begin{equation}
    \mathbf{x}^t_m = \mathbf{x}^a_m + t \cdot (\mathbf{x}^b_m - \mathbf{x}^a_m),
    \qquad t \in \left\{\frac{1}{T+1}, \frac{2}{T+1}, \ldots, \frac{T}{T+1}\right\}
    \label{eq:interior-points}
\end{equation}

\noindent The endpoints are excluded from this set. This is because they are already used separately in the importance scoring: the numerator evaluates the GP posterior gradient at the endpoints, while the denominator accounts for their uncertainty as individual terms. Including them in the interior set would double-count their contribution. The interior points are used to compute the posterior means $\mu^*(\mathbf{x}^t_m)$ and posterior standard deviations $\sigma^*(\mathbf{x}^t_m)$ of the latent GP along the segment using the frozen GP model.

% =========================================================
\subsection{Importance Scoring}
\label{sec:importance-scoring}
% =========================================================

For each candidate pair $(\mathbf{x}^a_m, \mathbf{x}^b_m)$, an importance score is computed. The score shows how informative the corresponding segment is for constructing an edge function. It calculates the signal-to-uncertainty ratio based on the GP posterior. The numerator measures the variation of the target function across the segments, while the denominator penalizes segments where the posterior GP is uncertain. The denominator 
% is the same across both scoring variants and 
is defined as the total posterior uncertainty along the segment, including the endpoints:

\begin{equation}
    D_m = \sigma^*(\mathbf{x}^a_m) + \sum_{t=1}^{T} \sigma^*(\mathbf{x}^t_m)
    + \sigma^*(\mathbf{x}^b_m) + \varepsilon
    \label{eq:score-denominator}
\end{equation}

\noindent where $\varepsilon > 0$ is a small constant added for numerical stability and division by zero error.
% The two scoring variants differ only in how the numerator is defined.

\subsubsection{Gradient-Based Score (Score-G)}
\label{sec:gradient-based-score}

Score-G measures the change in the posterior gradient across the segment. The
posterior gradient $\nabla \mu^*(\mathbf{x})$ is evaluated at both endpoints using the GP posterior mean. The numerator is the
$\ell_\infty$ norm of the gradient difference:

\begin{equation}
    N^G_m = \left\|\nabla \mu^*(\mathbf{x}^a_m) - \nabla \mu^*(\mathbf{x}^b_m)
    \right\|_\infty
    \label{eq:score-g-numerator}
\end{equation}

\noindent This quantity measures how much the gradient of the posterior mean changes between the two endpoints, which approximates the curvature of $\mu^*$ along the segment. Pairs spanning high-curvature regions of the posterior mean receive high scores, since an edge function placed in such a region captures more of the target function's nonlinear structure. The full Score-G for
candidate pair $m$ is then:

\begin{equation}
    s^G_m = \frac{N^G_m}{D_m} =
    \frac{\left\|\nabla \mu^*(\mathbf{x}^a_m) -
    \nabla \mu^*(\mathbf{x}^b_m)\right\|_\infty}
    {\sigma^*(\mathbf{x}^a_m) + \sum_{t=1}^{T}
    \sigma^*(\mathbf{x}^t_m) + \sigma^*(\mathbf{x}^b_m) + \varepsilon}
    \label{eq:score-g}
\end{equation}

\noindent The scores are normalized to form a probability distribution over the $M$
candidate pairs:

\begin{equation}
    p^G_m = \frac{s^G_m}{\sum_{m'=1}^{M} s^G_{m'}}, \qquad m = 1, \ldots, M
    \label{eq:score-g-probs}
\end{equation}

\noindent This distribution is used in the next stage to select $K$ pairs
proportionally to their importance.

\noindent Score-G exploits the analytic gradient field provided by the GP surrogate. Since both the RBF and Mat\'{e}rn kernels are differentiable with respect to their inputs, the gradient $\nabla \mu^*(\mathbf{x})$ is available in closed form and is an inherent property of the fitted GP model. As a result, SG-KAN obtains exact gradient information at any point in the input domain directly from the GP posterior.

% =========================================================
\subsection{Edge Function Construction}
\label{sec:edge-function-construction}
% =========================================================

Once the probability distribution over candidate pairs is defined, $K$ pairs are
selected by sampling proportionally to $p^G_m$. Sampling is performed with
replacement, so the same pair can be selected multiple times if its score is
sufficiently high. Let $\{(\mathbf{x}^a_i, \mathbf{x}^b_i)\}_{i=1}^{K}$ denote
the selected pairs.

\noindent For each selected pair $i$, a dense grid of $S$ points is placed along
the segment connecting $\mathbf{x}^a_i$ and $\mathbf{x}^b_i$:

\begin{equation}
    \mathbf{x}^s_i = \mathbf{x}^a_i + s \cdot (\mathbf{x}^b_i - \mathbf{x}^a_i),
    \qquad s \in \left\{0, \frac{1}{S-1}, \ldots, 1\right\}
    \label{eq:dense-grid}
\end{equation}

\noindent The edge function $\phi_i$ is defined as the GP posterior mean evaluated
at these $S$ points:

\begin{equation}
    \phi_i\!\left(\mathbf{x}^s_i\right) = \mu^*\!\left(\mathbf{x}^s_i\right),
    \qquad s = 0, \ldots, S-1
    \label{eq:edge-function}
\end{equation}

\noindent This gives a lookup table of $S$ input-output pairs for each edge
function. To evaluate $\phi_i$ at an arbitrary training or test point $\mathbf{x}$,
the scalar input to edge $i$ is projected onto the segment and the corresponding
output is obtained by linear interpolation from the lookup table. When $d = 1$, the
projection reduces to the raw input value $x$, and the interpolation is performed
directly over the 1D lookup table.

\noindent This construction is entirely gradient-free and non-iterative. The edge
functions are determined in a single pass of GP posterior evaluations, with no
optimization of any kind.

\noindent Unlike the standard KAN architecture, where each node sums $d$ separate univariate edge functions operating on individual input dimensions, each SG-KAN hidden unit evaluates a single univariate function along a learned projection direction in the input space. This projection-based construction is structurally closer to random feature models such as SWIM and ELM, but differs in that the projection directions and edge functions are constructed from GP posterior information rather than sampled randomly.

% =========================================================
\subsection{Output Layer}
\label{sec:output-layer}
% =========================================================

After constructing all $K$ edge functions, each training point $\mathbf{x}_i$ is
mapped to a feature vector by evaluating all edge functions at that point. This
produces the feature matrix $\mathbf{H} \in \mathbb{R}^{N \times K}$, where entry
$H_{ij} = \phi_j(\mathbf{x}_i)$ is the output of edge function $j$ at training
point $i$. A bias column of ones is appended to give
$\tilde{\mathbf{H}} \in \mathbb{R}^{N \times (K+1)}$.

\noindent The output layer weights $\mathbf{w} \in \mathbb{R}^{K+1}$ are then
estimated by ordinary least squares:

\begin{equation}
    \mathbf{w} = \left(\tilde{\mathbf{H}}^\top \tilde{\mathbf{H}}\right)^{-1}
    \tilde{\mathbf{H}}^\top \mathbf{y}
    \label{eq:ols-output}
\end{equation}

\noindent Since all edge functions are fixed before this step, the optimization
problem is convex and has a unique global solution. The final prediction for any
input $\mathbf{x}$ is:

\begin{equation}
    \hat{f}(\mathbf{x}) = \tilde{\mathbf{h}}(\mathbf{x})^\top \mathbf{w}
    \label{eq:final-prediction}
\end{equation}

\noindent where $\tilde{\mathbf{h}}(\mathbf{x}) \in \mathbb{R}^{K+1}$ is the
feature vector obtained by evaluating all edge functions at $\mathbf{x}$ and
appending a one. In practice, Eq.~\eqref{eq:ols} is solved using a numerically
stable least squares routine rather than the explicit normal equations.