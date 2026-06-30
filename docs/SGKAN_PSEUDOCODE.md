# SG-KAN Pseudocode

This document provides clear pseudocode for each stage of the Surrogate-Guided KAN (SG-KAN) algorithm in LaTeX format for thesis inclusion.

## 1. GP Surrogate Construction

```latex
\begin{algorithm}
\caption{Gaussian Process Surrogate Fitting}
\label{alg:gp-surrogate}
\begin{algorithmic}[1]
\Require Training data $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$
\Require Kernel type (RBF or Matérn), number of inducing points (if sparse)
\Ensure Fitted GP model $\mathcal{M}$ with frozen hyperparameters

\State Initialize GP prior: $f \sim \mathcal{GP}(\mu_0(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$
\State Optimize hyperparameters via marginal likelihood (ExactGP) or ELBO (SparseGP)
\State \textbf{return} Frozen GP model $\mathcal{M}$ for inference only
\end{algorithmic}
\end{algorithm}
```

## 2. Candidate Pair Sampling

```latex
\begin{algorithm}
\caption{Candidate Pair Sampling with Interior Points}
\label{alg:candidate-sampling}
\begin{algorithmic}[1]
\Require Training data $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$
\Require Number of candidate pairs $M$, number of interior points $T$
\Ensure Candidate pairs $\{(\mathbf{x}^a_m, \mathbf{x}^b_m)\}_{m=1}^{M}$ with interior points

\For{$m = 1$ to $M$}
    \State Draw source index $i \sim \text{Uniform}(\{0, \ldots, N-1\})$
    \State Draw offset $\delta \sim \text{Uniform}(\{1, \ldots, N-2\})$
    \State Set target index $j = (i + \delta) \bmod N$
    \State Set $\mathbf{x}^a_m = \mathbf{x}_i$, $\mathbf{x}^b_m = \mathbf{x}_j$
    
    \For{$t = 1$ to $T$}
        \State Interpolate: $\mathbf{x}^t_m = \mathbf{x}^a_m + \frac{t}{T+1} \cdot (\mathbf{x}^b_m - \mathbf{x}^a_m)$
        \State Query GP: obtain $\mu^*(\mathbf{x}^t_m)$ and $\sigma^*(\mathbf{x}^t_m)$
    \EndFor
\EndFor

\State \textbf{return} Candidate pairs and interior point evaluations
\end{algorithmic}
\end{algorithm}
```

## 3. Importance Scoring (Score-G)

```latex
\begin{algorithm}
\caption{Gradient-Based Importance Scoring (Score-G)}
\label{alg:score-g}
\begin{algorithmic}[1]
\Require Candidate pairs $\{(\mathbf{x}^a_m, \mathbf{x}^b_m)\}_{m=1}^{M}$ with interior points
\Require Frozen GP model $\mathcal{M}$, small constant $\varepsilon > 0$
\Ensure Normalized importance distribution $\mathbf{p} = \{p^G_m\}_{m=1}^{M}$

\For{$m = 1$ to $M$}
    \State \textit{Numerator:} Query GP gradient at endpoints
    \State $\mathbf{g}_a = \nabla \mu^*(\mathbf{x}^a_m)$, $\mathbf{g}_b = \nabla \mu^*(\mathbf{x}^b_m)$
    \State $N^G_m = \|\mathbf{g}_a - \mathbf{g}_b\|_\infty$
    
    \State \textit{Denominator:} Sum posterior uncertainty along segment
    \State $D_m = \sigma^*(\mathbf{x}^a_m) + \sum_{t=1}^{T} \sigma^*(\mathbf{x}^t_m) + \sigma^*(\mathbf{x}^b_m) + \varepsilon$
    
    \State \textit{Score:} $s^G_m = N^G_m / D_m$
\EndFor

\State \textit{Normalize:} $p^G_m = s^G_m / \sum_{m'=1}^{M} s^G_{m'}$ for all $m$
\State \textbf{return} Importance distribution $\mathbf{p}$
\end{algorithmic}
\end{algorithm}
```

## 4. Edge Function Construction

```latex
\begin{algorithm}
\caption{Edge Function Construction via Sampling and Lookup}
\label{alg:edge-functions}
\begin{algorithmic}[1]
\Require Importance distribution $\mathbf{p}$ over $M$ candidate pairs
\Require Frozen GP model $\mathcal{M}$, number of edge functions $K$, lookup table size $S$
\Ensure Edge functions $\{\phi_1, \ldots, \phi_K\}$ as lookup tables

\For{$i = 1$ to $K$}
    \State Sample pair index $m_i \sim \mathbf{p}$ (with replacement)
    \State Retrieve pair $(\mathbf{x}^a_i, \mathbf{x}^b_i)$
    
    \State \textit{Dense grid along segment:}
    \For{$s = 0$ to $S-1$}
        \State $\mathbf{x}^s_i = \mathbf{x}^a_i + \frac{s}{S-1} \cdot (\mathbf{x}^b_i - \mathbf{x}^a_i)$
        \State Query GP: $\phi_i(s) = \mu^*(\mathbf{x}^s_i)$
    \EndFor
\EndFor

\State \textbf{return} Edge function lookup tables $\{\phi_1, \ldots, \phi_K\}$
\end{algorithmic}
\end{algorithm}
```

## 5. Output Layer via Ordinary Least Squares

```latex
\begin{algorithm}
\caption{Output Layer Weight Computation}
\label{alg:output-layer}
\begin{algorithmic}[1]
\Require Training data $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$
\Require Fixed edge functions $\{\phi_1, \ldots, \phi_K\}$
\Ensure Output weights $\mathbf{w} \in \mathbb{R}^{K+1}$

\State \textit{Feature matrix construction:}
\For{$i = 1$ to $N$}
    \For{$j = 1$ to $K$}
        \State $H_{ij} = \phi_j(\mathbf{x}_i)$ (evaluate edge function $j$ at point $i$)
    \EndFor
\EndFor

\State Append bias column: $\tilde{\mathbf{H}} = [\mathbf{H} \mid \mathbf{1}]$

\State \textit{Solve OLS:} $\mathbf{w} = (\tilde{\mathbf{H}}^\top \tilde{\mathbf{H}})^{-1} \tilde{\mathbf{H}}^\top \mathbf{y}$

\State \textbf{return} Weight vector $\mathbf{w}$
\end{algorithmic}
\end{algorithm}
```

## 6. Complete SG-KAN Training Pipeline

```latex
\begin{algorithm}
\caption{SG-KAN: Complete Training Pipeline}
\label{alg:sgkan-full}
\begin{algorithmic}[1]
\Require Training data $\mathcal{D}$, hyperparameters: $M$ (candidates), $T$ (interior points), $K$ (edges), $S$ (lookup size)
\Ensure Trained SG-KAN model with prediction function $\hat{f}$

\State \textbf{Stage 1:} Fit GP surrogate to $\mathcal{D}$ (Algorithm~\ref{alg:gp-surrogate})
\State $\mathcal{M} \gets$ Frozen GP model

\State \textbf{Stage 2:} Sample $M$ candidate pairs with interior points (Algorithm~\ref{alg:candidate-sampling})

\State \textbf{Stage 3:} Compute importance scores (Algorithm~\ref{alg:score-g})
\State $\mathbf{p} \gets$ Normalized importance distribution

\State \textbf{Stage 4:} Construct $K$ edge functions (Algorithm~\ref{alg:edge-functions})
\State $\Phi \gets \{\phi_1, \ldots, \phi_K\}$

\State \textbf{Stage 5:} Compute output weights via OLS (Algorithm~\ref{alg:output-layer})
\State $\mathbf{w} \gets$ Output weights

\State \textbf{Define prediction:} $\hat{f}(\mathbf{x}) = \tilde{\mathbf{h}}(\mathbf{x})^\top \mathbf{w}$, where $\tilde{\mathbf{h}}(\mathbf{x}) = [\phi_1(\mathbf{x}), \ldots, \phi_K(\mathbf{x}), 1]^\top$

\State \textbf{return} Trained model with $\mathcal{M}$, $\Phi$, $\mathbf{w}$
\end{algorithmic}
\end{algorithm}
```

---

## LaTeX Preamble Required

To use these algorithms in your thesis, include the following in your document preamble:

```latex
\usepackage{algorithm}
\usepackage{algpseudocode}
```

Then include each algorithm block directly in your thesis sections using `\input{} ` or copy-paste the algorithm environments.