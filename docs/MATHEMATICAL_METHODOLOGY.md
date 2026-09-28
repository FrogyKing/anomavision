# Mathematical Methodology & Algorithmic Foundations of AnomaVision

This document details the mathematical formulations, geometric optimization objectives, complexity bounds, and statistical evaluation procedures implemented in **AnomaVision** (`src/anomavision/`). Every equation corresponds directly to an executable component in the codebase.

---

## 1. Dense Self-Supervised Representation Geometry (`src/anomavision/models/dinov3.py`)

### 1.1 Patch Tokenization and Rotary Position Embeddings (RoPE)
Let $X \in \mathbb{R}^{3 \times H_{\text{orig}} \times W_{\text{orig}}}$ denote an input RGB industrial inspection image. Preprocessing deterministically resizes $X$ via bicubic interpolation to $\tilde{X} \in \mathbb{R}^{3 \times H_{\text{in}} \times W_{\text{in}}}$, where $H_{\text{in}} = W_{\text{in}} = 448$ is an exact multiple of the Vision Transformer patch stride $P = 16$, and applies channel-wise standardization:

$$\tilde{X}_{c, u, v} = \frac{X'_{c, u, v} / 255 - \mu_c}{\sigma_c}, \quad \mu = (0.485, 0.456, 0.406), \quad \sigma = (0.229, 0.224, 0.225)$$

The non-overlapping patch projection partitions $\tilde{X}$ into a regular 2D lattice of size $(H_p, W_p) = (H_{\text{in}}/P, W_{\text{in}}/P) = (28, 28)$, yielding $N = H_p \cdot W_p = 784$ spatial tokens of dimension $d_0 = 384$. Unlike fixed-resolution absolute positional embeddings, DINOv3 ViT-S/16 (`affineisallyouneed/dinov3-vits16-pretrain-lvd1689m`, Siméoni et al., 2025) applies 2D Axial Rotary Position Embeddings (RoPE, base frequency $\theta = 100.0$) inside self-attention heads:

$$\text{Attention}_h(Q_h, K_h, V_h) = \text{softmax}\!\left(\frac{(\mathcal{R}_{\Theta} Q_h)(\mathcal{R}_{\Theta} K_h)^\top}{\sqrt{d_k}}\right) V_h$$

where $\mathcal{R}_{\Theta}$ is block-diagonal rotation parameterized by relative 2D grid coordinates $(r, c) \in \{0, \dots, H_p-1\} \times \{0, \dots, W_p-1\}$. Consequently, inner products between query and key representations depend on relative spatial displacement $(r_1 - r_2, c_1 - c_2)$, allowing seamless evaluation at $448 \times 448$ ($784$ patches, $4\times$ denser than $224 \times 224$) without fine-tuning.

### 1.2 Register Token Slicing and Spatial Grid Reconstruction
At each transformer block $l \in \{1, \dots, 12\}$, the sequence tensor $T^{(l)} \in \mathbb{R}^{(1 + R + N) \times d_0}$ has length $L = 1 + 4 + 784 = 789$:
- $T^{(l)}_{0} \in \mathbb{R}^{d_0}$: Global `[CLS]` token trained via DINO self-distillation and KoLeo entropy regularization.
- $T^{(l)}_{1 : 1+R} \in \mathbb{R}^{R \times d_0}$ ($R = 4$): Register tokens (Darcet et al., ICLR 2024). In standard ViTs without registers, redundant low-informative background patches are repurposed by the network as high-norm internal memory buffers, corrupting local spatial anomaly maps. Slicing out indices $1 \dots 4$ removes these high-norm global artifacts prior to spatial reconstruction.
- $T^{(l)}_{1+R : 1+R+N} \in \mathbb{R}^{N \times d_0}$: Row-major spatial patch tokens corresponding to grid coordinates $(r, c) = (\lfloor q / W_p \rfloor, q \bmod W_p)$ for $q \in \{0, \dots, N-1\}$.

### 1.3 Multi-Layer Aggregation and Local Neighborhood Smoothing
Purely deep tokens ($l=12$) capture high-level semantic part identity (e.g., thread vs. collar of a bottle) while intermediate blocks ($l \in \{6, 9\}$) retain fine-grained edge, scratch, and surface-texture sensitivity. In `process_token_sequence` (`src/anomavision/models/dinov3.py`), we extract intermediate states from layers $\mathcal{L} = \{6, 9, 12\}$, apply channel-wise Layer Normalization $\widehat{T}^{(l)} = \text{LN}(T^{(l)})$ to equalize layer variances, reshape the spatial tokens to $\Phi^{(l)} \in \mathbb{R}^{d_0 \times H_p \times W_p}$, and apply local $p \times p$ neighborhood average pooling ($p = 3$, stride $1$, boundary-corrected):

$$\widetilde{\Phi}^{(l)}_{:, r, c} = \frac{1}{|\mathcal{N}_p(r, c)|} \sum_{(r', c') \in \mathcal{N}_p(r, c)} \widehat{\Phi}^{(l)}_{:, r', c'}$$

where $\mathcal{N}_p(r, c) = \{(r', c') \in \{0,\dots,H_p-1\}\times\{0,\dots,W_p-1\} : \max(|r'-r|, |c'-c|) \le \lfloor p/2 \rfloor\}$. Multi-layer descriptors are averaged across $l \in \mathcal{L}$ and projected onto the unit hypersphere $\mathbb{S}^{d-1}$ ($d = 384$):

$$z_q = \frac{\bar{\phi}_q}{\max(\|\bar{\phi}_q\|_2, \epsilon)}, \quad \bar{\phi}_q = \frac{1}{|\mathcal{L}|} \sum_{l \in \mathcal{L}} \widetilde{\Phi}^{(l)}_{:, \lfloor q/W_p \rfloor, q \bmod W_p}$$

---

## 2. Minimax Coreset Selection via Johnson-Lindenstrauss Greedy $k$-Center (`src/anomavision/core/coreset.py`)

### 2.1 Covering Objective
Let $\mathcal{D}_{\text{train\_fit}} = \{X_1, \dots, X_{n_{\text{fit}}}\}$ denote the normal training split (strictly disjoint from the calibration split $\mathcal{D}_{\text{cal}}$ and test split $\mathcal{D}_{\text{test}}$). Extracting $N = 784$ patch descriptors per image yields the candidate normal feature pool:

$$\mathcal{P} = \bigcup_{i=1}^{n_{\text{fit}}} \{z_{i, 1}, \dots, z_{i, N}\} \subset \mathbb{R}^d, \quad M_{\text{total}} = |\mathcal{P}| = n_{\text{fit}} \cdot N$$

For MVTec AD `bottle` ($n_{\text{fit}} = 178$) and `metal_nut` ($n_{\text{fit}} = 187$), $M_{\text{total}}$ is $139,552$ and $146,608$ patches, respectively. To compress $\mathcal{P}$ into a memory bank $\mathcal{M} \subset \mathcal{P}$ of target cardinality $K = \lfloor \gamma M_{\text{total}} \rfloor$ ($\gamma \in (0, 1]$) while preserving uniform coverage of the normal data manifold, we formulate the minimax facility-location ($k$-center) problem (Sener & Savarese, ICLR 2018; Roth et al., CVPR 2022):

$$\mathcal{M}^* = \arg\min_{\mathcal{M} \subset \mathcal{P},\, |\mathcal{M}| = K} \; \max_{z \in \mathcal{P}} \min_{m \in \mathcal{M}} \|z - m\|_2$$

### 2.2 Greedy 2-Approximation and Johnson-Lindenstrauss Acceleration
Exact $k$-center optimization is NP-hard, but Gonzalez's greedy farthest-point algorithm selects centers iteratively via:

$$m_t = \arg\max_{z \in \mathcal{P}} D_{t-1}(z), \quad D_t(z) = \min\bigl(D_{t-1}(z), \|z - m_t\|_2\bigr), \quad t = 1, \dots, K$$

which guarantees a covering radius $r(\mathcal{M}_K) = \max_{z \in \mathcal{P}} D_K(z) \le 2 r(\mathcal{M}^*)$.

By the **Johnson-Lindenstrauss (JL) Lemma**, for any $0 < \epsilon < 1/2$ and projection dimension $d_{\text{proj}} = \mathcal{O}(\epsilon^{-2} \log M_{\text{total}})$, a random Gaussian matrix $\Pi \in \mathbb{R}^{d \times d_{\text{proj}}}$ with entries $\Pi_{ij} \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, 1/d_{\text{proj}})$ satisfies:

$$(1 - \epsilon)\|z - z'\|_2^2 \le \|\Pi^\top z - \Pi^\top z'\|_2^2 \le (1 + \epsilon)\|z - z'\|_2^2$$

with high probability for all pairs $(z, z') \in \mathcal{P}^2$. In `select_greedy_kcenter_coreset`, we project $\mathcal{P}$ onto $d_{\text{proj}} = 128$ exclusively for evaluating the farthest-point argmax updates, while storing the selected centers in their exact full-dimensional space $\mathcal{M}_K \subset \mathbb{R}^{384}$. This reduces complexity from $\mathcal{O}(K \cdot M_{\text{total}} \cdot d)$ to $\mathcal{O}(K \cdot M_{\text{total}} \cdot d_{\text{proj}})$.

### 2.3 Prefix-Nesting Property for Multi-Ratio Ablations
Because initialization ($m_1 = \arg\max_{z \in \mathcal{P}} \|\Pi^\top z - \mu_{\Pi}\|_2$) and distance updates $D_t(z)$ are strictly causal and deterministic given a fixed seed and projection matrix $\Pi$, the sequence of selected indices $(i_1, i_2, \dots, i_{K_{\max}})$ satisfies the **prefix-nesting invariant**:

$$\forall K_1 < K_2 \le K_{\max}, \quad \mathcal{M}_{K_1} = \{m_1, \dots, m_{K_1}\} \subset \mathcal{M}_{K_2}$$

In `_run_mathematical_ablations` (`src/anomavision/pipeline/stages.py`), this property allows exact evaluation across coreset ratios $\gamma \in \{0.02, 0.05, 0.10, 0.25\}$ from a single master $k$-center trajectory without redundant farthest-point passes.

---

## 3. Anomaly Scoring Geometry & Spatial Localization (`src/anomavision/core/detector.py`)

### 3.1 Cosine vs. Euclidean Metric Geometry
For a query patch descriptor $z_q \in \mathbb{R}^d$ and coreset element $m \in \mathcal{M}$, `PatchCoreDetector` supports both:
1. **Normalized Cosine Distance**:
   $$d_{\cos}(z_q, m) = 1 - \frac{\langle z_q, m \rangle}{\|z_q\|_2 \|m\|_2} \in [0, 2]$$
2. **Euclidean ($L_2$) Distance**:
   $$d_{L_2}(z_q, m) = \|z_q - m\|_2 = \sqrt{\max\bigl(0, \|z_q\|_2^2 + \|m\|_2^2 - 2\langle z_q, m \rangle\bigr)}$$

**Proposition (Chordal Isometry on $\mathbb{S}^{d-1}$):** When patch embeddings are $L_2$-normalized ($\|z_q\|_2 = \|m\|_2 = 1$), Euclidean distance is a strictly increasing monotonic function of cosine distance:

$$d_{L_2}(z_q, m) = \sqrt{2 - 2\langle z_q, m \rangle} = \sqrt{2\,d_{\cos}(z_q, m)}$$

Consequently, the nearest-neighbor ordering $m_{(1)}, \dots, m_{(b)}$ on $\mathbb{S}^{d-1}$ is identical under both metrics (verified in `tests/test_core.py::test_cosine_vs_euclidean_isometry_on_unit_sphere`). Conversely, when embeddings are unnormalized (`l2_normalize=False`), $d_{L_2}$ couples angular semantic divergence with radial norm variations $\|z_q\|_2 - \|m\|_2$, which degrades localization on reflective metallic surfaces (`metal_nut` AUPRO drops from `0.9472` with `euc_l2` to `0.9424` with `euc_raw`).

### 3.2 Density-Reweighted Image Anomaly Score
For each query patch $z_q$ ($q = 1, \dots, N$), let $d_{(1)}(z_q) \le d_{(2)}(z_q) \le \dots \le d_{(b)}(z_q)$ denote the distances to its $b = 9$ nearest neighbors in $\mathcal{M}$. The patch-level anomaly score is $s_q = d_{(1)}(z_q)$. Let $q^* = \arg\max_{q} s_q$ be the most anomalous patch. To penalize isolated outliers in sparse regions of $\mathcal{M}$ more heavily than boundary points near dense normal clusters, we compute a scale-invariant softmax density weight:

$$w(z_{q^*}) = 1 - \frac{1}{\sum_{j=1}^{b} \exp\!\left(\frac{d_{(j)}(z_{q^*}) - d_{(1)}(z_{q^*})}{\tau_s}\right)}, \quad \tau_s = \max\bigl(\text{median}_{j}(d_{(j)}(z_{q^*})),\, 0.05\bigr)$$

and combine the density-reweighted peak score with the mean of the top-$k$ patches ($k = 5$):

$$S(X) = 0.85 \cdot \bigl(1 + w(z_{q^*})\bigr) d_{(1)}(z_{q^*}) + 0.15 \cdot \frac{1}{k} \sum_{r=1}^{k} s_{(r)}$$

### 3.3 Coordinate-Aligned Pixel Anomaly Map Reconstruction
The 1D patch score vector $(s_1, \dots, s_N)$ is reshaped into the 2D patch grid $S_{\text{grid}} \in \mathbb{R}^{H_p \times W_p}$, upsampled to original image dimensions $(H_{\text{orig}}, W_{\text{orig}})$ via bilinear interpolation (`align_corners=False`, which aligns patch centers with pixel-grid sub-blocks), and convolved with an isotropic 2D Gaussian kernel $G_{\sigma}$ ($\sigma = 4.0$ pixels):

$$A(u, v) = \bigl(\mathcal{U}_{H_{\text{orig}} \times W_{\text{orig}}}(S_{\text{grid}}) * G_{\sigma}\bigr)(u, v)$$

---

## 4. Object-Aware Foreground Gating with SAM 3 (`src/anomavision/models/sam.py`)

Given an image $X$ and category text prompt $p_{\text{text}}$ (e.g., `"bottle"`, `"metal nut"`), `SAM3ObjectSegmenter` (`tkclam/sam3`) produces binary instance/semantic masks which are hole-filled and morphologically dilated by a disk structuring element $B_r$ of radius $r = 5$ pixels:

$$S_{\text{dil}}(u, v) = \bigl(\text{FillHoles}(S_{\text{raw}}) \oplus B_r\bigr)(u, v) \in \{0, 1\}$$

To prevent conveyor belts or background illumination artifacts from dominating pixel anomaly maps while retaining full sensitivity on the object, `apply_object_mask_refinement` applies soft spatial gating:

$$\tilde{A}(u, v) = A(u, v) \cdot \Bigl(S_{\text{dil}}(u, v) + \gamma_{\text{bg}} \bigl(1 - S_{\text{dil}}(u, v)\bigr)\Bigr), \quad \gamma_{\text{bg}} = 0.15$$

---

## 5. Unsupervised Calibration, Linear-Time AUPRO, and MMD Drift (`src/anomavision/evaluation/`, `src/anomavision/core/drift.py`)

### 5.1 Anti-Leakage Unsupervised Threshold Calibration
Using the held-out normal calibration set $\mathcal{D}_{\text{cal}}$ ($15\%$ of normal training data, never included in $\mathcal{M}$), we compute out-of-memory normal image scores $\{S(X_i)\}_{i=1}^{n_{\text{cal}}}$ with sample mean $\mu_{\text{cal}}$ and standard deviation $\sigma_{\text{cal}}$. The operational decision threshold is set without seeing any anomalous samples:

$$\tau_{\text{img}} = \max\bigl(Q_{0.98}(\{S(X_i)\}_{\text{cal}}),\; \mu_{\text{cal}} + 2.5\,\sigma_{\text{cal}}\bigr)$$

### 5.2 Linear-Time $\mathcal{O}(N_{\text{pixels}})$ Cumulative Histogram AUPRO
For pixel-level localization evaluation, let $\{C_{i, c}\}$ denote the connected ground-truth defect components across anomalous test masks (extracted via 8-connected component labeling `scipy.ndimage.label`) and let $\mathcal{B} = \{(i, u, v) : Y_i(u, v) = 0\}$ denote all normal background pixels. Rather than re-scanning tens of millions of pixels across $T$ thresholds in $\mathcal{O}(T \cdot |\mathcal{B}|)$ time, `compute_aupro` (`src/anomavision/evaluation/metrics.py`) partitions $[A_{\min}, A_{\max}]$ into $T = 120$ uniform bins with edges $b_0 < b_1 < \dots < b_T$ and computes single-pass histograms $h^{\text{bg}} = \text{hist}(A|_{\mathcal{B}})$ and $h^{(i, c)} = \text{hist}(A|_{C_{i, c}})$. Reverse cumulative sums yield the exact False Positive Rate and Per-Region Overlap at every bin edge in $\mathcal{O}(|\mathcal{B}| + T)$ time:

$$\text{FPR}(b_t) = \frac{1}{|\mathcal{B}|} \sum_{r=t}^{T-1} h^{\text{bg}}_r, \quad \text{PRO}(b_t) = \frac{1}{|\mathcal{C}|} \sum_{(i, c) \in \mathcal{C}} \frac{1}{|C_{i, c}|} \sum_{r=t}^{T-1} h^{(i, c)}_r$$

Integrating $\text{PRO}$ with respect to $\text{FPR}$ over $[0, 0.30]$ via the trapezoidal rule and normalizing by $0.30$ yields $\text{AUPRO}_{0.30} \in [0, 1]$.

### 5.3 Multi-Bandwidth RBF Maximum Mean Discrepancy ($\text{MMD}^2$)
To monitor production covariate shift when ground-truth labels are absent or delayed, `compute_rbf_mmd_squared` (`src/anomavision/core/drift.py`) measures the squared reproducing kernel Hilbert space (RKHS) distance between reference normal patch anchors $X_{\text{ref}} = \{x_i\}_{i=1}^m$ and incoming query patches $Y_q = \{y_j\}_{j=1}^n$ using a multi-bandwidth Gaussian RBF mixture kernel ($\Sigma = \{0.25, 0.5, 1.0, 2.0\}$):

$$k(u, v) = \frac{1}{|\Sigma|} \sum_{\sigma \in \Sigma} \exp\!\left(-\frac{\|u - v\|_2^2}{2\sigma^2}\right), \quad \text{MMD}^2(X_{\text{ref}}, Y_q) = \frac{1}{m^2}\sum_{i,i'} k(x_i, x_{i'}) + \frac{1}{n^2}\sum_{j,j'} k(y_j, y_{j'}) - \frac{2}{mn}\sum_{i,j} k(x_i, y_j)$$

An unsupervised drift alert is raised when $\text{MMD}^2(X_{\text{ref}}, Y_q) > \mu_{\text{MMD, cal}} + 4\sigma_{\text{MMD, cal}}$.
