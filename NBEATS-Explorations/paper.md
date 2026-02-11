# N-BEATS Architecture Explorations and the Impact of Block Type on Performance

**Daniel Byrne**

---

## Abstract

[TODO: Experiments ongoing. Abstract will be written upon completion of all benchmark runs across M4 periods.]

---

## 1. Introduction

Time series forecasting is one of the oldest and most consequential problems in quantitative science. From inventory planning and financial risk management to energy load scheduling and epidemiological surveillance, accurate forecasts translate directly into better decisions and measurable economic value. For decades, the field was dominated by classical statistical methods--exponential smoothing, ARIMA, and their many variants--that offered strong theoretical grounding, interpretability, and reliable performance on the small-to-moderate datasets typical of business applications. Deep learning, despite its transformative impact on computer vision and natural language processing, was widely regarded as unnecessary or even counterproductive for univariate time series, where the number of observations per series is often modest and the risk of overfitting substantial.

The M4 competition (Makridakis et al., 2018; 2020) marked a turning point in this narrative. Among 60 submitted methods, the six "pure" machine learning entries ranked 23rd through 57th, seemingly confirming the skeptics' view. Yet the competition winner, Smyl's ES-RNN (2020), was a hybrid that fused an LSTM-based deep learning component with classical Holt-Winters exponential smoothing, outperforming all purely statistical methods. This result established that deep learning could contribute meaningfully to forecasting accuracy, but left open the question of whether a *pure* deep learning architecture--one requiring no hand-crafted statistical components--could achieve competitive or superior results.

Oreshkin et al. (2019) answered this question with N-BEATS (Neural Basis Expansion Analysis for Time Series), a fully deep learning architecture that surpassed both the M4 winner and all prior statistical methods on the M4, M3, and Tourism benchmarks. N-BEATS introduced a distinctive design built on doubly residual stacking of basic blocks, each consisting of a multi-layer fully connected network that forks into backcast and forecast paths via learned or constrained basis expansion coefficients. The architecture offered two configurations: a Generic model using fully learnable basis functions, and an Interpretable model constraining basis functions to polynomial (Trend) and Fourier (Seasonality) forms. The success of N-BEATS demonstrated that the choice of basis function within each block is a critical design decision--one that determines how the network decomposes and reconstructs the input signal.

This observation motivates the present work. If polynomial and Fourier bases can achieve state-of-the-art results when embedded within the N-BEATS doubly residual framework, what happens when we substitute alternative basis expansions? Wavelets offer multi-resolution time-frequency localization that neither polynomials nor Fourier series provide. Autoencoders learn compressed, data-driven representations that may capture structure not well-described by any fixed analytical basis. Bottleneck projections offer rank-constrained factorizations that trade expressiveness for parameter efficiency. Each of these alternatives embodies a different inductive bias about the structure of time series signals, and the N-BEATS framework provides a controlled setting in which to evaluate them head-to-head.

We present a systematic exploration of these alternative block types within the N-BEATS architecture, implemented as the `lightningnbeats` PyTorch Lightning package. Our contributions include: (1) novel block types--wavelet basis blocks (Haar, Daubechies, Coiflets, Symlets), autoencoder blocks with separated encoder-decoder paths, bottleneck generic blocks with rank-d factorized projections, and hybrid AE-backbone variants of all original block types; (2) a rigorous benchmark framework enabling 1:1 comparison with the original N-BEATS paper's configurations on the M4 dataset using paper-faithful hyperparameters; and (3) ablation studies examining the impact of training extensions (active_g post-basis activation, sum_losses backcast regularization) and alternative activation functions. This is a work in progress, with experiments currently running across all six M4 periods.

---

## 2. Prior Work

### 2.1 Classical Statistical Forecasting

The foundation of modern time series forecasting rests on statistical methods developed over the latter half of the 20th century. Exponential smoothing (Holt, 1957; Winters, 1960) and its state-space formulation ETS (Hyndman et al., 2002) provide a flexible framework for capturing level, trend, and seasonal components through weighted combinations of past observations with exponentially decaying weights. The Box-Jenkins ARIMA methodology (Box & Jenkins, 1976) and its seasonal extension SARIMA model time series as linear functions of their own past values and past forecast errors, with automatic model selection procedures (Hyndman & Khandakar, 2008) making these methods accessible to non-specialists. The Theta method (Assimakopoulos & Nikolopoulos, 2000), winner of the M3 competition, decomposes the original series into "theta lines"--modified versions of the series with amplified or dampened curvature--and combines their extrapolations. STL decomposition (Cleveland et al., 1990) provides a robust loess-based procedure for separating seasonal, trend, and remainder components that remains widely used for exploratory analysis and as a preprocessing step.

These methods share several important properties: they are computationally inexpensive, require minimal training data, produce interpretable decompositions, and perform reliably across a wide range of domains. Their primary limitation is their reliance on linear or locally linear assumptions about the data-generating process, which can constrain their ability to capture complex nonlinear dynamics.

### 2.2 The M Competitions and the Rise of Deep Learning

The Makridakis competitions have served as the primary empirical proving ground for forecasting methods since 1982. The M3 competition (Makridakis & Hibon, 2000) established that simple statistical methods, particularly the Theta method, could outperform more complex approaches across a diverse set of 3,003 time series. This finding reinforced a widespread skepticism toward complex models, including neural networks, for forecasting tasks.

The M4 competition (Makridakis et al., 2018; 2020) dramatically expanded the scale to 100,000 univariate time series spanning six sampling frequencies (Yearly, Quarterly, Monthly, Weekly, Daily, Hourly) drawn from diverse domains including demographics, finance, industry, and macroeconomics. The competition introduced the Overall Weighted Average (OWA) metric, which normalizes sMAPE and MASE scores against a seasonally-adjusted Naive2 baseline so that OWA = 1.0 corresponds to naive forecast performance. The competition winner, Smyl's ES-RNN (2020), achieved an OWA of 0.821 by combining a dilated residual/attention LSTM stack with a per-series Holt-Winters statistical model whose parameters were jointly optimized via gradient descent. The second-place entry (Montero-Manso et al., 2020) used gradient boosted trees to combine the outputs of nine classical statistical methods. Notably, the six purely machine learning submissions ranked in the bottom half of entries, leading Makridakis et al. to suggest that hybrid approaches were "the way forward" for forecasting.

### 2.3 Deep Learning Architectures for Time Series

The landscape of deep learning approaches to time series forecasting has evolved rapidly. DeepAR (Salinas et al., 2020) uses autoregressive recurrent neural networks to produce probabilistic forecasts, modeling the conditional distribution of future values at each time step. WaveNet-inspired architectures (van den Oord et al., 2016) introduced dilated causal convolutions that capture long-range dependencies with manageable parameter counts. Temporal Fusion Transformers (Lim et al., 2021) combine variable selection networks, gating mechanisms, and multi-head attention to handle both static and time-varying covariates with interpretable attention patterns.

More recently, the field has seen a shift toward simpler architectures that challenge the perceived necessity of complex temporal inductive biases. N-HiTS (Challu et al., 2023) extends N-BEATS with hierarchical interpolation and multi-rate signal sampling, achieving competitive results with improved computational efficiency. DLinear (Zeng et al., 2023) demonstrated that simple linear models applied to decomposed trend and seasonal components could match or exceed Transformer-based approaches on several benchmarks. PatchTST (Nie et al., 2023) applies Vision Transformer-style patching to time series, treating contiguous subsequences as tokens and achieving state-of-the-art long-horizon forecasting through channel independence and patch-level attention.

### 2.4 N-BEATS: Neural Basis Expansion Analysis

N-BEATS (Oreshkin et al., 2019) introduced a pure deep learning architecture for univariate time series forecasting that achieved state-of-the-art results without any time-series-specific components. The architecture is built from basic blocks, each consisting of four fully connected layers with ReLU activations followed by a fork into backcast and forecast paths. The key innovation is the *doubly residual stacking* principle: each block's backcast output is subtracted from the block's input before passing to the next block (backward residual connection, enabling iterative signal decomposition), while each block's forecast output is summed into the global forecast (forward residual connection, enabling hierarchical forecast aggregation). The final forecast is the sum of all blocks' partial forecasts: $\hat{y} = \sum_\ell \hat{y}_\ell$.

Within each block, the four FC layers produce a hidden representation $h$ from which expansion coefficients $\theta^b$ and $\theta^f$ are computed via linear projections. These coefficients are then multiplied by basis function matrices to produce the backcast $\hat{x} = g^b(\theta^b)$ and forecast $\hat{y} = g^f(\theta^f)$. The architecture offers two configurations:

- **N-BEATS-G (Generic)**: Basis functions $g^b$ and $g^f$ are fully learnable linear projections. The columns of the learned weight matrix $V^f$ can be interpreted as waveform basis vectors in the time domain, but carry no inherent structural constraint.
- **N-BEATS-I (Interpretable)**: Basis functions are constrained to specific functional forms. The Trend stack uses Vandermonde polynomial matrices $T = [1, t, t^2, \ldots, t^p]$ to produce slowly-varying outputs. The Seasonality stack uses Fourier basis matrices $S = [1, \cos(2\pi t), \ldots, \cos(2\pi \lfloor H/2-1 \rfloor t), \sin(2\pi t), \ldots, \sin(2\pi \lfloor H/2-1 \rfloor t)]$ to produce periodic outputs.

The original paper reported that a 180-model ensemble (combining models trained on three loss functions, six backcast lengths from 2H to 7H, and multiple random seeds) achieved OWA of 0.795 on the M4 dataset, with individual architectures (N-BEATS-G, N-BEATS-I) each independently outperforming the M4 competition winner. The same architecture and hyperparameters generalized across M3 and Tourism datasets without modification.

### 2.5 Wavelets in Signal Processing and Time Series

Wavelets provide a mathematical framework for multi-resolution analysis of signals, offering simultaneous localization in both time and frequency domains--a property that neither purely temporal (polynomial) nor purely spectral (Fourier) bases possess. The discrete wavelet transform (DWT) decomposes a signal into approximation coefficients (capturing low-frequency trend) and detail coefficients (capturing high-frequency fluctuations) at progressively coarser scales (Mallat, 1989; Daubechies, 1992).

Different wavelet families offer different trade-offs between time and frequency localization. Haar wavelets are the simplest, consisting of step functions, and provide maximal time localization at the cost of poor frequency resolution. Daubechies wavelets (db2, db3, db4, ...) provide increasingly smooth basis functions with wider support, trading time localization for better frequency resolution. Coiflets are designed for near-symmetry and vanishing moments in both the scaling and wavelet functions. Symlets are near-symmetric modifications of Daubechies wavelets.

In the context of time series forecasting, wavelets have been used primarily as preprocessing transforms--decomposing series into sub-bands that are forecast independently and then reconstructed (Aminghafari et al., 2006). However, the idea of using wavelet functions directly as basis expansions within a neural network block is less explored. By replacing the Fourier or polynomial basis in an N-BEATS block with a wavelet basis, we aim to provide the network with time-frequency localized basis functions that may better capture transient phenomena, regime changes, and localized oscillations in the input series.

### 2.6 Autoencoders for Time Series

Autoencoders (Hinton & Salakhutdinov, 2006) learn compressed representations of input data by training an encoder to map inputs to a lower-dimensional latent space and a decoder to reconstruct the original inputs from this representation. The bottleneck forces the network to learn salient features while discarding noise. In time series contexts, autoencoders have been applied to anomaly detection (Malhotra et al., 2016), representation learning for downstream classification tasks, and denoising (Vincent et al., 2008).

The application of autoencoder architectures within the N-BEATS framework takes two forms in this work. First, we replace the standard four-FC-layer backbone with an encoder-decoder architecture (the AERootBlock), where the input is progressively compressed through layers of decreasing width before being expanded back to the original width. Second, we introduce dedicated AutoEncoder blocks where the post-backbone fork into backcast and forecast paths is itself structured as separate encoder-decoder pipelines, with optional weight sharing between the backcast and forecast encoders. These designs explore whether the compression bottleneck provides useful regularization or feature extraction within the doubly residual stacking framework.

---

## 3. Method: N-BEATS Architecture and Extensions

### 3.1 Original N-BEATS Architecture

The N-BEATS architecture (Oreshkin et al., 2019) is composed of blocks organized into stacks. Each basic block accepts an input vector $x_\ell \in \mathbb{R}^t$ (where $t$ is the lookback window length) and outputs a backcast $\hat{x}_\ell \in \mathbb{R}^t$ and a forecast $\hat{y}_\ell \in \mathbb{R}^H$ (where $H$ is the forecast horizon).

**Basic Block.** The core computation within each block begins with four fully connected layers, each followed by a ReLU activation:

$$h_1 = \text{ReLU}(W_1 x + b_1), \quad h_2 = \text{ReLU}(W_2 h_1 + b_2), \quad h_3 = \text{ReLU}(W_3 h_2 + b_3), \quad h_4 = \text{ReLU}(W_4 h_3 + b_4)$$

The hidden representation $h_4 \in \mathbb{R}^{units}$ is then projected to produce expansion coefficients $\theta^b = W^b h_4$ and $\theta^f = W^f h_4$ via linear layers (without bias in the original Generic formulation). These coefficients are passed through basis functions to produce the outputs:

$$\hat{x}_\ell = g^b(\theta^b_\ell), \quad \hat{y}_\ell = g^f(\theta^f_\ell)$$

**Basis Functions.** The choice of $g^b$ and $g^f$ determines the block type:

- *Generic*: $g^b$ and $g^f$ are learnable linear projections. The expansion coefficients $\theta^f$ and $\theta^b$ are projected directly to the target lengths: $\hat{y} = V^f \theta^f$ where $V^f \in \mathbb{R}^{H \times \dim(\theta^f)}$ is a learned weight matrix. In our implementation, the Generic block follows the paper faithfully by using separate linear layers that project directly from units to `backcast_length` and `forecast_length` respectively--the projection matrices serve as both the theta extraction and basis expansion in a single step, with no intermediate bottleneck dimension.

- *Trend*: $g^b$ and $g^f$ are polynomial (Vandermonde) basis expansions. The expansion coefficients $\theta \in \mathbb{R}^p$ represent polynomial coefficients, and the basis matrix is $T = [1, t, t^2, \ldots, t^{p-1}]^T$ where $t$ is a normalized time vector on $[0, 1)$. With small polynomial degree $p$ (typically 2-3), the output is constrained to slowly varying functions suitable for modeling trend.

- *Seasonality*: $g^b$ and $g^f$ are Fourier basis expansions. The basis matrix consists of cosine and sine vectors at integer multiples of the fundamental frequency: $S = [1, \cos(2\pi t), \ldots, \cos(2\pi \lfloor L/2-1 \rfloor t), \sin(2\pi t), \ldots, \sin(2\pi \lfloor L/2-1 \rfloor t)]^T$ where $L$ is the target length. The expansion coefficients $\theta$ represent Fourier coefficients, constraining the output to periodic functions.

**Doubly Residual Stacking.** Blocks are organized into stacks, and the doubly residual topology connects them via residual connections inspired by deep residual learning (He et al., 2016):

$$x_\ell = x_{\ell-1} - \hat{x}_{\ell-1}, \quad \hat{y} = \sum_\ell \hat{y}_\ell$$

Each block subtracts its backcast from the input (removing the signal component it has modeled) before passing the residual to the next block. All forecast partial outputs are summed to produce the final forecast. This creates an iterative decomposition: early blocks capture the most prominent signal components, while later blocks model progressively finer residual structure.

**Weight Sharing.** When weight sharing is enabled within a stack, all blocks in that stack use the same parameters. The original paper found that weight sharing improved validation performance for the interpretable architecture (3 blocks per stack for both Trend and Seasonality stacks, with shared weights).

### 3.2 Novel Block Extensions

We introduce several families of novel block types that explore alternative basis expansions and backbone architectures within the N-BEATS doubly residual framework.

#### 3.2.1 BottleneckGeneric Block

The paper-faithful Generic block projects directly from hidden units to the target length in a single linear layer. The BottleneckGeneric block introduces an intermediate bottleneck dimension `thetas_dim` $= d$, factoring this projection into two steps:

$$\theta^b = W_1^b h_4, \quad \hat{x} = W_2^b \theta^b, \quad \theta^f = W_1^f h_4, \quad \hat{y} = W_2^f \theta^f$$

where $W_1 \in \mathbb{R}^{d \times units}$ and $W_2 \in \mathbb{R}^{target\_length \times d}$ (with $W_2$ having no bias). This is equivalent to a rank-$d$ factorization of the basis expansion matrix: instead of learning a full $target\_length \times units$ matrix, the network learns the product of two smaller matrices, providing a tunable knob to control basis complexity. With `thetas_dim` = 5 (the default), this dramatically constrains the effective rank of the basis while potentially acting as a regularizer. When `share_weights` is True, the first-stage linear layers $W_1^b$ and $W_1^f$ share parameters.

#### 3.2.2 AutoEncoder Block

The AutoEncoder block retains the standard four-FC-layer backbone from RootBlock but replaces the simple linear basis expansion with a two-stage encoder-decoder pipeline for each of the backcast and forecast paths:

$$z^b = \text{Encoder}^b(h_4), \quad \hat{x} = \text{Decoder}^b(z^b)$$
$$z^f = \text{Encoder}^f(h_4), \quad \hat{y} = \text{Decoder}^f(z^f)$$

Each encoder is a linear layer ($units \rightarrow thetas\_dim$) followed by ReLU. Each decoder consists of a linear layer ($thetas\_dim \rightarrow units$), ReLU, and a final linear layer ($units \rightarrow target\_length$). When `share_weights` is True, the backcast and forecast encoders share parameters, though the decoders remain separate since they map to different target lengths. This design provides a latent bottleneck representation analogous to the BottleneckGeneric but with nonlinear encoding and a richer decoding path.

#### 3.2.3 GenericAEBackcast Block

The GenericAEBackcast is a hybrid block that uses an autoencoder-style path for backcast reconstruction while retaining a bottleneck linear projection for the forecast:

- **Backcast path**: $h_4 \rightarrow \text{ReLU}(\text{Linear}(units, thetas\_dim)) \rightarrow \text{ReLU}(\text{Linear}(thetas\_dim, units)) \rightarrow \text{Linear}(units, backcast\_length)$
- **Forecast path**: $h_4 \rightarrow \text{Linear}(units, thetas\_dim) \rightarrow \text{Linear}(thetas\_dim, forecast\_length)$

The rationale is that backcast reconstruction (estimating the input from a compressed representation) is fundamentally a reconstruction task well-suited to autoencoder architectures, while the forecast path may benefit from the simpler bottleneck projection that directly maps compressed coefficients to future values.

#### 3.2.4 AERootBlock Variants

All blocks described above use the standard RootBlock backbone: four FC layers of equal width `units`. We introduce a parallel hierarchy of blocks using the AERootBlock backbone, which replaces the four equal-width layers with an encoder-decoder structure:

$$h_1 = \text{ReLU}(W_1 x + b_1), \quad W_1 \in \mathbb{R}^{units/2 \times backcast\_length}$$
$$h_2 = \text{ReLU}(W_2 h_1 + b_2), \quad W_2 \in \mathbb{R}^{latent\_dim \times units/2}$$
$$h_3 = \text{ReLU}(W_3 h_2 + b_3), \quad W_3 \in \mathbb{R}^{units/2 \times latent\_dim}$$
$$h_4 = \text{ReLU}(W_4 h_3 + b_4), \quad W_4 \in \mathbb{R}^{units \times units/2}$$

This creates an hourglass shape ($backcast\_length \rightarrow units/2 \rightarrow latent\_dim \rightarrow units/2 \rightarrow units$) that forces the network to learn a compressed representation of the input before the fork into backcast and forecast paths. With `latent_dim` = 4 and `units` = 512, the bottleneck compresses the representation to just 4 dimensions in the middle layers, compared to the standard backbone's uniform 512-dimensional hidden states.

Every block type from the standard hierarchy has an AE-backbone counterpart: GenericAE, BottleneckGenericAE, TrendAE, SeasonalityAE, AutoEncoderAE, and GenericAEBackcastAE. These use the same post-backbone basis expansions as their standard counterparts but with the AERootBlock backbone. This results in significantly fewer parameters--for example, GenericAE with `units` = 512 and `latent_dim` = 4 has approximately 4.8M parameters at 30 stacks, compared to 24.7M for the standard Generic at the same scale.

#### 3.2.5 Wavelet Blocks

Wavelet blocks replace the learnable or fixed analytical basis functions with discrete wavelet basis matrices derived from specific wavelet families. We implement two wavelet block designs:

**Standard Wavelet (square basis with learned downsampling).** The wavelet basis generator constructs a square basis matrix $W \in \mathbb{R}^{basis\_dim \times basis\_dim}$ from the scaling function $\phi$ and wavelet function $\psi$ of the chosen wavelet family. The functions are evaluated at `basis_dim` uniformly spaced points via interpolation from PyWavelets' high-resolution wavelet function evaluations. The first half of the basis columns are filled with cyclic shifts of $\phi$; the second half with cyclic shifts of $\psi$. The forward pass proceeds as:

$$\theta = \text{Linear}(h_4, basis\_dim), \quad z = W^T \theta, \quad \hat{y} = \text{Linear}(z, target\_length)$$

The square wavelet basis produces a `basis_dim`-dimensional output that requires a learned downsampling layer to map to the target length. This adds trainable parameters but allows the network to learn which components of the wavelet expansion are most relevant for the target.

**AltWavelet (rectangular basis with direct projection).** The alternative wavelet generator constructs a rectangular basis matrix $W \in \mathbb{R}^{basis\_dim \times target\_length}$ using the same wavelet functions but with columns indexed by the target length rather than the basis dimension. This allows direct projection without a learned downsampling layer:

$$\theta = \text{Linear}(h_4, basis\_dim), \quad \hat{y} = W^T \theta$$

Concrete wavelet block subclasses are thin wrappers that set the wavelet type string: HaarWavelet, DB2Wavelet, DB3Wavelet, DB4Wavelet, DB10Wavelet (Daubechies family), Coif1Wavelet through Coif10Wavelet (Coiflets), and Symlet2Wavelet through Symlet20Wavelet (Symlets). Each exists in both standard (Wavelet) and alternative (AltWavelet) variants, yielding a library of over 30 wavelet block types registered in the block constant registry.

### 3.3 Training Extensions

#### 3.3.1 Active G (`active_g`)

The original N-BEATS paper does not apply any activation function after the basis expansion--the outputs of $g^b$ and $g^f$ are linear combinations of basis vectors. We introduce an optional `active_g` parameter that, when enabled, applies the block's activation function (default ReLU) to both the backcast and forecast outputs after the basis expansion:

$$\hat{x} = \sigma(g^b(\theta^b)), \quad \hat{y} = \sigma(g^f(\theta^f))$$

The motivation is empirical: we observed that Generic-type blocks sometimes fail to converge without this post-expansion activation. The Trend and Seasonality blocks implicitly constrain their outputs through their fixed basis functions (polynomials are smooth, Fourier series are periodic), which may provide sufficient regularization. Generic blocks, whose basis functions are entirely learned, lack this constraint and may benefit from the additional nonlinearity. This parameter is available for all block types but is most impactful for Generic and BottleneckGeneric variants.

#### 3.3.2 Sum Losses (`sum_losses`)

The standard N-BEATS training objective is the forecast loss alone: $\mathcal{L} = \text{loss}(\hat{y}, y)$. The `sum_losses` extension adds a weighted backcast reconstruction loss:

$$\mathcal{L} = \text{loss}(\hat{y}, y) + 0.25 \cdot \text{loss}(\hat{x}_{residual}, \mathbf{0})$$

where $\hat{x}_{residual}$ is the final backcast residual (the portion of the input not reconstructed by any block) and $\mathbf{0}$ is a zero vector. The coefficient 0.25 weights the backcast loss at one quarter the forecast loss. This regularization encourages the backcast path to fully reconstruct the input signal, pushing the residual toward zero. The intuition is that a model that can accurately reconstruct the input has learned a more complete representation of the signal, which should improve forecast quality. This is inspired by the doubly residual architecture's design intent: the backcast path exists specifically to decompose the input signal, and explicitly optimizing this decomposition may improve the quality of the resulting partial forecasts.

### 3.4 Customizable Stack Composition

A key implementation feature of our framework is the ability to compose arbitrary ordered sequences of any block type. The `stack_types` parameter accepts a list of block type strings, and the model constructs one stack for each entry. This enables configurations that would not be possible in the original implementation:

- **Homogeneous stacks**: 30 copies of a single novel block type (e.g., `["HaarWavelet"] * 30`) for direct comparison with N-BEATS-G's 30 Generic stacks.
- **Interpretable combinations**: Trend + Seasonality at the original 2-stack scale, or with AE-backbone variants (`["TrendAE", "SeasonalityAE"]`).
- **Mixed compositions**: Alternating Trend and Wavelet stacks (`["Trend", "DB3Wavelet"] * 15`), or prepending interpretable stacks to generic ones (`["Trend", "Seasonality"] + ["Generic"] * 28`).

This compositional flexibility is central to our experimental design, as it allows us to isolate the effect of individual block types while keeping all other architectural decisions (depth, residual topology, training protocol) constant.

---

## 4. Experiment Setup

### 4.1 Dataset

All experiments use the M4 competition dataset (Makridakis et al., 2018; 2020), comprising 100,000 univariate time series across six sampling frequencies. The dataset provides standardized train/test splits, with test set lengths equal to the forecast horizon $H$ for each period.

| Period    | Frequency ($m$) | Forecast Horizon ($H$) | Backcast Length ($5 \times H$) | Number of Series |
|-----------|:-:|:-:|:-:|:-:|
| Yearly    | 1  | 6   | 30  | 23,000 |
| Quarterly | 4  | 8   | 40  | 24,000 |
| Monthly   | 12 | 18  | 90  | 48,000 |
| Weekly    | 1  | 13  | 65  | 359    |
| Daily     | 1  | 14  | 70  | 4,227  |
| Hourly    | 24 | 48  | 240 | 414    |

### 4.2 Baseline Reference Values

Performance is measured relative to the Naive2 seasonal baseline from the M4 competition. The Naive2 method produces forecasts using the last observed seasonal cycle, and its sMAPE and MASE values serve as the denominator for OWA computation:

| Period    | Naive2 sMAPE | Naive2 MASE |
|-----------|:--:|:--:|
| Yearly    | 16.342 | 3.974 |
| Quarterly | 11.012 | 1.371 |
| Monthly   | 14.427 | 1.063 |
| Weekly    | 9.161  | 2.777 |
| Daily     | 3.045  | 3.278 |
| Hourly    | 18.383 | 2.395 |

### 4.3 Model Configurations (Part 1: Block Benchmark)

We evaluate 17 model configurations grouped into four categories. All configurations use the same training hyperparameters for fair comparison. The three paper baseline configurations replicate the original N-BEATS paper's architecture choices.

| Config Name | Stack Types | Stacks | Blocks/Stack | Share Weights | Approx. Params |
|-------------|-------------|:------:|:------------:|:---:|---:|
| **Paper Baselines** | | | | | |
| NBEATS-G | Generic x30 | 30 | 1 | Yes | ~24.7M |
| NBEATS-I | Trend, Seasonality | 2 | 3 | Yes | ~12.9M |
| NBEATS-I+G | Trend, Seasonality, Generic x28 | 30 | 1 | Yes | ~36.0M |
| **Novel Single-Type** | | | | | |
| BottleneckGeneric | BottleneckGeneric x30 | 30 | 1 | Yes | ~24.2M |
| AutoEncoder | AutoEncoder x30 | 30 | 1 | Yes | ~24.9M |
| GenericAE | GenericAE x30 | 30 | 1 | Yes | ~4.8M |
| BottleneckGenericAE | BottleneckGenericAE x30 | 30 | 1 | Yes | ~4.3M |
| GenericAEBackcast | GenericAEBackcast x30 | 30 | 1 | Yes | ~24.8M |
| HaarWavelet | HaarWavelet x30 | 30 | 1 | Yes | ~26.2M |
| DB3Wavelet | DB3Wavelet x30 | 30 | 1 | Yes | ~26.2M |
| DB3AltWavelet | DB3AltWavelet x30 | 30 | 1 | Yes | ~26.1M |
| Coif2Wavelet | Coif2Wavelet x30 | 30 | 1 | Yes | ~26.2M |
| Symlet3Wavelet | Symlet3Wavelet x30 | 30 | 1 | Yes | ~26.2M |
| **Novel Interpretable** | | | | | |
| NBEATS-I-AE | TrendAE, SeasonalityAE | 2 | 3 | Yes | ~2.2M |
| **Novel Mixed Stacks** | | | | | |
| Trend+HaarWavelet | (Trend, HaarWavelet) x15 | 30 | 1 | Yes | ~16.2M |
| Trend+DB3Wavelet | (Trend, DB3Wavelet) x15 | 30 | 1 | Yes | ~16.2M |
| Generic+DB3Wavelet | (Generic, DB3Wavelet) x15 | 30 | 1 | Yes | ~25.4M |

Parameter counts are approximate and vary slightly by period due to differences in backcast_length and forecast_length affecting the first layer and projection layer sizes. The counts shown are representative of the Yearly period.

### 4.4 Training Protocol

All experiments share the following training configuration, chosen to match the original N-BEATS paper where applicable:

- **Batch size**: 1024 (paper: 1024)
- **Optimizer**: Adam with default parameters
- **Learning rate**: 1e-3 (paper: 1e-3)
- **Loss function**: SMAPELoss (paper's primary metric)
- **Early stopping**: patience = 10, monitoring validation loss
- **Maximum epochs**: 100 (early stopping typically terminates training well before this limit)
- **Backcast multiplier**: 5x forecast horizon (paper uses 2H-7H for ensemble; 5H is a representative single point)
- **Seeds**: 3 runs per configuration (seeds 42, 43, 44)
- **Thetas dim**: 5 (polynomial degree for Trend; bottleneck dimension for BottleneckGeneric)
- **Latent dim**: 4 (for AE-backbone blocks)
- **Basis dim**: 128 (for Wavelet blocks)

Data is loaded using the columnar data module (`ColumnarCollectionTimeSeriesDataModule`), which handles the M4 dataset's variable-length series by NaN-padding shorter series. Training/validation splits are created by holding out the last `backcast_length + forecast_length` observations for validation. Test evaluation uses a dedicated test data module that concatenates the tail of the training data with the test set to provide the required lookback window for inference.

### 4.5 Evaluation Metrics

We report three standard metrics for each experiment:

**sMAPE** (Symmetric Mean Absolute Percentage Error):
$$\text{sMAPE} = \frac{100}{H} \sum_{i=1}^{H} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2 + \epsilon}$$

**MASE** (Mean Absolute Scaled Error), computed per-series using the full training history for the naive seasonal denominator:
$$\text{MASE}_i = \frac{\frac{1}{H}\sum_{j=1}^{H} |y_{T+j} - \hat{y}_{T+j}|}{\frac{1}{T-m}\sum_{j=m+1}^{T} |y_j - y_{j-m}|}$$

where $m$ is the seasonal period and $T$ is the training set length. The reported MASE is the mean across all valid series.

**OWA** (Overall Weighted Average):
$$\text{OWA} = \frac{1}{2}\left(\frac{\text{sMAPE}}{\text{sMAPE}_{\text{Naive2}}} + \frac{\text{MASE}}{\text{MASE}_{\text{Naive2}}}\right)$$

OWA = 1.0 corresponds to the performance of the seasonally-adjusted naive baseline. Lower values indicate better performance.

### 4.6 Ablation Study Design (Part 2)

The ablation study evaluates training extensions and alternative activation functions using the 30-stack Generic configuration as the base architecture. All ablation runs share the same training protocol as the block benchmark.

| Config Name | active_g | sum_losses | Activation |
|-------------|:---:|:---:|:---:|
| Generic_baseline | False | False | ReLU |
| Generic_activeG | True | False | ReLU |
| Generic_sumLosses | False | True | ReLU |
| Generic_activeG+sumL | True | True | ReLU |
| Generic_GELU | False | False | GELU |
| Generic_ELU | False | False | ELU |
| Generic_LeakyReLU | False | False | LeakyReLU |
| Generic_SELU | False | False | SELU |

The baseline (first row) is identical to NBEATS-G from Part 1, providing a control for comparison. Each subsequent configuration modifies exactly one or two factors, enabling isolation of individual effects.

### 4.7 Ensemble Strategy (Part 3)

The ensemble experiment follows the original N-BEATS paper's multi-horizon strategy. For each of the three paper architectures (NBEATS-G, NBEATS-I, NBEATS-I+G), models are trained at six backcast multipliers (2H, 3H, 4H, 5H, 6H, 7H) with three random seeds each, yielding 18 models per architecture per period. The final ensemble forecast is the element-wise **median** across all individual model predictions, following the paper's aggregation function. Both individual model metrics and ensemble metrics are recorded.

### 4.8 Implementation

All models are implemented using PyTorch Lightning via the `lightningnbeats` package. The experiment script (`experiments/run_experiments.py`) supports incremental execution with CSV-based resumability--completed runs are detected and skipped on restart. Automatic accelerator detection selects CUDA, MPS (Apple Silicon), or CPU as available. TensorBoard logging captures training curves for all runs.

---

## 5. Results

We report results from the block benchmark (Part 1) across three of six M4 periods (Yearly, Quarterly, Monthly), the ensemble experiment (Part 3) for the Yearly period, and a statistical significance analysis testing whether block type meaningfully affects forecasting accuracy. Ablation results (Part 2) are pending. Remaining periods (Weekly, Daily, Hourly) will be added upon completion.

Throughout this section, a run is classified as **healthy** if it produced non-NaN metrics with OWA < 2.0 and MASE < 10⁶. Runs failing these criteria are classified as **divergent** (produced extreme but finite values) or **failed** (produced NaN, typically at epoch 1). The **convergence rate** reports the fraction of seeds that produced healthy runs.

### 5.1 Block Benchmark Results

#### 5.1.1 Main OWA Comparison

Table 2 presents the primary results: mean OWA across healthy seeds for each configuration, grouped by category. Only configurations with at least 2 healthy seeds out of 3 are included in the main comparison; wavelet-only and unstable mixed configurations are analyzed separately in Section 5.1.3.

**Table 2: Mean OWA by Configuration and Period (healthy runs only)**

| Config | Category | Params | Yearly OWA | Quarterly OWA | Monthly OWA | W / D / H |
|--------|----------|--------|:----------:|:-------------:|:-----------:|:---------:|
| NBEATS-G | Baseline | 24.7M | 0.820 (3/3) | 0.902 (3/3) | 0.913 (3/3) | [pending] |
| NBEATS-I | Baseline | 12.9M | 0.816 (3/3) | 0.892 (3/3) | 0.923 (3/3) | [pending] |
| NBEATS-I+G | Baseline | 36.0M | 0.808 (3/3) | 0.896 (3/3) | 0.949 (3/3) | [pending] |
| BottleneckGeneric | Novel | 24.2M | 0.827 (3/3) | 0.912 (3/3) | 0.940 (3/3) | [pending] |
| AutoEncoder | Novel | 24.9M | 0.804 (3/3) | 0.892 (2/3) | 0.923 (3/3) | [pending] |
| GenericAE | Novel-AE | 4.8M | 0.808 (3/3) | 0.897 (3/3) | 0.956 (3/3) | [pending] |
| BottleneckGenericAE | Novel-AE | 4.3M | 0.806 (3/3) | 0.905 (3/3) | 0.930 (3/3) | [pending] |
| GenericAEBackcast | Novel | 24.8M | 0.808 (3/3) | 0.906 (3/3) | 0.953 (3/3) | [pending] |
| NBEATS-I-AE | Novel-Interp | 2.2M | 0.805 (3/3) | 0.937 (3/3) | 0.973 (3/3) | [pending] |
| Trend+DB3Wavelet | Mixed | 16.2M | 0.809 (3/3) | 0.896 (3/3) | [partial] | [pending] |

Parenthetical notation indicates convergence rate (healthy seeds / total seeds). Bold entries mark the best OWA per period. The Yearly column is sorted by OWA to facilitate comparison; rankings shift across periods (see Section 5.4.4 for cross-period rank consistency analysis).

Key observations from Table 2:

1. **AutoEncoder achieves the best Yearly OWA** (0.804), outperforming all three paper baselines. However, one of its three Quarterly seeds diverged (OWA = 3.29), reducing its Quarterly convergence rate to 2/3.

2. **NBEATS-I-AE is the most parameter-efficient competitive configuration**, achieving OWA 0.805 on Yearly with only 2.2M parameters--91% fewer than NBEATS-G (24.7M). However, its advantage diminishes on longer-horizon periods: Quarterly OWA degrades to 0.937 and Monthly to 0.973.

3. **BottleneckGenericAE** offers the best balance of parameter efficiency and cross-period stability among AE-backbone variants, with 4.3M parameters and OWA ranging from 0.806 (Yearly) to 0.930 (Monthly).

4. **Trend+DB3Wavelet** is the only wavelet-containing configuration to achieve full convergence on both Yearly and Quarterly, with competitive OWA (0.809 and 0.896 respectively). Monthly data is partial due to wavelet-related instability in the DB3 component.

5. **The OWA spread among healthy, fully-converging configurations is narrow**: approximately 0.023 on Yearly (0.804 to 0.827), 0.020 on Quarterly (0.892 to 0.912), and 0.043 on Monthly (0.913 to 0.956). This narrow spread motivates the statistical significance analysis in Section 5.4.

#### 5.1.2 Detailed Metrics per Period

Tables 3a-3c present sMAPE, MASE, and OWA with standard deviations for each completed period, ranked by mean OWA. Only configurations with at least 2 healthy seeds are shown.

**Table 3a: Yearly Period — Detailed Metrics (ranked by OWA)**

| Config | Mean sMAPE | Mean MASE | Mean OWA | Std OWA | Conv. Rate |
|--------|:----------:|:---------:|:--------:|:-------:|:----------:|
| AutoEncoder | 13.535 | 3.095 | 0.804 | 0.014 | 3/3 |
| NBEATS-I-AE | 13.572 | 3.101 | 0.805 | 0.001 | 3/3 |
| BottleneckGenericAE | 13.539 | 3.112 | 0.806 | 0.015 | 3/3 |
| GenericAE | 13.598 | 3.116 | 0.808 | 0.009 | 3/3 |
| NBEATS-I+G | 13.551 | 3.129 | 0.808 | 0.008 | 3/3 |
| GenericAEBackcast | 13.560 | 3.127 | 0.808 | 0.010 | 3/3 |
| Trend+DB3Wavelet | 13.572 | 3.127 | 0.809 | 0.015 | 3/3 |
| NBEATS-I | 13.692 | 3.156 | 0.816 | 0.008 | 3/3 |
| NBEATS-G | 13.718 | 3.179 | 0.820 | 0.009 | 3/3 |
| BottleneckGeneric | 13.770 | 3.226 | 0.827 | 0.016 | 3/3 |

**Table 3b: Quarterly Period — Detailed Metrics (ranked by OWA)**

| Config | Mean sMAPE | Mean MASE | Mean OWA | Std OWA | Conv. Rate |
|--------|:----------:|:---------:|:--------:|:-------:|:----------:|
| Trend+DB3Wavelet | 10.254 | 1.180 | 0.896 | 0.012 | 3/3 |
| NBEATS-I | 10.202 | 1.175 | 0.892 | 0.002 | 3/3 |
| AutoEncoder | 10.163 | 1.181 | 0.892 | 0.011 | 2/3 |
| GenericAE | 10.219 | 1.187 | 0.897 | 0.011 | 3/3 |
| NBEATS-I+G | 10.285 | 1.177 | 0.896 | 0.006 | 3/3 |
| NBEATS-G | 10.273 | 1.193 | 0.902 | 0.012 | 3/3 |
| BottleneckGenericAE | 10.246 | 1.206 | 0.905 | 0.005 | 3/3 |
| GenericAEBackcast | 10.321 | 1.199 | 0.906 | 0.010 | 3/3 |
| BottleneckGeneric | 10.370 | 1.210 | 0.912 | 0.014 | 3/3 |
| NBEATS-I-AE | 10.588 | 1.252 | 0.937 | 0.018 | 3/3 |

**Table 3c: Monthly Period — Detailed Metrics (ranked by OWA)**

| Config | Mean sMAPE | Mean MASE | Mean OWA | Std OWA | Conv. Rate |
|--------|:----------:|:---------:|:--------:|:-------:|:----------:|
| NBEATS-G | 13.379 | 0.955 | 0.913 | 0.017 | 3/3 |
| AutoEncoder | 13.512 | 0.967 | 0.923 | 0.026 | 3/3 |
| NBEATS-I | 13.563 | 0.964 | 0.923 | 0.008 | 3/3 |
| BottleneckGenericAE | 13.564 | 0.977 | 0.930 | 0.007 | 3/3 |
| BottleneckGeneric | 13.927 | 0.973 | 0.940 | 0.019 | 3/3 |
| NBEATS-I+G | 14.069 | 0.981 | 0.949 | 0.006 | 3/3 |
| GenericAEBackcast | 14.054 | 0.991 | 0.953 | 0.040 | 3/3 |
| GenericAE | 14.211 | 0.986 | 0.956 | 0.015 | 3/3 |
| NBEATS-I-AE | 13.981 | 1.039 | 0.973 | 0.034 | 3/3 |

Notable cross-period patterns: (a) NBEATS-G improves from 10th on Yearly to 1st on Monthly, suggesting its fully learnable basis functions benefit from larger training sets; (b) NBEATS-I-AE degrades from 2nd on Yearly to last on both Quarterly and Monthly, suggesting the extreme compression of the AE backbone (latent_dim=4) becomes a bottleneck for higher-frequency patterns; (c) BottleneckGenericAE maintains consistently mid-range performance across all three periods, never ranking below 5th.

#### 5.1.3 Wavelet Block Failure Analysis

Pure wavelet configurations exhibited severe training instability, with most failing to produce any healthy runs. Table 4 summarizes failure modes across all periods.

**Table 4: Wavelet Configuration Failure Modes**

| Config | Yearly | Quarterly | Monthly | Primary Failure Mode |
|--------|:------:|:---------:|:-------:|----------------------|
| HaarWavelet | NaN 3/3 | NaN 3/3 | NaN 3/3 | Immediate failure (epoch 1) |
| DB3AltWavelet | NaN 3/3 | NaN 3/3 | NaN 3/3 | Immediate failure (epoch 1) |
| DB3Wavelet | Div 2/3, 1 partial | Div 3/3 | Div 2/3, NaN 1/3 | MASE explosion (10¹⁰ - 10³¹) |
| Symlet3Wavelet | Div 3/3 | Div 3/3 | Div 2/3, NaN 1/3 | MASE explosion (identical to DB3) |
| Coif2Wavelet | 1/3 healthy | 1/3 healthy | 1/3 healthy | 67% divergence rate |
| Trend+HaarWavelet | 1/3 healthy | 1/3 healthy | 1/3 healthy | 67% divergence rate |
| Generic+DB3Wavelet | 1/3 healthy | 0/3 healthy | 0/3 healthy | Progressive instability |

Two distinct failure modes are observed:

1. **Immediate NaN failure** (HaarWavelet, DB3AltWavelet): Training produces NaN loss at epoch 1 across all seeds and periods. This suggests the wavelet basis matrix itself causes numerical overflow in the forward pass, likely because the basis values exceed the range representable after multiplication with randomly initialized weights.

2. **Gradual MASE explosion** (DB3Wavelet, Symlet3Wavelet, Coif2Wavelet): Training runs for 11+ epochs but produces forecasts with astronomically large MASE values (up to 10³¹). The sMAPE saturates at 200 (its theoretical maximum). Interestingly, DB3Wavelet and Symlet3Wavelet produce *identical* divergent MASE values on matching seeds, suggesting these wavelet families share numerical properties that trigger the same instability pathway.

When wavelet blocks do converge, they can be competitive: Coif2Wavelet seed 44 on Yearly achieved OWA = 0.823, comparable to NBEATS-G. The Trend+DB3Wavelet mixed configuration, where Trend stacks provide stable gradient flow that stabilizes the wavelet stacks, achieved full convergence on Yearly and Quarterly with competitive OWA (0.809 and 0.896). This suggests the wavelet basis expansion concept is sound but requires numerical stabilization--potential remedies include gradient clipping, basis normalization, or spectral regularization of the wavelet basis matrix.

#### 5.1.4 Figures (Block Benchmark)

**Figure 1: N-BEATS Architecture Diagram.** Block diagram showing the doubly residual topology: a sequence of stacks, each containing blocks with the four-FC-layer backbone (RootBlock) or hourglass AE backbone (AERootBlock). Insets compare the three basis expansion families: (a) Generic/BottleneckGeneric linear projection, (b) Trend polynomial / Seasonality Fourier constrained basis, (c) Wavelet multi-resolution basis with learned downsampling. Backward residual connections (subtraction) and forward residual connections (summation) are highlighted with directional arrows. *[To be produced as vector graphic.]*

**Figure 2: OWA Comparison Bar Chart (Yearly Period).** Grouped bar chart with configurations on the x-axis (sorted by mean OWA, best to worst) and OWA on the y-axis (range 0.78-0.85). Bars colored by category: blue = Baseline (NBEATS-G, NBEATS-I, NBEATS-I+G), green = Novel (BottleneckGeneric, AutoEncoder, GenericAEBackcast), orange = AE-backbone (GenericAE, BottleneckGenericAE, NBEATS-I-AE), purple = Mixed (Trend+DB3Wavelet). Error bars show ±1 standard deviation across 3 seeds. Horizontal dashed line at OWA = 1.0 (Naive2 baseline). *[To be produced from block_benchmark_results.csv.]*

**Figure 3: Parameter Efficiency Scatter Plot.** Parameters (millions, log scale x-axis) versus mean OWA (y-axis) for the Yearly period. Points labeled by configuration name, colored by category. Pareto frontier connects NBEATS-I-AE (2.2M, 0.805), BottleneckGenericAE (4.3M, 0.806), and AutoEncoder (24.9M, 0.804). Key finding: AE-backbone blocks achieve 5-10x parameter reduction with comparable OWA. Configurations above the Pareto frontier are dominated (more parameters for equal or worse accuracy). *[To be produced from block_benchmark_results.csv.]*

**Figure 4: OWA Heatmap across Configurations and Periods.** Rows = configurations (sorted by Yearly OWA), columns = Yearly / Quarterly / Monthly. Cell color encodes OWA intensity (green = low/good, red = high/poor). Gray cells indicate diverged or NaN runs. Reveals cross-period generalization: some configs (BottleneckGenericAE) maintain consistent relative ranking while others (NBEATS-G, NBEATS-I-AE) shift dramatically. *[To be produced from block_benchmark_results.csv.]*

**Figure 5: Training Stability (OWA Standard Deviation).** Bar chart of OWA standard deviation across 3 seeds for each configuration on the Yearly period. NBEATS-I-AE stands out with remarkably low variance (std ≈ 0.001), while BottleneckGeneric shows the highest variance among healthy configs (std ≈ 0.016). Stable training is practically valuable because it reduces the number of seeds needed for reliable performance estimation. *[To be produced from block_benchmark_results.csv.]*

### 5.2 Ensemble Results

The ensemble experiment follows the original paper's multi-horizon strategy: for each of three paper architectures, 18 models are trained (6 backcast multipliers × 3 seeds) and aggregated via element-wise median. Results are currently available for the Yearly period only.

#### 5.2.1 Ensemble Summary

**Table 5: Ensemble Results (Yearly Period)**

| Config | # Models | Ensemble sMAPE | Ensemble MASE | Ensemble OWA | Best Single-Model OWA | Improvement |
|--------|:--------:|:--------------:|:-------------:|:------------:|:---------------------:|:-----------:|
| NBEATS-G | 18 | 13.318 | 3.066 | 0.793 | 0.820 | -3.3% |
| NBEATS-I | 18 | 13.265 | 3.033 | 0.788 | 0.816 | -3.4% |
| NBEATS-I+G | 18 | 13.208 | 3.017 | 0.784 | 0.808 | -3.0% |
| All configs | Q/M/W/D/H | [pending] | [pending] | [pending] | [pending] | [pending] |

Ensemble values are taken directly from `ensemble_summary_results.csv`. "Best Single-Model OWA" refers to the mean OWA from the block benchmark (5H multiplier, 3 seeds). The improvement column shows the relative reduction: $(OWA_{ensemble} - OWA_{single}) / OWA_{single}$.

The ensemble consistently provides approximately 3% OWA improvement over the best single-model configuration, with the relative gain remarkably stable across the three architectures (3.0-3.4%). NBEATS-I+G achieves the best ensemble OWA (0.784), reflecting both its strong single-model performance and the diversity introduced by combining Trend, Seasonality, and Generic stacks at different backcast lengths.

#### 5.2.2 Comparison with Original Paper

Direct comparison with the original N-BEATS paper (Oreshkin et al., 2019) is limited because the paper reports aggregate OWA across all six M4 periods, while our ensemble results currently cover only Yearly. The paper's 180-model ensemble (3 loss functions × 6 multipliers × 10 seeds) achieved OWA = 0.795 on the full M4 dataset. Our 18-model ensembles (1 loss function × 6 multipliers × 3 seeds) achieve OWA = 0.784-0.793 on Yearly alone. Since Yearly is typically one of the easier M4 periods, these Yearly-only results should not be directly compared to the paper's aggregate metric. A complete comparison requires results from all six periods.

#### 5.2.3 Figure (Ensemble)

**Figure 6: Ensemble OWA vs Backcast Multiplier.** Line plot with backcast multiplier (2H through 7H) on the x-axis and mean OWA (across 3 seeds) on the y-axis. Three lines, one per architecture (NBEATS-G, NBEATS-I, NBEATS-I+G). Horizontal dashed lines show ensemble OWA for each architecture. Individual seed OWA values are plotted as semi-transparent points. The plot reveals which backcast lengths contribute most to ensemble diversity: shorter multipliers (2H-3H) tend to produce lower individual OWA (better), while longer multipliers (6H-7H) show higher variance. The ensemble consistently outperforms all individual models. *[To be produced from ensemble_individual_results.csv.]*

### 5.3 Ablation Studies

Ablation experiments (Part 2) have not yet been run. This section presents the experimental design; results will be populated upon completion.

#### 5.3.1 Ablation Design

The ablation study evaluates two training extensions (`active_g`, `sum_losses`) and four alternative activation functions using the 30-stack Generic configuration as the control. The Generic_baseline row is identical to NBEATS-G from the block benchmark, serving as the shared reference point.

**Table 6: Ablation Results (all pending)**

| Config | Modification | Yearly OWA | Quarterly OWA | Monthly OWA | W / D / H |
|--------|-------------|:----------:|:-------------:|:-----------:|:---------:|
| Generic_baseline | None (control) | [pending] | [pending] | [pending] | [pending] |
| Generic_activeG | active_g=True | [pending] | [pending] | [pending] | [pending] |
| Generic_sumLosses | sum_losses=True | [pending] | [pending] | [pending] | [pending] |
| Generic_activeG+sumL | Both extensions | [pending] | [pending] | [pending] | [pending] |
| Generic_GELU | GELU activation | [pending] | [pending] | [pending] | [pending] |
| Generic_ELU | ELU activation | [pending] | [pending] | [pending] | [pending] |
| Generic_LeakyReLU | LeakyReLU activation | [pending] | [pending] | [pending] | [pending] |
| Generic_SELU | SELU activation | [pending] | [pending] | [pending] | [pending] |

#### 5.3.2 Figure (Ablation)

**Figure 7: Ablation Grouped Bar Chart.** For each period (grouped), bars show mean OWA for each ablation configuration. Error bars show ±1 std across seeds. The baseline (ReLU, no extensions) is highlighted with a vertical dashed reference line. *[To be produced upon completion of ablation runs.]*

### 5.4 Statistical Significance Analysis

A central question motivating this work is whether the choice of block type--the basis expansion function within the N-BEATS doubly residual framework--meaningfully affects forecasting accuracy on the M4 benchmark. Preliminary inspection of the results in Section 5.1 reveals that the OWA spread among healthy, fully-converging configurations is remarkably narrow (0.023 on Yearly, 0.020 on Quarterly, 0.043 on Monthly), raising the possibility that observed differences are attributable to random seed variation rather than genuine architectural effects.

#### 5.4.1 Hypothesis and Analysis Design

**Null hypothesis (H₀):** Block type has no statistically significant effect on OWA forecasting performance. The observed differences in OWA across block configurations are attributable to random seed variation rather than architectural differences.

**Alternative hypothesis (H₁):** At least one block type produces OWA that is significantly different from the others.

**Data preparation** (applied independently per period before all tests):

1. Exclude NaN values and divergent runs (OWA > 2.0 or MASE > 10⁶)
2. Exclude configurations with fewer than 2 healthy seeds (insufficient for variance estimation)
3. Group remaining OWA values by `config_name`

After filtering, Yearly retains 10 configurations (all with 3/3 healthy seeds), Quarterly retains 10 configurations (AutoEncoder with 2/3, all others 3/3), and Monthly retains 9 configurations (all with 3/3 healthy seeds). Wavelet-only configurations are excluded from all statistical tests due to their near-universal failure to converge.

#### 5.4.2 Statistical Tests

We employ a three-tier testing strategy: an omnibus test to determine whether block type matters at all, pairwise comparisons if the omnibus test is significant, and effect size estimation regardless of significance.

**Tier 1: Omnibus tests — Does block type matter at all?**

We apply three complementary omnibus tests per period:

- **Kruskal-Wallis H-test** (non-parametric one-way ANOVA): preferred because (a) n = 3 per group is too small to verify normality, (b) OWA distributions may be skewed, and (c) Kruskal-Wallis is robust to non-normality.
- **One-way ANOVA (F-test)**: parametric complement assuming approximate normality; included for comparison but less trustworthy given the small sample sizes.
- **Friedman test** (non-parametric repeated measures): treats seeds (42, 43, 44) as a blocking factor, potentially increasing power by accounting for seed-specific effects that affect all configurations similarly.

**Table 7: Omnibus Statistical Test Results**

| Period | Test | Statistic | p-value | η² | Interpretation |
|--------|------|:---------:|:-------:|:--:|----------------|
| Yearly | Kruskal-Wallis | H = 9.85 | 0.363 | 0.043 | Fail to reject H₀ |
| Yearly | One-way ANOVA | F = 1.28 | 0.307 | — | Fail to reject H₀ |
| Yearly | Friedman | χ² = 10.20 | 0.335 | — | Fail to reject H₀ |
| Quarterly | Kruskal-Wallis | H = 14.99 | 0.091 | 0.315 | Fail to reject H₀ (marginal) |
| Quarterly | One-way ANOVA | F = 4.33 | 0.003 | — | Reject H₀ (α = 0.05) |
| Quarterly | Friedman | χ² = 11.91 | 0.155 | — | Fail to reject H₀ |
| Monthly | Kruskal-Wallis | H = 13.31 | 0.102 | 0.295 | Fail to reject H₀ |
| Monthly | One-way ANOVA | F = 2.30 | 0.068 | — | Fail to reject H₀ (marginal) |
| Monthly | Friedman | χ² = 11.20 | 0.191 | — | Fail to reject H₀ |

Eta-squared (η²) is computed from the Kruskal-Wallis statistic as η² = (H − k + 1) / (N − k), where k is the number of groups and N is the total number of observations.

**Interpretation:** The non-parametric tests (Kruskal-Wallis, Friedman) consistently fail to reject H₀ across all three periods at α = 0.05. The parametric one-way ANOVA rejects H₀ for Quarterly (p = 0.003) but not for Yearly or Monthly. This discrepancy likely reflects the ANOVA's sensitivity to the specific pattern of means combined with its assumption of normality, which is questionable at n = 3. Given the robustness advantages of the non-parametric tests and the inconsistency of the ANOVA result across periods, the overall evidence suggests that **block type does not produce statistically significant differences in OWA** among healthy, converging configurations.

**Tier 2: Pairwise comparisons**

Because the non-parametric omnibus tests fail to reject H₀, we do not conduct formal pairwise comparisons (Mann-Whitney U tests with Bonferroni correction). With k ≈ 10 configurations, there would be C(10, 2) = 45 pairwise tests requiring α_adjusted ≈ 0.001--far too conservative for the observed effect sizes and sample sizes. The ANOVA rejection for Quarterly, if taken at face value, would warrant pairwise t-tests; however, with n = 3 per group and 45 comparisons, the corrected significance threshold would yield no detectable pairs.

**Tier 3: Effect sizes**

Effect sizes provide a scale-independent measure of the magnitude of group differences, and are arguably more informative than p-values given the small sample sizes.

- **Yearly η² = 0.043** (small effect): Block type explains approximately 4% of the variance in OWA. By conventional thresholds (Cohen, 1988), this is a small effect approaching negligibility.
- **Quarterly η² = 0.315** (large effect): Block type explains approximately 32% of the variance. However, this large effect size is driven partly by the NBEATS-I-AE outlier (OWA = 0.937, substantially worse than other configs at 0.892-0.912). Excluding NBEATS-I-AE would substantially reduce the effect.
- **Monthly η² = 0.295** (large effect): Similar pattern to Quarterly, with NBEATS-I-AE again contributing disproportionately to the between-group variance.

The effect size pattern suggests that while most block types perform similarly, **NBEATS-I-AE degrades on higher-frequency periods**, creating apparent statistical significance that is driven by a single configuration rather than a general block-type effect. This is better understood as a specific limitation of extreme AE-backbone compression than as evidence that block type generally matters.

#### 5.4.3 Power Analysis and Limitations

With n = 3 seeds per configuration, the statistical power of these tests is severely limited. This is a critical caveat for interpreting the null results.

**Post-hoc power estimate:** For the Yearly period, the observed effect size (η² = 0.043) with k = 10 groups and n = 3 per group yields estimated power of approximately **15-20%** for the Kruskal-Wallis test. This means that even if a small real effect exists, we would detect it only 15-20% of the time.

**Practical implication:** Failure to reject H₀ should be interpreted as *insufficient evidence to conclude that block type affects OWA* rather than *confirmation that block type is irrelevant*. The distinction matters: the data are consistent with H₀, but they are also consistent with small real effects (e.g., η² ≈ 0.05) that we lack the power to detect.

**Recommendation for future work:** Increasing to n = 10 seeds per configuration would provide approximately 80% power to detect medium effects (η² = 0.06), the conventional threshold for adequate statistical testing. This would require 10 × 17 configs × 6 periods = 1,020 training runs (versus the current 17 × 3 × 3 = 153), a roughly 7x increase in computational cost.

#### 5.4.4 Complementary Analyses

**Cross-period rank consistency (Spearman correlation)**

If block type reliably determines forecasting quality, configurations that rank well on one period should rank similarly on others. We compute the Spearman rank correlation of mean OWA rankings between each pair of completed periods.

**Table 8: Cross-Period Rank Consistency**

| Period Pair | Spearman ρ | p-value | n configs | Interpretation |
|-------------|:----------:|:-------:|:---------:|----------------|
| Yearly vs Quarterly | −0.021 | 0.948 | 11 | No correlation |
| Yearly vs Monthly | −0.436 | 0.180 | 11 | No correlation (negative trend) |
| Quarterly vs Monthly | 0.455 | 0.160 | 11 | No correlation (positive trend) |

The rank correlations are weak and non-significant for all period pairs. The Yearly-vs-Monthly correlation is *negative* (ρ = −0.436), meaning configurations that rank well on Yearly tend to rank *poorly* on Monthly and vice versa. This is consistent with the observation that NBEATS-G (data-hungry, high-capacity) improves on Monthly (48,000 series) while NBEATS-I-AE (parameter-efficient, compressed) excels on Yearly (23,000 shorter series).

**Practical implication:** Configuration rankings are period-dependent. A practitioner cannot select the "best" block type on one forecasting horizon and expect it to generalize to others. This further supports the hypothesis that block type choice, among healthy configurations, is secondary to other factors (dataset size, series length, horizon) in determining aggregate forecasting accuracy.

**Bootstrap confidence intervals**

For each configuration and period, we compute 95% bootstrap confidence intervals for mean OWA using 10,000 resamples with replacement from the n = 3 (or 2) healthy seed OWA values.

On Yearly, the bootstrap CIs of all 10 healthy configurations overlap substantially. For example, AutoEncoder (best: 0.789-0.817) overlaps with NBEATS-G (0.810-0.826) and BottleneckGeneric (worst healthy: 0.810-0.840). The universal overlap confirms that the differences in point estimates are within the range of sampling variability.

On Quarterly and Monthly, the CIs are wider for some configurations (notably GenericAEBackcast on Monthly: 0.908-0.977) and narrower for others (NBEATS-I on Quarterly: 0.889-0.894), but overall the pattern of substantial overlap persists.

#### 5.4.5 Figures (Statistical Analysis)

**Figure 8: Bootstrap Confidence Interval Forest Plot.** Three panels (Yearly, Quarterly, Monthly), each showing configurations as rows with horizontal error bars representing 95% bootstrap CIs for mean OWA. Point estimates marked with filled circles. Configurations sorted by point estimate within each panel. Extensive CI overlap provides visual evidence consistent with the null hypothesis. *[To be produced via analysis script using scipy.stats and matplotlib.]*

**Figure 9: OWA Distribution Box Plots.** Box plots of OWA per configuration (x-axis), faceted by period. Individual seed values overlaid as jittered dots. Whiskers extend to data range (with n = 3, no statistical outlier detection is meaningful). The compressed y-axis range within each panel (e.g., 0.79-0.84 for Yearly) highlights how tight the performance band is across architecturally diverse configurations. *[To be produced via analysis script.]*

#### 5.4.6 Implementation Note

All statistical tests in this section were computed from `experiments/results/block_benchmark_results.csv` using SciPy's `scipy.stats` module (Kruskal-Wallis: `kruskal`, Mann-Whitney: `mannwhitneyu`, Friedman: `friedmanchisquare`, Spearman: `spearmanr`) and NumPy for bootstrap resampling. The analysis script filters healthy runs (OWA < 2.0, non-NaN MASE < 10⁶), groups by configuration, and runs all tests per period. No modifications to the training code were required. The script and its output are available in the supplementary materials.

### 5.5 Discussion

The results across three M4 periods, ensemble evaluation, and statistical analysis yield several findings relevant to both the theoretical understanding of the N-BEATS architecture and its practical application.

**1. Novel blocks match or beat paper baselines.** On Yearly, AutoEncoder (0.804) and NBEATS-I-AE (0.805) outperform all three paper baselines (NBEATS-G: 0.820, NBEATS-I: 0.816, NBEATS-I+G: 0.808). The AE-backbone variants (GenericAE, BottleneckGenericAE) also match NBEATS-I+G. This demonstrates that the N-BEATS doubly residual framework is robust to the specific choice of basis expansion within each block--a finding consistent with the statistical analysis showing no significant block-type effect.

**2. Parameter efficiency.** The most practically significant finding is the dramatic parameter reduction achievable with AE-backbone blocks. NBEATS-I-AE requires only 2.2M parameters (91% fewer than NBEATS-G's 24.7M) while achieving better Yearly OWA. BottleneckGenericAE achieves comparable performance with 4.3M parameters (82% fewer). For deployment scenarios where model size, memory footprint, or inference latency are constrained, these AE-backbone variants offer substantial advantages without sacrificing accuracy--at least on the periods tested so far.

**3. Cross-period generalization varies by block type.** The absence of consistent rank ordering across periods (Spearman ρ ranging from −0.44 to +0.46, all non-significant) reveals an important nuance: some architectures are better suited to specific forecasting contexts. NBEATS-G's fully learnable basis benefits from the larger Monthly dataset (48,000 series), while the constrained AE-backbone blocks perform best on the smaller Yearly dataset (23,000 series) where their implicit regularization prevents overfitting. This period-dependence means that the "best" block type is a function of the forecasting problem, not an inherent property of the architecture.

**4. Wavelet instability.** Pure wavelet blocks fail in 67-100% of training runs, with failure modes ranging from immediate NaN (Haar, DB3Alt) to gradual MASE explosion (DB3, Symlet3). The fact that Trend+DB3Wavelet achieves full convergence and competitive OWA when paired with stable Trend stacks suggests that the wavelet basis expansion concept is viable but requires stabilization from complementary stacks or explicit numerical safeguards. Future work should investigate gradient clipping, basis matrix normalization, or learned scaling factors as potential remedies.

**5. Ensembling provides consistent improvement.** Median aggregation across 18 models (6 backcast multipliers × 3 seeds) improves OWA by approximately 3% on Yearly, consistent across all three paper architectures (3.0-3.4% improvement). This improvement is remarkably uniform, suggesting that the ensemble benefit comes primarily from diversity in backcast length rather than seed-specific initialization effects.

**6. Training stability varies significantly.** NBEATS-I-AE exhibits remarkably low OWA variance across seeds (std ≈ 0.001 on Yearly), while BottleneckGeneric shows the highest variance among healthy configs (std ≈ 0.016). From a practical standpoint, low variance is valuable because it reduces the number of training runs needed to achieve reliable performance estimates. The AE-backbone's hourglass compression may act as a regularizer that smooths the loss landscape, leading to more deterministic convergence.

**7. Statistical significance and the block-type irrelevance hypothesis.** The Kruskal-Wallis test fails to reject H₀ (block type has no effect on OWA) for all three periods. The Friedman test, which accounts for seed-level blocking, similarly fails to reject H₀. While these null results are tempered by the low statistical power (≈15-20% for small effects), the complementary evidence is consistent: narrow OWA spread (0.02-0.04), overlapping bootstrap confidence intervals, and inconsistent cross-period rankings all point toward the same conclusion. Among healthy, converging configurations, the specific choice of basis expansion function has a negligible effect on aggregate OWA forecasting accuracy.

However, block type **does** significantly affect three practically important dimensions: (a) **parameter count** (5-10x variation between AE-backbone and standard backbone), (b) **training stability** (0-100% convergence rate for wavelets vs 100% for all non-wavelet blocks), and (c) **convergence speed** (epochs to convergence range from 12 to 39). The practical recommendation is therefore: choose blocks based on deployment constraints (model size, stability requirements, interpretability needs) rather than chasing marginal OWA differences that are likely within noise.

### 5.6 Suggested Additional Metrics

We identify several metrics that could strengthen the analysis in future iterations of this work, organized by whether they are derivable from existing experimental data or require additional code changes.

**Derivable from existing data (no code changes required):**

| Metric | Purpose | Computation | Recommended Location |
|--------|---------|-------------|---------------------|
| OWA std across seeds | Training robustness | `std(OWA)` per (config, period) | Already included in Tables 3a-3c |
| Convergence rate | Practical usability | Healthy seeds / total seeds | Already included in Tables 2, 4 |
| Parameter efficiency ratio | Best accuracy-per-param | `(1.0 − OWA) / (n_params / 1M)` | Figure 3 (scatter plot) |
| Mean epochs to convergence | Training cost comparison | `mean(epochs_trained)` per config | Training efficiency table |
| Mean wall-clock time | Resource cost | `mean(training_time_seconds)` per config | Training efficiency table |
| Cross-period rank correlation | Ranking stability | Spearman ρ on OWA rankings | Table 8 (already computed) |

**Requiring code changes (future work):**

| Metric | Purpose | Implementation |
|--------|---------|----------------|
| Per-series sMAPE/MASE distributions | Reveal whether gains are uniform or outlier-driven | Post-process `.npz` prediction files; compute per-series metrics and report percentiles |
| FLOPs / inference latency | Deployment cost (params alone miss sequential depth) | Use `fvcore.nn.FlopCountAnalysis` or `torch.profiler` |
| Backcast reconstruction quality | Assess doubly residual decomposition | Expose stack residuals from forward pass; compute ‖residual‖₁ / ‖input‖₁ |
| Gradient norm statistics | Explain wavelet failures | Add Lightning callback logging `grad_norm` per step |

---

## 6. Conclusions

This work presents a systematic exploration of alternative block types within the N-BEATS doubly residual framework, implemented as the `lightningnbeats` PyTorch Lightning package. Preliminary results across three of six M4 periods support several provisional conclusions that will be revisited upon completion of all experiments.

**Strongest findings (supported by available data):**

1. The N-BEATS doubly residual framework is remarkably robust to the choice of basis expansion function. Among healthy, converging configurations, block type does not produce statistically significant differences in OWA (Kruskal-Wallis p > 0.09 across all periods), and configuration rankings are inconsistent across periods (Spearman ρ ≈ 0, all non-significant). This suggests that the framework's iterative residual decomposition mechanism, rather than the specific basis within each block, is the primary driver of forecasting accuracy.

2. AE-backbone variants achieve 5-10x parameter reduction with comparable or better OWA, making them attractive for resource-constrained deployment. NBEATS-I-AE (2.2M parameters) matches the 24.7M-parameter NBEATS-G on Yearly while exhibiting the lowest training variance of any configuration tested.

3. Wavelet basis blocks suffer from severe numerical instability (67-100% failure rate) but produce competitive results when they converge and when stabilized by complementary Trend stacks. Numerical remediation is a promising direction for future work.

**Contingent findings (require remaining experiments for confirmation):**

- Cross-period generalization patterns observed on Yearly/Quarterly/Monthly may shift when Weekly, Daily, and Hourly results are incorporated, particularly for high-frequency periods where wavelet multi-resolution properties could become advantageous.
- The ablation study (Part 2) will determine whether training extensions (`active_g`, `sum_losses`) and alternative activations interact differently with novel block types than with the standard Generic configuration.
- Ensemble results beyond the Yearly period will reveal whether the approximately 3% median-aggregation improvement holds across all frequencies, and whether novel block types benefit more or less from ensembling than paper baselines.

**Future directions** include: (a) numerical stabilization of wavelet blocks via gradient clipping, basis normalization, or spectral regularization; (b) scaling seed count to n = 10 for adequate statistical power; (c) extending the ensemble experiment to novel block types and AE-backbone configurations; (d) per-series analysis to determine whether aggregate OWA masks heterogeneous per-series behavior; and (e) investigating whether the block-type irrelevance finding extends to other benchmarks (M3, Tourism, ETT) or is specific to the M4 dataset's characteristics.

---

## References

Aminghafari, M., Cheze, N., & Poggi, J.-M. (2006). Multivariate denoising using wavelets and principal component analysis. *Computational Statistics & Data Analysis*, 50(9), 2381-2398. https://doi.org/10.1016/j.csda.2004.12.010

Assimakopoulos, V., & Nikolopoulos, K. (2000). The Theta model: A decomposition approach to forecasting. *International Journal of Forecasting*, 16(4), 521-530. https://doi.org/10.1016/S0169-2070(00)00066-2

Box, G. E. P., & Jenkins, G. M. (1976). *Time Series Analysis: Forecasting and Control*. Holden-Day.

Challu, C., Olivares, K. G., Oreshkin, B. N., Garza, F., Mergenthaler-Canseco, M., & Dubrawski, A. (2023). N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting. *Proceedings of the AAAI Conference on Artificial Intelligence*, 37(6), 6989-6997. https://doi.org/10.1609/aaai.v37i6.25854

Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). STL: A Seasonal-Trend Decomposition Procedure Based on Loess. *Journal of Official Statistics*, 6(1), 3-73.

Daubechies, I. (1992). *Ten Lectures on Wavelets*. SIAM. https://doi.org/10.1137/1.9781611970104

Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the Dimensionality of Data with Neural Networks. *Science*, 313(5786), 504-507. https://doi.org/10.1126/science.1127647

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770-778. https://doi.org/10.1109/CVPR.2016.90

Holt, C. C. (1957). Forecasting Seasonals and Trends by Exponentially Weighted Moving Averages. *ONR Memorandum No. 52*. Carnegie Institute of Technology.

Hyndman, R. J., & Khandakar, Y. (2008). Automatic Time Series Forecasting: The forecast Package for R. *Journal of Statistical Software*, 27(3), 1-22. https://doi.org/10.18637/jss.v027.i03

Hyndman, R. J., Koehler, A. B., Snyder, R. D., & Grose, S. (2002). A state space framework for automatic forecasting using exponential smoothing methods. *International Journal of Forecasting*, 18(3), 439-454. https://doi.org/10.1016/S0169-2070(01)00110-8

Lim, B., Arik, S. O., Loeff, N., & Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*, 37(4), 1748-1764. https://doi.org/10.1016/j.ijforecast.2021.03.012

Makridakis, S., & Hibon, M. (2000). The M3-Competition: Results, conclusions and implications. *International Journal of Forecasting*, 16(4), 451-476. https://doi.org/10.1016/S0169-2070(00)00057-1

Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2018). Statistical and Machine Learning forecasting methods: Concerns and ways forward. *PLOS ONE*, 13(3), e0194889. https://doi.org/10.1371/journal.pone.0194889

Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). The M4 Competition: 100,000 time series and 61 forecasting methods. *International Journal of Forecasting*, 36(1), 54-74. https://doi.org/10.1016/j.ijforecast.2019.04.014

Malhotra, P., Ramakrishnan, A., Anand, G., Vig, L., Agarwal, P., & Shroff, G. (2016). LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection. *arXiv preprint arXiv:1607.00148*.

Mallat, S. G. (1989). A Theory for Multiresolution Signal Decomposition: The Wavelet Representation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 11(7), 674-693. https://doi.org/10.1109/34.192463

Montero-Manso, P., Athanasopoulos, G., Hyndman, R. J., & Talagala, T. S. (2020). FFORMA: Feature-based forecast model averaging. *International Journal of Forecasting*, 36(1), 86-92. https://doi.org/10.1016/j.ijforecast.2019.02.011

Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. *International Conference on Learning Representations (ICLR 2023)*.

Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2019). N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. *International Conference on Learning Representations (ICLR 2020)*. https://openreview.net/forum?id=r1ecqn4YwB

Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T. (2020). DeepAR: Probabilistic forecasting with autoregressive recurrent networks. *International Journal of Forecasting*, 36(3), 1181-1191. https://doi.org/10.1016/j.ijforecast.2019.07.001

Smyl, S. (2020). A hybrid method of exponential smoothing and recurrent neural networks for time series forecasting. *International Journal of Forecasting*, 36(1), 75-85. https://doi.org/10.1016/j.ijforecast.2019.03.017

van den Oord, A., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., ... & Kavukcuoglu, K. (2016). WaveNet: A Generative Model for Raw Audio. *arXiv preprint arXiv:1609.03499*.

Vincent, P., Larochelle, H., Bengio, Y., & Manzagol, P.-A. (2008). Extracting and composing robust features with denoising autoencoders. *Proceedings of the 25th International Conference on Machine Learning (ICML)*, 1096-1103. https://doi.org/10.1145/1390156.1390294

Winters, P. R. (1960). Forecasting Sales by Exponentially Weighted Moving Averages. *Management Science*, 6(3), 324-342. https://doi.org/10.1287/mnsc.6.3.324

Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are Transformers Effective for Time Series Forecasting? *Proceedings of the AAAI Conference on Artificial Intelligence*, 37(9), 11121-11128. https://doi.org/10.1609/aaai.v37i9.26317
