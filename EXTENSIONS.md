# N-BEATS Lightning: Novel Extensions Beyond the Original Paper

This document describes features in this implementation that extend the original N-BEATS architecture as defined in Oreshkin et al. (2019), "N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting" (ICLR 2020, arXiv:1905.10437).

---

## 1. `active_g` — Activation After Basis Expansion

**What:** Optional activation function applied to the output of the basis expansion functions g_b and g_f.

**Paper difference:** The paper specifies no activation after basis expansion. The official ServiceNow implementation has no such feature.

**Mathematical formulation:**

Standard (paper):
```
x_hat = g_b(theta_b)
y_hat = g_f(theta_f)
```

With `active_g=True`:
```
x_hat = sigma(g_b(theta_b))
y_hat = sigma(g_f(theta_f))
```

where `sigma` is the configurable activation function (ReLU, GELU, etc.).

**Motivation:** In the paper's interpretable blocks (Trend, Seasonality), the basis expansion inherently applies non-linear transformations (polynomial, Fourier). Two successive linear layers without activation (theta computation + basis expansion) limit Generic blocks to linear functions of the FC output. Adding activation makes Generic blocks structurally consistent with interpretable blocks and empirically improves convergence.

**Code:** `blocks/blocks.py` — present in `BottleneckGeneric`, `Generic`, `AutoEncoder`, `GenericAEBackcast`, and their AE variants.

**Parameter:** `active_g: bool = False` (default preserves paper behavior)

---

## 2. `sum_losses` — Combined Forecast + Backcast Loss

**What:** Training objective that combines forecast loss with a weighted backcast reconstruction loss.

**Paper difference:** The paper trains on forecast loss only.

**Mathematical formulation:**
```
L = L_forecast(y_hat, y) + 0.25 * L_backcast(x_residual, 0)
```

where `x_residual` is the final residual after all blocks have subtracted their backcasts. The backcast loss penalizes non-zero residuals, pushing the stack of blocks to fully decompose the input signal.

**Motivation:** Directly optimizing backcast quality incentivizes each block to better decompose the input signal, producing cleaner residuals for downstream blocks. The 0.25 weighting prevents the backcast objective from dominating the forecast objective.

**Code:** `models.py` — `training_step`, `validation_step`, `test_step`

**Parameter:** `sum_losses: bool = False`

---

## 3. BottleneckGeneric Block (Low-Rank Factorized Basis)

**What:** A Generic block variant that uses separate theta computation and basis expansion layers with an intermediate bottleneck dimension (`thetas_dim`), instead of the paper's single linear projection.

**Paper's Generic architecture (5-layer path):**
```
4 FC+ReLU -> Linear(units, B+H) -> slice into backcast[:B] and forecast[B:]
```

**BottleneckGeneric architecture (6-layer path):**
```
4 FC+ReLU -> backcast_linear(units, d) -> backcast_g(d, B)
          -> forecast_linear(units, d) -> forecast_g(d, H)
```

where `d = thetas_dim`.

**Mathematical formulation:** The two-layer projection is equivalent to a rank-d factorization:
```
x_hat = V_b * W_b * h_4
```
where `V_b in R^{B x d}`, `W_b in R^{d x units}`, and `d = thetas_dim`. The product `V_b * W_b` has rank <= `thetas_dim`.

**Motivation:** The bottleneck regularizes the learned basis by limiting its rank, providing a tunable knob (`thetas_dim`) to control basis complexity. This is analogous to the theta dimension in Trend and Seasonality blocks, which use low-dimensional parameterizations (polynomial degree, number of Fourier harmonics) to constrain the function space.

**Code:** `blocks/blocks.py` — `BottleneckGeneric` (RootBlock backbone), `BottleneckGenericAE` (AERootBlock backbone)

**Block names:** `"BottleneckGeneric"`, `"BottleneckGenericAE"`

---

## 4. AutoEncoder Block

**What:** A block with explicit encoder-decoder branches for both backcast and forecast paths.

**Paper difference:** The paper uses linear theta + basis structure for all blocks.

**Mathematical formulation:**
```
z_b = ReLU(W_enc_b * h_4)                  [encode: units -> thetas_dim]
x_hat = W_dec2_b * ReLU(W_dec1_b * z_b)    [decode: thetas_dim -> units -> B]

z_f = ReLU(W_enc_f * h_4)                  [encode: units -> thetas_dim]
y_hat = W_dec2_f * ReLU(W_dec1_f * z_f)    [decode: thetas_dim -> units -> H]
```

**Motivation:** The information bottleneck forces a compressed representation of the input signal, potentially improving generalization on noisy data. Unlike the BottleneckGeneric block, the decoder includes a non-linear activation between layers.

**Code:** `blocks/blocks.py` — `AutoEncoder` (RootBlock backbone), `AutoEncoderAE` (AERootBlock backbone)

**Block names:** `"AutoEncoder"`, `"AutoEncoderAE"`

---

## 5. AERootBlock — Bottleneck Pre-Split Autoencoder Backbone

**What:** Replaces the paper's 4 uniform-width FC layers with a funnel-expand (encoder-decoder) structure as the shared backbone before the backcast/forecast split.

**Paper's backbone:**
```
h_1 = ReLU(W_1 * x),   W_1 in R^{units x B}
h_2 = ReLU(W_2 * h_1),  W_2 in R^{units x units}
h_3 = ReLU(W_3 * h_2),  W_3 in R^{units x units}
h_4 = ReLU(W_4 * h_3),  W_4 in R^{units x units}
```

**AERootBlock backbone:**
```
h_1 = ReLU(W_1 * x),   W_1 in R^{(units/2) x B}
h_2 = ReLU(W_2 * h_1),  W_2 in R^{d x (units/2)},  d = latent_dim
h_3 = ReLU(W_3 * h_2),  W_3 in R^{(units/2) x d}
h_4 = ReLU(W_4 * h_3),  W_4 in R^{units x (units/2)}
```

**Derived variants:** All blocks with "AE" suffix use this backbone: `GenericAE`, `BottleneckGenericAE`, `TrendAE`, `SeasonalityAE`, `AutoEncoderAE`, `GenericAEBackcastAE`.

**Motivation:** The bottleneck in the pre-split layers forces extraction of salient features before the backcast/forecast split. This may help with noisy or highly variable data by acting as a denoising mechanism.

**Code:** `blocks/blocks.py` — `AERootBlock`

**Parameter:** `latent_dim: int = 5`

---

## 6. GenericAEBackcast — Asymmetric Backcast/Forecast Architecture

**What:** Uses an AutoEncoder structure for the backcast path and a standard Generic (theta + linear expansion) structure for the forecast path.

**Paper difference:** The paper uses symmetric structure for both branches.

**Mathematical formulation:**
```
Backcast (AutoEncoder path):
  z_b = ReLU(W_enc_b * h_4)
  h_b = ReLU(W_dec_b * z_b)
  x_hat = W_out_b * h_b

Forecast (Generic path):
  theta_f = W_theta_f * h_4
  y_hat = W_g_f * theta_f              [bias=False]
```

**Motivation:** Backcast (reconstruction) benefits from the denoising properties of an AE bottleneck, while forecast benefits from the unconstrained expressiveness of the Generic architecture.

**Code:** `blocks/blocks.py` — `GenericAEBackcast` (RootBlock backbone), `GenericAEBackcastAE` (AERootBlock backbone)

**Block names:** `"GenericAEBackcast"`, `"GenericAEBackcastAE"`

---

## 7. Wavelet Basis Expansion Blocks

**What:** Blocks that use wavelet basis functions replacing the polynomial (Trend) or Fourier (Seasonality) bases defined in the paper. This is the largest novel contribution.

**Paper difference:** The paper defines only polynomial and Fourier bases.

### Architecture Variant 1: `Wavelet` (Square Basis + Learned Downsampling)

```
theta_b = W_b * h_4                     [linear: units -> basis_dim]
z_b = theta_b * Phi                     [Phi in R^{basis_dim x basis_dim}, fixed wavelet basis]
x_hat = V_b * z_b                       [linear: basis_dim -> B, bias=False, learned]
```

Where `Phi` is constructed from the wavelet's scaling (phi) and mother (psi) functions resampled to `basis_dim` points, with columns being circular shifts.

### Architecture Variant 2: `AltWavelet` (Rectangular Basis, Direct Output)

```
theta_b = W_b * h_4                     [linear: units -> basis_dim]
x_hat = theta_b * Psi                   [Psi in R^{basis_dim x B}, fixed wavelet basis]
```

No learned downsampling — the rectangular basis directly maps to target length.

### Wavelet Families Supported

| Family | Classes | Wavelet Type |
|--------|---------|-------------|
| Haar | `HaarWavelet`, `HaarAltWavelet` | `haar` |
| Daubechies | `DB2Wavelet`/`AltWavelet`, `DB3...`, `DB4...`, `DB10...`, `DB20AltWavelet` | `db2`-`db20` |
| Coiflets | `Coif1Wavelet`/`AltWavelet`, `Coif2...`, `Coif3...`, `Coif10...` | `coif1`-`coif10` |
| Symlets | `Symlet2Wavelet`/`AltWavelet`, `Symlet3Wavelet`, `Symlet10Wavelet`, `Symlet20Wavelet` | `sym2`-`sym20` |

### Motivation

1. **Multi-resolution analysis:** Captures features at multiple time scales simultaneously, unlike the single-scale representations of polynomial or Fourier bases.
2. **Time-frequency localization:** Unlike the global Fourier basis (Seasonality block), wavelets can represent transient events and local patterns that are not periodic.
3. **Non-trainable basis preserves interpretability:** Like Trend and Seasonality blocks, the wavelet basis is fixed (non-learnable), maintaining interpretability of the theta parameters.
4. **Different wavelet families suit different data characteristics:** Haar for piecewise constant signals, Daubechies for smooth signals, Coiflets for near-symmetric wavelets, Symlets for symmetric wavelets.

**Code:** `blocks/blocks.py` — lines 357-678

**Parameter:** `basis_dim: int = 32`

---

## 8. Configurable Activation Functions

**What:** 10 activation function options for the FC layers in all blocks.

**Paper difference:** The paper uses ReLU exclusively.

**Available activations:** ReLU, RReLU, PReLU, ELU, Softplus, Tanh, SELU, LeakyReLU, Sigmoid, GELU

**Code:** `constants.py` — `ACTIVATIONS`

**Parameter:** `activation: str = 'ReLU'`

---

## 9. NormalizedDeviationLoss

**What:** A scale-normalized L1 loss function.

**Paper difference:** The paper uses SMAPE, MASE, and MAPE only.

**Mathematical formulation:**
```
ND = sum(|y_hat - y|) / sum(|y|)
```

**Motivation:** Provides a simple, interpretable measure of forecast accuracy that is scale-invariant. Unlike SMAPE, it does not suffer from asymmetry issues when actual values are near zero.

**Code:** `losses.py` — `NormalizedDeviationLoss`

---

## 10. Flexible Stack Composition

**What:** Arbitrary block types in any order via the `stack_types` list parameter.

**Paper difference:** The paper defines only two architectures:
- **Generic** (G): 30 stacks of Generic blocks
- **Interpretable** (I): Trend stacks followed by Seasonality stacks

This implementation allows arbitrary combinations such as:
```python
stack_types = ["Trend", "DB3Wavelet", "Generic", "AutoEncoder"]
stack_types = ["BottleneckGeneric", "Seasonality", "HaarWavelet"]
stack_types = ["GenericAE", "TrendAE", "SeasonalityAE"]
```

**Motivation:** Different time series may benefit from different decomposition strategies. Allowing arbitrary stack compositions enables experimentation with novel architectures without code changes.

**Code:** `models.py` — `__init__` and `create_stack`

**Parameter:** `stack_types: list` (required)
