
ACTIVATIONS = [
    "ReLU",
    "RReLU",
    "PReLU",
    "ELU",
    "Softplus",
    "Tanh",
    "SELU",
    "LeakyReLU",
    "Sigmoid",
    "GELU"
]

LOSSES = [
    "MAPELoss",
    "SMAPELoss",
    "MASELoss",
    "NormalizedDeviationLoss",
    "MSELoss",
    "L1Loss",
    "SmoothL1Loss",
    "BCEWithLogitsLoss",
    "BCELoss",
    "CrossEntropyLoss",
    "NLLLoss",
    "PoissonNLLLoss",
    "KLDivLoss",
    "MarginRankingLoss",
    "HingeEmbeddingLoss",
    "MultiLabelMarginLoss",
    "CosineEmbeddingLoss",
    "MultiMarginLoss",
    "TripletMarginLoss",
    "TripletMarginWithDistanceLoss"
]

OPTIMIZERS = [
  "Adam",
  "SGD",
  "RMSprop",
  "Adagrad",
  "Adadelta",
  "AdamW"
]

BLOCKS = [
  "Generic",
  "BottleneckGeneric",
  "GenericAE",
  "BottleneckGenericAE",
  "GenericAEBackcast",
  "GenericAEBackcastAE",
  "Trend",
  "TrendAE",
  "Seasonality",
  "SeasonalityAE",
  "AutoEncoder",
  "AutoEncoderAE",
  # V3 Wavelet blocks (orthonormal DWT basis)
  "HaarWaveletV3",
  "DB2WaveletV3",
  "DB3WaveletV3",
  "DB4WaveletV3",
  "DB10WaveletV3",
  "DB20WaveletV3",
  "Coif1WaveletV3",
  "Coif2WaveletV3",
  "Coif3WaveletV3",
  "Coif10WaveletV3",
  "Symlet2WaveletV3",
  "Symlet3WaveletV3",
  "Symlet10WaveletV3",
  "Symlet20WaveletV3",
]
