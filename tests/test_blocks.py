"""Tests for block implementations — activation, instantiation, forward pass, attribute names."""
import pytest
import torch
from torch import nn

from lightningnbeats.blocks import blocks as b
from lightningnbeats.constants import ACTIVATIONS


BACKCAST_LENGTH = 20
FORECAST_LENGTH = 5
UNITS = 32
THETAS_DIM = 4
BASIS_DIM = 16
LATENT_DIM = 4


# --- RootBlock activation validation ---

class TestRootBlockActivation:
    """Verify RootBlock validates activation parameter."""

    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError, match="is not in"):
            b.RootBlock(BACKCAST_LENGTH, UNITS, activation="InvalidActivation")

    def test_valid_activations(self):
        for act in ACTIVATIONS:
            block = b.RootBlock(BACKCAST_LENGTH, UNITS, activation=act)
            assert isinstance(block.activation, nn.Module)

    def test_pep8_not_in_syntax(self):
        """Ensure 'activation not in' works correctly (PEP 8 fix)."""
        with pytest.raises(ValueError):
            b.RootBlock(BACKCAST_LENGTH, UNITS, activation="FakeReLU")
        block = b.RootBlock(BACKCAST_LENGTH, UNITS, activation="ReLU")
        assert block is not None


# --- AutoEncoderAE instantiation (critical bug fix) ---

class TestAutoEncoderAE:
    """Verify AutoEncoderAE can be instantiated and run forward pass."""

    def test_instantiation_no_crash(self):
        block = b.AutoEncoderAE(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU", latent_dim=LATENT_DIM)
        assert block is not None

    def test_instantiation_shared_weights(self):
        block = b.AutoEncoderAE(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=True, activation="ReLU", latent_dim=LATENT_DIM)
        assert block.b_encoder is block.f_encoder

    def test_instantiation_different_activations(self):
        for act in ["ReLU", "LeakyReLU", "GELU", "ELU"]:
            block = b.AutoEncoderAE(
                units=UNITS, backcast_length=BACKCAST_LENGTH,
                forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
                share_weights=False, activation=act, latent_dim=LATENT_DIM)
            assert block is not None

    def test_forward_pass(self):
        block = b.AutoEncoderAE(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU", latent_dim=LATENT_DIM)
        x = torch.randn(4, BACKCAST_LENGTH)
        backcast, forecast = block(x)
        assert backcast.shape == (4, BACKCAST_LENGTH)
        assert forecast.shape == (4, FORECAST_LENGTH)

    def test_sequential_has_activation_modules(self):
        block = b.AutoEncoderAE(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU", latent_dim=LATENT_DIM)
        encoder_modules = list(block.b_encoder.modules())
        has_activation = any(isinstance(m, nn.ReLU) for m in encoder_modules)
        assert has_activation


# --- SeasonalityAE forward (dead code removal) ---

class TestSeasonalityAE:
    """Verify SeasonalityAE forward returns correct tensors."""

    def test_forward_returns_two_tensors(self):
        block = b.SeasonalityAE(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU", latent_dim=LATENT_DIM)
        x = torch.randn(4, BACKCAST_LENGTH)
        result = block(x)
        assert len(result) == 2
        backcast, forecast = result
        assert backcast.shape == (4, BACKCAST_LENGTH)
        assert forecast.shape == (4, FORECAST_LENGTH)


# --- Wavelet share_weights attribute name ---

class TestWaveletAttributes:
    """Verify Wavelet classes use correct attribute name (not typo)."""

    def test_wavelet_has_share_weights(self):
        block = b.Wavelet(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, basis_dim=BASIS_DIM,
            share_weights=False, activation="ReLU")
        assert hasattr(block, "share_weights")
        assert not hasattr(block, "sharre_weights")
        assert block.share_weights is False

    def test_alt_wavelet_has_share_weights(self):
        block = b.AltWavelet(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, basis_dim=BASIS_DIM,
            share_weights=True, activation="ReLU")
        assert hasattr(block, "share_weights")
        assert not hasattr(block, "sharre_weights")
        assert block.share_weights is True

    def test_haar_wavelet_forward(self):
        block = b.HaarWavelet(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, basis_dim=BASIS_DIM,
            share_weights=False, activation="ReLU")
        x = torch.randn(4, BACKCAST_LENGTH)
        backcast, forecast = block(x)
        assert backcast.shape == (4, BACKCAST_LENGTH)
        assert forecast.shape == (4, FORECAST_LENGTH)


# --- Generic vs BottleneckGeneric architecture tests ---

class TestGenericArchitecture:
    """Verify paper-faithful Generic block architecture."""

    def test_generic_has_theta_fc_layers(self):
        block = b.Generic(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU")
        assert hasattr(block, "theta_b_fc")
        assert hasattr(block, "theta_f_fc")

    def test_generic_does_not_have_bottleneck_layers(self):
        block = b.Generic(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU")
        assert not hasattr(block, "backcast_linear")
        assert not hasattr(block, "forecast_linear")
        assert not hasattr(block, "backcast_g")
        assert not hasattr(block, "forecast_g")

    def test_generic_forward_pass_shape(self):
        block = b.Generic(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU")
        x = torch.randn(4, BACKCAST_LENGTH)
        backcast, forecast = block(x)
        assert backcast.shape == (4, BACKCAST_LENGTH)
        assert forecast.shape == (4, FORECAST_LENGTH)

    def test_generic_theta_b_fc_output_size(self):
        block = b.Generic(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU")
        assert block.theta_b_fc.out_features == BACKCAST_LENGTH

    def test_generic_theta_f_fc_output_size(self):
        block = b.Generic(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU")
        assert block.theta_f_fc.out_features == FORECAST_LENGTH


class TestBottleneckGenericArchitecture:
    """Verify BottleneckGeneric block architecture with thetas_dim bottleneck."""

    def test_bottleneck_generic_has_bottleneck_layers(self):
        block = b.BottleneckGeneric(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU")
        assert hasattr(block, "backcast_linear")
        assert hasattr(block, "forecast_linear")
        assert hasattr(block, "backcast_g")
        assert hasattr(block, "forecast_g")

    def test_bottleneck_generic_does_not_have_theta_fc_layers(self):
        block = b.BottleneckGeneric(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU")
        assert not hasattr(block, "theta_b_fc")
        assert not hasattr(block, "theta_f_fc")

    def test_bottleneck_generic_forward_pass_shape(self):
        block = b.BottleneckGeneric(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU")
        x = torch.randn(4, BACKCAST_LENGTH)
        backcast, forecast = block(x)
        assert backcast.shape == (4, BACKCAST_LENGTH)
        assert forecast.shape == (4, FORECAST_LENGTH)

    def test_bottleneck_generic_backcast_linear_output_size(self):
        block = b.BottleneckGeneric(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU")
        assert block.backcast_linear.out_features == THETAS_DIM

    def test_bottleneck_generic_forecast_linear_output_size(self):
        block = b.BottleneckGeneric(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU")
        assert block.forecast_linear.out_features == THETAS_DIM

    def test_bottleneck_generic_backcast_g_output_size(self):
        block = b.BottleneckGeneric(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU")
        assert block.backcast_g.out_features == BACKCAST_LENGTH

    def test_bottleneck_generic_forecast_g_output_size(self):
        block = b.BottleneckGeneric(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU")
        assert block.forecast_g.out_features == FORECAST_LENGTH


class TestGenericAEArchitecture:
    """Verify paper-faithful GenericAE block architecture."""

    def test_generic_ae_has_theta_fc_layers(self):
        block = b.GenericAE(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU", latent_dim=LATENT_DIM)
        assert hasattr(block, "theta_b_fc")
        assert hasattr(block, "theta_f_fc")

    def test_generic_ae_does_not_have_bottleneck_layers(self):
        block = b.GenericAE(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU", latent_dim=LATENT_DIM)
        assert not hasattr(block, "backcast_linear")
        assert not hasattr(block, "forecast_linear")
        assert not hasattr(block, "backcast_g")
        assert not hasattr(block, "forecast_g")

    def test_generic_ae_forward_pass_shape(self):
        block = b.GenericAE(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU", latent_dim=LATENT_DIM)
        x = torch.randn(4, BACKCAST_LENGTH)
        backcast, forecast = block(x)
        assert backcast.shape == (4, BACKCAST_LENGTH)
        assert forecast.shape == (4, FORECAST_LENGTH)


class TestBottleneckGenericAEArchitecture:
    """Verify BottleneckGenericAE block architecture with thetas_dim bottleneck."""

    def test_bottleneck_generic_ae_has_bottleneck_layers(self):
        block = b.BottleneckGenericAE(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU", latent_dim=LATENT_DIM)
        assert hasattr(block, "backcast_linear")
        assert hasattr(block, "forecast_linear")
        assert hasattr(block, "backcast_g")
        assert hasattr(block, "forecast_g")

    def test_bottleneck_generic_ae_does_not_have_theta_fc_layers(self):
        block = b.BottleneckGenericAE(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU", latent_dim=LATENT_DIM)
        assert not hasattr(block, "theta_b_fc")
        assert not hasattr(block, "theta_f_fc")

    def test_bottleneck_generic_ae_forward_pass_shape(self):
        block = b.BottleneckGenericAE(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU", latent_dim=LATENT_DIM)
        x = torch.randn(4, BACKCAST_LENGTH)
        backcast, forecast = block(x)
        assert backcast.shape == (4, BACKCAST_LENGTH)
        assert forecast.shape == (4, FORECAST_LENGTH)


# --- Trend block defaults tests ---

class TestTrendDefaults:
    """Verify Trend block default parameter values."""

    def test_trend_activation_defaults_to_relu(self):
        block = b.Trend(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM)
        assert isinstance(block.activation, torch.nn.ReLU)

    def test_trend_share_weights_defaults_to_false(self):
        block = b.Trend(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM)
        assert block.share_weights is False
        assert block.backcast_linear is not block.forecast_linear


class TestTrendAEDefaults:
    """Verify TrendAE block default parameter values."""

    def test_trend_ae_activation_defaults_to_relu(self):
        block = b.TrendAE(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            latent_dim=LATENT_DIM)
        assert isinstance(block.activation, torch.nn.ReLU)

    def test_trend_ae_share_weights_defaults_to_false(self):
        block = b.TrendAE(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            latent_dim=LATENT_DIM)
        assert block.backcast_linear is not block.forecast_linear


# --- Seasonality block bias tests ---

class TestSeasonalityBias:
    """Verify Seasonality block has bias=False on linear layers."""

    def test_seasonality_backcast_linear_has_no_bias(self):
        block = b.Seasonality(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM)
        assert block.backcast_linear.bias is None

    def test_seasonality_forecast_linear_has_no_bias(self):
        block = b.Seasonality(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM)
        assert block.forecast_linear.bias is None


# --- Constants registry validation tests ---

class TestBlocksRegistry:
    """Verify every entry in BLOCKS has a corresponding class in blocks module."""

    def test_all_blocks_have_classes(self):
        from lightningnbeats.constants import BLOCKS
        for block_name in BLOCKS:
            assert hasattr(b, block_name), f"Block class {block_name} not found in blocks module"

    def test_all_block_classes_instantiable(self):
        from lightningnbeats.constants import BLOCKS
        for block_name in BLOCKS:
            block_class = getattr(b, block_name)
            assert callable(block_class), f"Block {block_name} is not callable"


class TestLossesRegistry:
    """Verify every entry in LOSSES has a corresponding class."""

    def test_all_losses_exist(self):
        from lightningnbeats.constants import LOSSES
        from lightningnbeats import losses
        import torch.nn as nn

        for loss_name in LOSSES:
            if hasattr(losses, loss_name):
                assert callable(getattr(losses, loss_name))
            elif hasattr(nn, loss_name):
                assert callable(getattr(nn, loss_name))
            else:
                raise AssertionError(f"Loss {loss_name} not found in losses module or torch.nn")


# --- All blocks output shape tests ---

class TestAllBlocksOutputShapes:
    """Parametrized test for all registered blocks to verify output shapes."""

    @pytest.mark.parametrize("block_name", [
        "Generic", "BottleneckGeneric", "GenericAE", "BottleneckGenericAE",
        "GenericAEBackcast", "GenericAEBackcastAE",
        "Trend", "TrendAE", "Seasonality", "SeasonalityAE",
        "AutoEncoder", "AutoEncoderAE",
        "HaarWavelet", "HaarAltWavelet",
        "DB2Wavelet", "DB2AltWavelet",
        "DB3Wavelet", "DB3AltWavelet",
        "DB4Wavelet", "DB4AltWavelet",
        "DB10Wavelet", "DB10AltWavelet", "DB20AltWavelet",
        "Coif1Wavelet", "Coif1AltWavelet",
        "Coif2Wavelet", "Coif2AltWavelet",
        "Coif3Wavelet", "Coif3AltWavelet",
        "Coif10Wavelet", "Coif10AltWavelet",
        "Symlet2Wavelet", "Symlet2AltWavelet",
        "Symlet3Wavelet", "Symlet10Wavelet", "Symlet20Wavelet"
    ])
    def test_block_output_shape(self, block_name):
        block_class = getattr(b, block_name)

        kwargs = {
            "units": UNITS,
            "backcast_length": BACKCAST_LENGTH,
            "forecast_length": FORECAST_LENGTH,
            "activation": "ReLU"
        }

        ae_root_blocks = ["SeasonalityAE", "GenericAEBackcastAE", "AutoEncoderAE",
                          "GenericAE", "BottleneckGenericAE", "TrendAE"]
        if block_name in ae_root_blocks:
            kwargs["latent_dim"] = LATENT_DIM

        if "Wavelet" in block_name:
            kwargs["basis_dim"] = BASIS_DIM
        else:
            kwargs["thetas_dim"] = THETAS_DIM

        if block_name in ["AutoEncoder", "AutoEncoderAE", "GenericAEBackcast", "GenericAEBackcastAE"]:
            kwargs["share_weights"] = False
        elif block_name in ["Trend", "TrendAE"]:
            kwargs["share_weights"] = False

        block = block_class(**kwargs)

        x = torch.randn(4, BACKCAST_LENGTH)
        backcast, forecast = block(x)

        assert backcast.shape == (4, BACKCAST_LENGTH), f"{block_name} backcast shape incorrect"
        assert forecast.shape == (4, FORECAST_LENGTH), f"{block_name} forecast shape incorrect"

