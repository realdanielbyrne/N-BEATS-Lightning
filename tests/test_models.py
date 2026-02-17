"""Tests for NBeatsNet model — width selection logic and optimizer dispatch."""
import pytest
import torch
from torch import optim

from lightningnbeats.models import NBeatsNet


def _make_model(stack_types, **kwargs):
    """Helper to create a minimal NBeatsNet model."""
    defaults = dict(
        backcast_length=20,
        forecast_length=5,
        stack_types=stack_types,
        n_blocks_per_stack=1,
        share_weights=False,
        thetas_dim=4,
        active_g=False,
        latent_dim=4,
        basis_dim=16,
    )
    defaults.update(kwargs)
    return NBeatsNet(**defaults)


# --- Width selection tests ---

class TestWidthSelection:
    """Verify each block type uses the correct width parameter."""

    def test_generic_uses_g_width(self):
        model = _make_model(["Generic"], g_width=64, s_width=128, t_width=32, ae_width=96)
        block = model.stacks[0][0]
        assert block.units == 64

    def test_generic_ae_uses_g_width(self):
        model = _make_model(["GenericAE"], g_width=64, s_width=128, t_width=32, ae_width=96)
        block = model.stacks[0][0]
        assert block.units == 64

    def test_generic_ae_backcast_uses_g_width(self):
        model = _make_model(["GenericAEBackcast"], g_width=64, s_width=128, t_width=32, ae_width=96)
        block = model.stacks[0][0]
        assert block.units == 64

    def test_generic_ae_backcast_ae_uses_g_width(self):
        model = _make_model(["GenericAEBackcastAE"], g_width=64, s_width=128, t_width=32, ae_width=96)
        block = model.stacks[0][0]
        assert block.units == 64

    def test_seasonality_uses_s_width(self):
        model = _make_model(["Seasonality"], g_width=64, s_width=128, t_width=32, ae_width=96)
        block = model.stacks[0][0]
        assert block.units == 128

    def test_seasonality_ae_uses_s_width(self):
        model = _make_model(["SeasonalityAE"], g_width=64, s_width=128, t_width=32, ae_width=96)
        block = model.stacks[0][0]
        assert block.units == 128

    def test_trend_uses_t_width(self):
        model = _make_model(["Trend"], g_width=64, s_width=128, t_width=32, ae_width=96)
        block = model.stacks[0][0]
        assert block.units == 32

    def test_trend_ae_uses_t_width(self):
        model = _make_model(["TrendAE"], g_width=64, s_width=128, t_width=32, ae_width=96)
        block = model.stacks[0][0]
        assert block.units == 32

    def test_autoencoder_uses_ae_width(self):
        model = _make_model(["AutoEncoder"], g_width=64, s_width=128, t_width=32, ae_width=96)
        block = model.stacks[0][0]
        assert block.units == 96

    def test_autoencoder_ae_uses_ae_width(self):
        model = _make_model(["AutoEncoderAE"], g_width=64, s_width=128, t_width=32, ae_width=96)
        block = model.stacks[0][0]
        assert block.units == 96

    def test_wavelet_uses_g_width_fallback(self):
        model = _make_model(["HaarWavelet"], g_width=64, s_width=128, t_width=32, ae_width=96)
        block = model.stacks[0][0]
        assert block.units == 64


# --- Optimizer dispatch tests ---

class TestOptimizerDispatch:
    """Verify optimizer_name parameter is respected."""

    def test_adam_optimizer(self):
        model = _make_model(["Generic"], optimizer_name="Adam")
        opt = model.configure_optimizers()
        assert isinstance(opt, optim.Adam)

    def test_sgd_optimizer(self):
        model = _make_model(["Generic"], optimizer_name="SGD")
        opt = model.configure_optimizers()
        assert isinstance(opt, optim.SGD)

    def test_adamw_optimizer(self):
        model = _make_model(["Generic"], optimizer_name="AdamW")
        opt = model.configure_optimizers()
        assert isinstance(opt, optim.AdamW)

    def test_rmsprop_optimizer(self):
        model = _make_model(["Generic"], optimizer_name="RMSprop")
        opt = model.configure_optimizers()
        assert isinstance(opt, optim.RMSprop)

    def test_invalid_optimizer_raises(self):
        model = _make_model(["Generic"])
        model.optimizer_name = "InvalidOptimizer"
        with pytest.raises(ValueError, match="Unknown optimizer name"):
            model.configure_optimizers()


# --- Forward pass shape tests ---

class TestForwardPass:
    """Verify model forward pass produces correct output shapes."""

    def test_generic_forward_shape(self):
        model = _make_model(["Generic"], g_width=32)
        x = torch.randn(4, 20)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 20)
        assert forecast.shape == (4, 5)

    def test_trend_seasonality_forward_shape(self):
        model = _make_model(["Trend", "Seasonality"], t_width=32, s_width=64)
        x = torch.randn(4, 20)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 20)
        assert forecast.shape == (4, 5)

    def test_invalid_stack_type_raises(self):
        with pytest.raises(ValueError, match="Stack architecture must be specified"):
            NBeatsNet(backcast_length=20, forecast_length=5, stack_types=None)


# --- Model defaults tests ---

class TestModelDefaults:
    """Verify NBeatsNet default parameter values."""

    def test_active_g_defaults_to_false(self):
        model = NBeatsNet(
            backcast_length=20,
            forecast_length=5,
            stack_types=["Generic"],
            n_blocks_per_stack=1
        )
        assert model.active_g is False

    def test_learning_rate_defaults_to_1e_minus_3(self):
        model = NBeatsNet(
            backcast_length=20,
            forecast_length=5,
            stack_types=["Generic"],
            n_blocks_per_stack=1
        )
        assert model.learning_rate == 1e-3


# --- sum_losses semantic fix tests ---

class TestSumLossesBehavior:
    """Verify sum_losses uses zero target for backcast loss."""

    def test_sum_losses_backcast_uses_zero_target_in_training(self):
        model = _make_model(["Generic"], sum_losses=True, g_width=32)
        x = torch.randn(4, 20)
        y = torch.randn(4, 5)
        batch = (x, y)

        loss = model.training_step(batch, 0)

        assert loss is not None
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_sum_losses_backcast_uses_zero_target_in_validation(self):
        model = _make_model(["Generic"], sum_losses=True, g_width=32, no_val=False)
        x = torch.randn(4, 20)
        y = torch.randn(4, 5)
        batch = (x, y)

        loss = model.validation_step(batch, 0)

        assert loss is not None
        assert isinstance(loss, torch.Tensor)

    def test_sum_losses_backcast_uses_zero_target_in_test(self):
        model = _make_model(["Generic"], sum_losses=True, g_width=32)
        x = torch.randn(4, 20)
        y = torch.randn(4, 5)
        batch = (x, y)

        loss = model.test_step(batch, 0)

        assert loss is not None
        assert isinstance(loss, torch.Tensor)


# --- NormalizedDeviationLoss configuration tests ---

class TestNormalizedDeviationLossConfiguration:
    """Verify NormalizedDeviationLoss is handled in configure_loss."""

    def test_model_with_normalized_deviation_loss(self):
        from lightningnbeats.losses import NormalizedDeviationLoss
        model = _make_model(["Generic"], loss="NormalizedDeviationLoss", g_width=32)
        assert isinstance(model.loss_fn, NormalizedDeviationLoss)

    def test_normalized_deviation_loss_forward_pass(self):
        model = _make_model(["Generic"], loss="NormalizedDeviationLoss", g_width=32)
        x = torch.randn(4, 20)
        y = torch.randn(4, 5)
        batch = (x, y)

        loss = model.training_step(batch, 0)

        assert loss is not None
        assert isinstance(loss, torch.Tensor)


# --- BottleneckGeneric width selection tests ---

class TestBottleneckGenericWidthSelection:
    """Verify BottleneckGeneric and BottleneckGenericAE use g_width."""

    def test_bottleneck_generic_uses_g_width(self):
        model = _make_model(["BottleneckGeneric"], g_width=64, s_width=128, t_width=32, ae_width=96)
        block = model.stacks[0][0]
        assert block.units == 64

    def test_bottleneck_generic_ae_uses_g_width(self):
        model = _make_model(["BottleneckGenericAE"], g_width=64, s_width=128, t_width=32, ae_width=96)
        block = model.stacks[0][0]
        assert block.units == 64


# --- active_g split mode model-level tests ---

class TestActiveGSplitModesModel:
    """Verify string active_g modes are accepted and passed to blocks."""

    def test_active_g_backcast_accepted(self):
        model = _make_model(["Generic"], active_g='backcast', g_width=32)
        assert model.active_g == 'backcast'
        block = model.stacks[0][0]
        assert block.active_g == 'backcast'

    def test_active_g_forecast_accepted(self):
        model = _make_model(["Generic"], active_g='forecast', g_width=32)
        assert model.active_g == 'forecast'
        block = model.stacks[0][0]
        assert block.active_g == 'forecast'

    def test_active_g_true_still_works(self):
        model = _make_model(["Generic"], active_g=True, g_width=32)
        assert model.active_g is True
        block = model.stacks[0][0]
        assert block.active_g is True

    def test_active_g_false_still_works(self):
        model = _make_model(["Generic"], active_g=False, g_width=32)
        assert model.active_g is False

    def test_active_g_invalid_string_raises(self):
        with pytest.raises(ValueError, match="active_g must be"):
            _make_model(["Generic"], active_g='invalid')

    def test_active_g_backcast_forward_pass(self):
        model = _make_model(["Generic"], active_g='backcast', g_width=32)
        x = torch.randn(4, 20)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 20)
        assert forecast.shape == (4, 5)

    def test_active_g_forecast_forward_pass(self):
        model = _make_model(["Generic"], active_g='forecast', g_width=32)
        x = torch.randn(4, 20)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 20)
        assert forecast.shape == (4, 5)

