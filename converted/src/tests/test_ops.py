# NNModelling — DSL for designing neural networks via visual node editor
# Copyright (C) 2026  Luca Sforza
#
# Licensed under the GNU General Public License v3 or later.
# Commercial licenses are available — contact Luca Sforza.
# See the LICENSE file for details.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
import torch
import pytest

from ops.addition import Addition
from ops.concat import Concat
from ops.einsum import Einsum
from ops.mat_mul import MatMul
from ops.scaled_dot_product import ScaledDotProduct
from ops.masked_scaled_dot_product import MaskedScaledDotProduct
from ops.subflow import Subflow
from ops.repeat import Repeat
from ops.horizontal_repeat import HorizontalRepeat
from ops.positional_encoding import PositionalEncoding
from ops.sequence_pool import SequencePool
from ops.unflatten import Unflatten


# -- Addition -----------------------------------------------------------------------

class TestAddition:
    def test_sum_two(self):
        a, b = torch.ones(3), torch.ones(3)
        out = Addition()([a, b])
        assert out.equal(torch.tensor([2., 2., 2.]))

    def test_sum_three(self):
        tensors = [torch.ones(3) for _ in range(3)]
        out = Addition()(tensors)
        assert out.equal(torch.tensor([3., 3., 3.]))

    def test_single_tensor_passthrough(self):
        t = torch.randn(3, 4)
        out = Addition()([t])
        assert out.equal(t)

    def test_empty_list(self):
        out = Addition()([])
        assert out == []

    def test_shape_mismatch_error(self):
        with pytest.raises(RuntimeError, match="shape mismatch"):
            Addition()([torch.ones(3), torch.ones(4)])


# -- Concat -------------------------------------------------------------------------

class TestConcat:
    def test_concat_last_dim(self):
        a, b = torch.randn(2, 3), torch.randn(2, 4)
        out = Concat(dim=-1)([a, b])
        assert out.shape == (2, 7)

    def test_concat_first_dim(self):
        a, b = torch.randn(2, 3), torch.randn(4, 3)
        out = Concat(dim=0)([a, b])
        assert out.shape == (6, 3)

    def test_single_tensor(self):
        t = torch.randn(2, 3)
        out = Concat()([t])
        assert out.shape == (2, 3)

    def test_default_dim(self):
        a, b = torch.randn(2, 3), torch.randn(2, 5)
        out = Concat()([a, b])
        assert out.shape == (2, 8)  # default dim=-1


# -- Unflatten ----------------------------------------------------------------------

class TestUnflatten:
    def test_accepts_hydra_style_sequence(self):
        tensor = torch.arange(16).reshape(2, 8)

        output = Unflatten(dim=1, unflattened_size=[2, 2, 2])(tensor)

        assert output.shape == (2, 2, 2, 2)
        assert output.flatten(start_dim=1).equal(tensor)


# -- Einsum -------------------------------------------------------------------------

class TestEinsum:
    def test_transpose(self):
        t = torch.randn(2, 3)
        out = Einsum(expr="ij->ji")([t])
        assert out.shape == (3, 2)
        assert out.equal(t.T)

    def test_batch_matmul(self):
        a, b = torch.randn(4, 2, 3), torch.randn(4, 3, 5)
        out = Einsum(expr="bij,bjk->bik")([a, b])
        assert out.shape == (4, 2, 5)

    def test_empty_expr_error(self):
        with pytest.raises(ValueError, match="non-empty"):
            Einsum(expr="")([torch.randn(2, 3)])


# -- MatMul -------------------------------------------------------------------------

class TestMatMul:
    def test_2d_matmul(self):
        a, b = torch.randn(3, 4), torch.randn(4, 5)
        out = MatMul()([a, b])
        assert out.shape == (3, 5)
        assert out.equal(a @ b)

    def test_batched_matmul(self):
        a, b = torch.randn(2, 3, 4), torch.randn(2, 4, 5)
        out = MatMul()([a, b])
        assert out.shape == (2, 3, 5)

    def test_input_ordering(self):
        """Verify inputs[0] @ inputs[1] — ordering matters for non-commutative."""
        a = torch.randn(3, 4)
        b = torch.randn(4, 5)
        out = MatMul()([a, b])
        expected = a @ b
        assert out.equal(expected)
        # reversed would fail or give different shape
        with pytest.raises(RuntimeError):
            MatMul()([b, a])


# -- ScaledDotProduct ---------------------------------------------------------------

class TestScaledDotProduct:
    def test_output_shape(self):
        q, k = torch.randn(2, 8, 64), torch.randn(2, 8, 64)
        out = ScaledDotProduct(d_model=64)([q, k])
        assert out.shape == (2, 8, 8)

    def test_scale_factor(self):
        d_model = 16
        q = torch.ones(1, 4, d_model)
        k = torch.ones(1, 4, d_model)
        out = ScaledDotProduct(d_model=d_model)([q, k])
        expected = torch.full((1, 4, 4), d_model * (d_model ** -0.5))
        assert torch.allclose(out, expected)

# -- MaskedScaledDotProduct ---------------------------------------------------------

class TestMaskedScaledDotProduct:
    def test_output_shape(self):
        q, k = torch.randn(2, 8, 64), torch.randn(2, 8, 64)
        out = MaskedScaledDotProduct(d_model=64)([q, k])
        assert out.shape == (2, 8, 8)

    def test_causal_mask(self):
        d_model = 8
        q = torch.randn(1, 4, d_model)
        k = torch.randn(1, 4, d_model)
        out = MaskedScaledDotProduct(d_model=d_model)([q, k])
        # upper triangle (j > i) should be -inf
        assert torch.all(out[0, 0, 1:] == float("-inf"))  # row 0, cols 1..n
        assert torch.all(out[0, 1, 2:] == float("-inf"))  # row 1, cols 2..n
        # diagonal and lower triangle should be finite
        assert torch.isfinite(out[0, 0, 0])
        assert torch.isfinite(out[0, 1, 1])
        assert torch.isfinite(out[0, 2, 0])  # lower


# -- Subflow ------------------------------------------------------------------------

class TestSubflow:
    def _identity_internal(self):
        return {
            "identity": {
                "type": "module",
                "children": [],
                "stereotype": "Identity",
                "_target_": "torch.nn.Identity",
            }
        }

    def _linear_internal(self, in_f=4, out_f=8):
        return {
            "linear": {
                "type": "module",
                "children": [],
                "stereotype": "Linear",
                "_target_": "torch.nn.Linear",
                "in_features": in_f,
                "out_features": out_f,
                "bias": False,
            }
        }

    def test_identity(self):
        sf = Subflow(entry_node="identity", internal_nodes=self._identity_internal())
        x = torch.randn(2, 10)
        out = sf(x)
        assert out.shape == (2, 10)
        assert out.equal(x)

    def test_linear(self):
        sf = Subflow(entry_node="linear", internal_nodes=self._linear_internal(4, 8))
        x = torch.randn(2, 4)
        out = sf(x)
        assert out.shape == (2, 8)

    def test_sequence_two_modules(self):
        internal = {
            "lin1": {
                "type": "module", "children": ["lin2"],
                "stereotype": "Linear", "_target_": "torch.nn.Linear",
                "in_features": 4, "out_features": 8, "bias": False,
            },
            "lin2": {
                "type": "module", "children": [],
                "stereotype": "Linear", "_target_": "torch.nn.Linear",
                "in_features": 8, "out_features": 16, "bias": False,
            },
        }
        sf = Subflow(entry_node="lin1", internal_nodes=internal)
        x = torch.randn(2, 4)
        out = sf(x)
        assert out.shape == (2, 16)

    def test_fork_addition_join(self):
        internal = {
            "fork": {
                "type": "module", "children": ["lin_a", "lin_b"],
                "stereotype": "Fork",
            },
            "lin_a": {
                "type": "module", "children": ["add"],
                "stereotype": "Linear", "_target_": "torch.nn.Linear",
                "in_features": 4, "out_features": 8, "bias": False,
            },
            "lin_b": {
                "type": "module", "children": ["add"],
                "stereotype": "Linear", "_target_": "torch.nn.Linear",
                "in_features": 4, "out_features": 8, "bias": False,
            },
            "add": {
                "type": "join", "children": [],
                "stereotype": "Addition", "_target_": "ops.Addition",
                "inputs": ["lin_a", "lin_b"],
            },
        }
        sf = Subflow(entry_node="fork", internal_nodes=internal)
        x = torch.randn(2, 4)
        out = sf(x)
        assert out.shape == (2, 8)

    def test_join_input_ordering(self):
        """Verify inputs list controls join argument order."""
        internal = {
            "fork": {
                "type": "module", "children": ["a", "b"],
                "stereotype": "Fork",
            },
            "a": {
                "type": "module", "children": ["cat"],
                "stereotype": "Identity", "_target_": "torch.nn.Identity",
            },
            "b": {
                "type": "module", "children": ["cat"],
                "stereotype": "Linear", "_target_": "torch.nn.Linear",
                "in_features": 4, "out_features": 8, "bias": False,
            },
            "cat": {
                "type": "join", "children": [],
                "stereotype": "Concat", "_target_": "ops.Concat",
                "dim": -1,
                "inputs": ["a", "b"],
            },
        }
        # With inputs=["a","b"], output is [Identity(x), Linear(x)] = concat([4, 8]) = 12
        sf = Subflow(entry_node="fork", internal_nodes=internal)
        x = torch.randn(2, 4)
        out = sf(x)
        assert out.shape == (2, 12)

    def test_empty_internal_nodes(self):
        sf = Subflow(entry_node="none", internal_nodes={})
        x = torch.randn(2, 4)
        out = sf(x)
        assert out.equal(x)

    def test_dict_input_no_dictconfig(self):
        """Subflow works with plain dicts (not just DictConfig)."""
        sf = Subflow(entry_node="identity", internal_nodes=self._identity_internal())
        x = torch.randn(2, 3)
        out = sf(x)
        assert out.equal(x)


# -- Repeat -------------------------------------------------------------------------

class TestRepeat:
    def _identity_subflow(self):
        return {
            "id": {
                "type": "module", "children": [],
                "stereotype": "Identity", "_target_": "torch.nn.Identity",
            }
        }

    def test_iterations_1(self):
        r = Repeat(entry_node="id", internal_nodes=self._identity_subflow(), iterations=1)
        x = torch.randn(2, 4)
        out = r(x)
        assert out.shape == (2, 4)

    def test_iterations_3(self):
        internal = {
            "lin": {
                "type": "module", "children": [],
                "stereotype": "Linear", "_target_": "torch.nn.Linear",
                "in_features": 4, "out_features": 4, "bias": False,
            }
        }
        r = Repeat(entry_node="lin", internal_nodes=internal, iterations=3)
        x = torch.randn(2, 4)
        out = r(x)
        assert out.shape == (2, 4)

    def test_value_propagation_identity(self):
        r = Repeat(entry_node="id", internal_nodes=self._identity_subflow(), iterations=5)
        x = torch.randn(2, 3)
        out = r(x)
        assert out.equal(x)


# -- HorizontalRepeat ---------------------------------------------------------------

class TestHorizontalRepeat:
    def _linear_internal(self, in_f=4, out_f=8):
        return {
            "lin": {
                "type": "module", "children": [],
                "stereotype": "Linear", "_target_": "torch.nn.Linear",
                "in_features": in_f, "out_features": out_f, "bias": False,
            }
        }

    def test_n_1_passthrough(self):
        hr = HorizontalRepeat(entry_node="lin", internal_nodes=self._linear_internal(4, 8), n=1)
        x = torch.randn(2, 4)
        out = hr(x)
        assert out.shape == (2, 8)

    def test_n_4_concat_output(self):
        hr = HorizontalRepeat(entry_node="lin", internal_nodes=self._linear_internal(4, 8), n=4)
        x = torch.randn(2, 4)
        out = hr(x)
        assert out.shape == (2, 4 * 8)  # n * d_head

    def test_n_0_error(self):
        with pytest.raises(ValueError, match=">= 1"):
            HorizontalRepeat(entry_node="id", internal_nodes={}, n=0)


# -- PositionalEncoding -------------------------------------------------------------

class TestPositionalEncoding:
    def test_pe_table_shape(self):
        pe = PositionalEncoding(d_model=64, max_len=100)
        assert pe.pe.shape == (1, 100, 64)

    def test_forward_shape(self):
        pe = PositionalEncoding(d_model=64)
        x = torch.randn(2, 50, 64)
        out = pe(x)
        assert out.shape == (2, 50, 64)

    def test_non_trainable_buffer(self):
        pe = PositionalEncoding(d_model=64, max_len=100)
        # pe is a buffer, not a parameter
        assert len(list(pe.parameters())) == 0
        # it IS in buffers
        buffer_names = dict(pe.named_buffers())
        assert "pe" in buffer_names


# -- SequencePool -------------------------------------------------------------------

class TestSequencePool:
    def test_pool_3d(self):
        x = torch.randn(2, 16, 64)
        out = SequencePool(dim=1)(x)
        assert out.shape == (2, 64)

    def test_2d_passthrough(self):
        x = torch.randn(2, 64)
        out = SequencePool()(x)
        assert out.shape == (2, 64)

    def test_custom_dim(self):
        x = torch.randn(2, 16, 64)
        out = SequencePool(dim=2)(x)
        assert out.shape == (2, 16)
