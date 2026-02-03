"""
Test for Mamba2 SSD (Structured State-Space Duality) chunk scan combined kernel.

Compares the CUTLASS CuTe DSL Blackwell implementation against the production
Triton implementation.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add CUTLASS mamba2_ssd path
CUTLASS_MAMBA2_SSD_PATH = (
    Path(__file__).resolve().parents[2]
    / "3rdparty"
    / "cutlass"
    / "examples"
    / "python"
    / "CuTeDSL"
    / "blackwell"
    / "mamba2_ssd"
)
sys.path.insert(0, str(CUTLASS_MAMBA2_SSD_PATH))

# Import Triton reference
from .triton_reference.ssd_chunk_state import _chunk_cumsum_fwd
from .triton_reference.ssd_combined import _mamba_chunk_scan_combined_fwd


def is_blackwell_available():
    """Check if Blackwell GPU (SM100) is available."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major >= 10  # SM100 = Blackwell


# Skip all tests if not on Blackwell
pytestmark = pytest.mark.skipif(
    not is_blackwell_available(),
    reason="Blackwell GPU (SM100+) required for CuTe DSL Mamba2 SSD kernel",
)


def import_cutlass_modules():
    """Import CUTLASS modules (only when needed, as they require SM100)."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    from cutlass.cute.runtime import from_dlpack
    from mamba2_ssd import SSDKernel

    return {
        "cuda": cuda,
        "cutlass": cutlass,
        "cute": cute,
        "cutlass_torch": cutlass_torch,
        "from_dlpack": from_dlpack,
        "SSDKernel": SSDKernel,
    }


class CutlassSSDWrapper:
    """
    Wrapper around CUTLASS CuTe DSL SSD kernel to match Triton API.

    The CUTLASS kernel expects:
    - Preprocessed cumsum_delta (step 1 already done)
    - Specific tensor layouts different from Triton
    - Output tensors preallocated

    This wrapper:
    1. Computes cumsum using Triton's step 1 (_chunk_cumsum_fwd)
    2. Converts tensors to CUTLASS layout
    3. Calls CUTLASS kernel
    4. Converts output back to Triton layout
    """

    def __init__(
        self,
        chunk_size: int,
        headdim: int,
        dstate: int,
        has_d: bool = True,
        d_has_hdim: bool = False,
        io_dtype=None,
        cumsum_dtype=None,
        acc_dtype=None,
    ):
        """
        Initialize the wrapper.

        Args:
            chunk_size: L - size of each chunk
            headdim: D - head dimension
            dstate: N - state dimension
            has_d: Whether to fuse D scaling (Y += X*D)
            d_has_hdim: If True, D is (headdim, nheads), else (1, nheads)
            io_dtype: Input/output dtype (default: cutlass.BFloat16)
            cumsum_dtype: Cumsum intermediate dtype (default: cutlass.Float32)
            acc_dtype: Accumulator dtype (default: cutlass.Float32)
        """
        self.modules = import_cutlass_modules()
        cutlass = self.modules["cutlass"]

        self.chunk_size = chunk_size
        self.headdim = headdim
        self.dstate = dstate
        self.has_d = has_d
        self.d_has_hdim = d_has_hdim

        self.io_dtype = io_dtype or cutlass.BFloat16
        self.cumsum_dtype = cumsum_dtype or cutlass.Float32
        self.acc_dtype = acc_dtype or cutlass.Float32

        # Create the kernel
        SSDKernel = self.modules["SSDKernel"]
        self.kernel = SSDKernel(
            self.io_dtype,
            self.cumsum_dtype,
            self.acc_dtype,
            chunk_size,
            headdim,
            dstate,
            has_d,
            d_has_hdim,
        )

        self._compiled_kernel = None

    def _create_cutlass_tensor(self, shape, permute_order, dtype, dynamic_modes):
        """
        Create a tensor using the exact logic from mamba2_ssd.py to ensure compatibility.

        Args:
            shape: Base shape of the tensor (before permutation)
            permute_order: Order to permute dimensions
            dtype: CUTLASS dtype
            dynamic_modes: List of modes to mark as dynamic

        Returns:
            (cute_tensor, torch_tensor): The CuTe tensor wrapper and the underlying PyTorch tensor on GPU
        """
        cutlass_torch = self.modules["cutlass_torch"]
        from_dlpack = self.modules["from_dlpack"]

        # Create a dummy CPU tensor with the base layout to establish the permutation pattern
        # mimicking create_and_permute_tensor from mamba2_ssd.py
        base_tensor = torch.empty(*shape, dtype=torch.float32)
        permuted_tensor = base_tensor.permute(permute_order)

        # Move to GPU with target dtype - this creates the specific layout CUTLASS expects
        torch_dtype = cutlass_torch.dtype(dtype)
        dst_tensor = permuted_tensor.to(torch_dtype).cuda()

        # Create CuTe tensor
        cute_tensor = from_dlpack(dst_tensor, assumed_align=16)
        for mode in dynamic_modes:
            cute_tensor = cute_tensor.mark_compact_shape_dynamic(
                mode=mode, stride_order=dst_tensor.dim_order()
            )

        return cute_tensor, dst_tensor

    def __call__(
        self,
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D=None,
        z=None,
        dt_bias=None,
        initial_states=None,
        seq_idx=None,
        dt_softplus=False,
        dt_limit=(0.0, float("inf")),
    ):
        """
        Run the SSD kernel with Triton-compatible API.

        Args:
            x: (batch, seqlen, nheads, headdim)
            dt: (batch, seqlen, nheads)
            A: (nheads,)
            B: (batch, seqlen, ngroups, dstate)
            C: (batch, seqlen, ngroups, dstate)
            chunk_size: Size of chunks
            D: Optional (nheads, headdim) or (nheads,)
            z: Optional gating tensor (not supported yet)
            dt_bias: Optional (nheads,)
            initial_states: Optional (batch, nheads, headdim, dstate)
            seq_idx: Optional sequence indices (not supported yet)
            dt_softplus: Whether to apply softplus to dt
            dt_limit: Limits for dt values

        Returns:
            out: (batch, seqlen, nheads, headdim)
            final_states: (batch, nheads, headdim, dstate)
        """
        cutlass = self.modules["cutlass"]
        cute = self.modules["cute"]

        # Validate inputs
        batch, seqlen, nheads, headdim = x.shape
        _, _, ngroups, dstate = B.shape
        nchunks = seqlen // chunk_size

        assert seqlen % chunk_size == 0, (
            f"seqlen ({seqlen}) must be divisible by chunk_size ({chunk_size})"
        )
        assert headdim == self.headdim, f"headdim mismatch: {headdim} vs {self.headdim}"
        assert dstate == self.dstate, f"dstate mismatch: {dstate} vs {self.dstate}"
        assert chunk_size == self.chunk_size, (
            f"chunk_size mismatch: {chunk_size} vs {self.chunk_size}"
        )

        if z is not None:
            raise NotImplementedError("z (gating) not yet supported in CUTLASS wrapper")
        if seq_idx is not None:
            raise NotImplementedError("seq_idx not yet supported in CUTLASS wrapper")
        if initial_states is not None:
            raise NotImplementedError(
                "initial_states not yet supported in CUTLASS wrapper"
            )

        # Step 1: Compute cumsum using Triton kernel
        # dA_cumsum: (batch, nheads, nchunks, chunk_size)
        # dt_processed: (batch, nheads, nchunks, chunk_size) - after softplus/bias
        dA_cumsum, dt_processed = _chunk_cumsum_fwd(
            dt,
            A,
            chunk_size,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            dt_limit=dt_limit,
        )

        # Convert tensors to CUTLASS layout using the same pattern as mamba2_ssd.py
        # Key: create contiguous tensor in base shape, then permute to get correct strides
        # CUTLASS expects specific permuted layouts for each tensor

        # x: Triton (batch, seqlen, nheads, headdim) -> CUTLASS (headdim, chunk_size, nchunks, nheads, batch)
        x_reshaped = x.reshape(batch, nchunks, chunk_size, nheads, headdim)
        x_tensor, x_dst = self._create_cutlass_tensor(
            [batch, nheads, headdim, nchunks, chunk_size],
            [2, 4, 3, 1, 0],
            self.io_dtype,
            [2, 3, 4],
        )
        x_dst.copy_(x_reshaped.permute(4, 2, 1, 3, 0).to(x_dst.dtype))

        # delta (dt_processed): (batch, nheads, nchunks, chunk_size) -> (chunk_size, nchunks, nheads, batch)
        delta_tensor, delta_dst = self._create_cutlass_tensor(
            [batch, nheads, nchunks, chunk_size], [3, 2, 1, 0], self.io_dtype, [1, 2, 3]
        )
        delta_dst.copy_(dt_processed.permute(3, 2, 1, 0).to(delta_dst.dtype))

        # cumsum_delta (dA_cumsum): same layout as delta
        cumsum_delta_tensor, cumsum_delta_dst = self._create_cutlass_tensor(
            [batch, nheads, nchunks, chunk_size],
            [3, 2, 1, 0],
            self.cumsum_dtype,
            [1, 2, 3],
        )
        cumsum_delta_dst.copy_(dA_cumsum.permute(3, 2, 1, 0).to(cumsum_delta_dst.dtype))

        # B: Triton (batch, seqlen, ngroups, dstate) -> CUTLASS (chunk_size, dstate, nchunks, ngroups, batch)
        B_reshaped = B.reshape(batch, nchunks, chunk_size, ngroups, dstate)
        b_tensor, b_dst = self._create_cutlass_tensor(
            [batch, ngroups, dstate, nchunks, chunk_size],
            [4, 2, 3, 1, 0],
            self.io_dtype,
            [2, 3, 4],
        )
        b_dst.copy_(B_reshaped.permute(2, 4, 1, 3, 0).to(b_dst.dtype))

        # C: same layout as B
        C_reshaped = C.reshape(batch, nchunks, chunk_size, ngroups, dstate)
        c_tensor, c_dst = self._create_cutlass_tensor(
            [batch, ngroups, dstate, nchunks, chunk_size],
            [4, 2, 3, 1, 0],
            self.io_dtype,
            [2, 3, 4],
        )
        c_dst.copy_(C_reshaped.permute(2, 4, 1, 3, 0).to(c_dst.dtype))

        # D: (nheads,) -> CUTLASS (1, nheads) or (headdim, nheads)
        if self.has_d and D is not None:
            if self.d_has_hdim:
                # D is (nheads, headdim) -> (headdim, nheads)
                if D.dim() == 1:
                    D = D.unsqueeze(1).expand(-1, headdim)
                d_tensor, d_dst = self._create_cutlass_tensor(
                    [nheads, headdim], [1, 0], self.io_dtype, [1]
                )
                d_dst.copy_(D.t().to(d_dst.dtype))
            else:
                # D is (nheads,) -> (1, nheads)
                if D.dim() == 2:
                    D = D[:, 0]
                d_tensor, d_dst = self._create_cutlass_tensor(
                    [nheads, 1], [1, 0], self.io_dtype, [1]
                )
                d_dst.copy_(D.unsqueeze(0).to(d_dst.dtype))
        else:
            d_tensor = None

        # Output tensors
        # y: (chunk_size, headdim, nchunks, nheads, batch)
        y_tensor, y_cutlass = self._create_cutlass_tensor(
            [batch, nheads, headdim, nchunks, chunk_size],
            [4, 2, 3, 1, 0],
            self.io_dtype,
            [2, 3, 4],
        )

        # fstate: (headdim, dstate, nheads, batch)
        fstate_tensor, fstate_cutlass = self._create_cutlass_tensor(
            [batch, nheads, headdim, dstate], [2, 3, 1, 0], self.io_dtype, [2, 3]
        )

        # Get max active clusters
        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(1)

        stream = cutlass.cuda.default_stream()

        # Compile kernel if not already done
        if self._compiled_kernel is None:
            self._compiled_kernel = cute.compile(
                self.kernel,
                x_tensor,
                cumsum_delta_tensor,
                delta_tensor,
                b_tensor,
                c_tensor,
                y_tensor,
                fstate_tensor,
                d_tensor,
                max_active_clusters,
                stream,
            )

        # Run kernel
        self._compiled_kernel(
            x_tensor,
            cumsum_delta_tensor,
            delta_tensor,
            b_tensor,
            c_tensor,
            y_tensor,
            fstate_tensor,
            d_tensor,
            stream,
        )

        # Convert outputs back to Triton layout
        # y_cutlass is (L, D, C, EH, B)
        # We need to map it back to (batch, seqlen, nheads, headdim)
        # Permute (L, D, C, EH, B) -> (B, C, L, EH, D)
        y_permuted = y_cutlass.permute(4, 2, 0, 3, 1)
        y_out = y_permuted.reshape(batch, seqlen, nheads, headdim)

        # fstate_cutlass is (D, N, EH, B)
        # We need (batch, nheads, headdim, dstate)
        # Permute (D, N, EH, B) -> (B, EH, D, N)
        fstate_out = fstate_cutlass.permute(3, 2, 0, 1).contiguous()

        return y_out, fstate_out


class TestChunkScanCombined:
    """Test class for chunk scan combined kernel."""

    # Test configuration - slightly relaxed tolerance for bf16 precision
    ATOL = 5e-2
    RTOL = 5e-2
    INPUT_DTYPE = torch.bfloat16

    @pytest.fixture(params=[1, 2])
    def batch(self, request):
        return request.param

    @pytest.fixture(params=[8])  # nheads must be divisible by ngroups
    def nheads(self, request):
        return request.param

    @pytest.fixture(params=[64])  # Must match kernel's D
    def headdim(self, request):
        return request.param

    @pytest.fixture(
        params=[128]
    )  # Must match kernel's N (CUTLASS kernel is hardcoded for N=128)
    def dstate(self, request):
        return request.param

    @pytest.fixture(params=[128])  # Must match kernel's L
    def chunk_size(self, request):
        return request.param

    @pytest.fixture(params=[1, 4])  # Number of chunks
    def nchunks(self, request):
        return request.param

    @pytest.fixture(params=[8])  # ngroups divides nheads
    def ngroups(self, request):
        return request.param

    @pytest.fixture
    def inputs(self, batch, nheads, headdim, dstate, chunk_size, nchunks, ngroups):
        """Create test inputs."""
        torch.manual_seed(42)

        seqlen = chunk_size * nchunks

        # x: (batch, seqlen, nheads, headdim)
        x = torch.randn(
            batch, seqlen, nheads, headdim, dtype=self.INPUT_DTYPE, device="cuda"
        )

        # dt: (batch, seqlen, nheads)
        dt = torch.randn(batch, seqlen, nheads, dtype=torch.float32, device="cuda")

        # A: (nheads,) - should be negative for stability
        A = -torch.rand(nheads, dtype=torch.float32, device="cuda") - 1.0

        # B: (batch, seqlen, ngroups, dstate)
        B = torch.randn(
            batch, seqlen, ngroups, dstate, dtype=self.INPUT_DTYPE, device="cuda"
        )

        # C: (batch, seqlen, ngroups, dstate)
        C = torch.randn(
            batch, seqlen, ngroups, dstate, dtype=self.INPUT_DTYPE, device="cuda"
        )

        # D: (nheads, headdim) or (nheads,)
        D = torch.randn(nheads, dtype=self.INPUT_DTYPE, device="cuda")

        # dt_bias: (nheads,)
        dt_bias = torch.rand(nheads, dtype=torch.float32, device="cuda") - 4.0

        return {
            "x": x,
            "dt": dt,
            "A": A,
            "B": B,
            "C": C,
            "D": D,
            "dt_bias": dt_bias,
            "chunk_size": chunk_size,
            "seqlen": seqlen,
            "nheads": nheads,
            "headdim": headdim,
            "dstate": dstate,
            "ngroups": ngroups,
        }

    @pytest.fixture
    def reference_output(self, inputs):
        """Compute reference output using Triton implementation."""
        out, dt_out, dA_cumsum, states, final_states = _mamba_chunk_scan_combined_fwd(
            inputs["x"],
            inputs["dt"],
            inputs["A"],
            inputs["B"],
            inputs["C"],
            inputs["chunk_size"],
            D=inputs["D"],
            z=None,
            dt_bias=inputs["dt_bias"],
            initial_states=None,
            seq_idx=None,
            dt_softplus=True,
        )
        return out, final_states

    def _print_mismatch_details(self, ref, test, name, atol, rtol):
        """Print detailed mismatch analysis."""
        ref_np = ref.detach().cpu().float().numpy()
        test_np = test.detach().cpu().float().numpy()

        mismatch_mask = ~np.isclose(ref_np, test_np, atol=atol, rtol=rtol)
        num_mismatches = np.sum(mismatch_mask)
        total_elements = ref_np.size

        print(f"\nDetailed {name} mismatch analysis:")
        print(
            f"Number of mismatched elements: {num_mismatches} / {total_elements} "
            f"({100 * num_mismatches / total_elements:.2f}%)"
        )

        if num_mismatches > 0:
            mismatch_indices = np.argwhere(mismatch_mask)
            print(f"First few {name} mismatch locations (up to 10):")
            for idx in mismatch_indices[:10]:
                idx_tuple = tuple(int(i) for i in idx)
                ref_val = ref_np[idx_tuple]
                test_val = test_np[idx_tuple]
                diff = abs(ref_val - test_val)
                rel_diff = diff / (abs(ref_val) + 1e-8)
                print(
                    f"  Index {idx_tuple}: ref={ref_val:.6f}, test={test_val:.6f}, "
                    f"diff={diff:.6e}, rel_diff={rel_diff:.6e}"
                )

    def test_output_correctness(self, inputs, reference_output):
        """Test that CUTLASS kernel output matches Triton reference."""
        out_ref, final_states_ref = reference_output

        # Create CUTLASS wrapper
        wrapper = CutlassSSDWrapper(
            chunk_size=inputs["chunk_size"],
            headdim=inputs["headdim"],
            dstate=inputs["dstate"],
            has_d=True,
            d_has_hdim=False,  # D is (nheads,) not (nheads, headdim)
        )

        # Run CUTLASS kernel
        out_test, final_states_test = wrapper(
            inputs["x"],
            inputs["dt"],
            inputs["A"],
            inputs["B"],
            inputs["C"],
            inputs["chunk_size"],
            D=inputs["D"],
            dt_bias=inputs["dt_bias"],
            dt_softplus=True,
        )

        # Compare outputs - cast to same dtype for comparison
        out_ref_cmp = out_ref.to(out_test.dtype)
        out_match = torch.allclose(
            out_ref_cmp, out_test, atol=self.ATOL, rtol=self.RTOL
        )

        if out_match:
            print(
                f"✓ Outputs match within tolerance (atol={self.ATOL}, rtol={self.RTOL})"
            )
        else:
            print("✗ Outputs do NOT match within tolerance")
            self._print_mismatch_details(
                out_ref_cmp, out_test, "output", self.ATOL, self.RTOL
            )

        # Compare final states - cast to same dtype for comparison
        final_states_ref_cmp = final_states_ref.to(final_states_test.dtype)
        states_match = torch.allclose(
            final_states_ref_cmp, final_states_test, atol=self.ATOL, rtol=self.RTOL
        )

        if states_match:
            print(
                f"✓ Final states match within tolerance (atol={self.ATOL}, rtol={self.RTOL})"
            )
        else:
            print("✗ Final states do NOT match within tolerance")
            self._print_mismatch_details(
                final_states_ref_cmp,
                final_states_test,
                "final_states",
                self.ATOL,
                self.RTOL,
            )

        assert out_match, "Output mismatch between CUTLASS and Triton"
        assert states_match, "Final states mismatch between CUTLASS and Triton"


class TestChunkScanCombinedNoD(TestChunkScanCombined):
    """Test chunk scan without D scaling."""

    @pytest.fixture(params=[1])
    def batch(self, request):
        return request.param

    @pytest.fixture(params=[1])
    def nchunks(self, request):
        return request.param

    @pytest.fixture
    def inputs(self, batch, nheads, headdim, dstate, chunk_size, nchunks, ngroups):
        """Create test inputs without D."""
        torch.manual_seed(42)

        seqlen = chunk_size * nchunks

        x = torch.randn(
            batch, seqlen, nheads, headdim, dtype=self.INPUT_DTYPE, device="cuda"
        )
        dt = torch.randn(batch, seqlen, nheads, dtype=torch.float32, device="cuda")
        A = -torch.rand(nheads, dtype=torch.float32, device="cuda") - 1.0
        B = torch.randn(
            batch, seqlen, ngroups, dstate, dtype=self.INPUT_DTYPE, device="cuda"
        )
        C = torch.randn(
            batch, seqlen, ngroups, dstate, dtype=self.INPUT_DTYPE, device="cuda"
        )
        dt_bias = torch.rand(nheads, dtype=torch.float32, device="cuda") - 4.0

        return {
            "x": x,
            "dt": dt,
            "A": A,
            "B": B,
            "C": C,
            "D": None,
            "dt_bias": dt_bias,
            "chunk_size": chunk_size,
            "seqlen": seqlen,
            "nheads": nheads,
            "headdim": headdim,
            "dstate": dstate,
            "ngroups": ngroups,
        }

    @pytest.fixture
    def reference_output(self, inputs):
        """Compute reference output using Triton implementation without D."""
        out, dt_out, dA_cumsum, states, final_states = _mamba_chunk_scan_combined_fwd(
            inputs["x"],
            inputs["dt"],
            inputs["A"],
            inputs["B"],
            inputs["C"],
            inputs["chunk_size"],
            D=None,
            z=None,
            dt_bias=inputs["dt_bias"],
            initial_states=None,
            seq_idx=None,
            dt_softplus=True,
        )
        return out, final_states

    def test_output_correctness(self, inputs, reference_output):
        """Test without D scaling."""
        out_ref, final_states_ref = reference_output

        wrapper = CutlassSSDWrapper(
            chunk_size=inputs["chunk_size"],
            headdim=inputs["headdim"],
            dstate=inputs["dstate"],
            has_d=False,
            d_has_hdim=False,
        )

        out_test, final_states_test = wrapper(
            inputs["x"],
            inputs["dt"],
            inputs["A"],
            inputs["B"],
            inputs["C"],
            inputs["chunk_size"],
            D=None,
            dt_bias=inputs["dt_bias"],
            dt_softplus=True,
        )

        # Cast to same dtype for comparison
        out_ref_cmp = out_ref.to(out_test.dtype)
        final_states_ref_cmp = final_states_ref.to(final_states_test.dtype)

        out_match = torch.allclose(
            out_ref_cmp, out_test, atol=self.ATOL, rtol=self.RTOL
        )
        states_match = torch.allclose(
            final_states_ref_cmp, final_states_test, atol=self.ATOL, rtol=self.RTOL
        )

        if out_match:
            print("✓ [NoD] Outputs match within tolerance")
        else:
            print("✗ [NoD] Outputs do NOT match")
            self._print_mismatch_details(
                out_ref_cmp, out_test, "output", self.ATOL, self.RTOL
            )

        if states_match:
            print("✓ [NoD] Final states match within tolerance")
        else:
            print("✗ [NoD] Final states do NOT match")
            self._print_mismatch_details(
                final_states_ref_cmp,
                final_states_test,
                "final_states",
                self.ATOL,
                self.RTOL,
            )

        assert out_match
        assert states_match
