# Unit tests

Unit tests isolate individual pipeline stages and verify specific numerical or robustness properties. They are fast and require no external data: inputs are generated in-process. Tests are built as part of the normal CMake build and registered with CTest.

> These unit tests were referenced from the AMD HIP proof-of-concept implementation: [AMD-HPC/CvxCompress @ amd/hip-poc](https://github.com/AMD-HPC/CvxCompress/tree/amd/hip-poc).

## Test executables

| CTest / executable | Source | Focus |
|---|---|---|
| `stage_isolation_test` | [stage_isolation/](stage_isolation) | Individual stages (wavelet, quantization, RLE) |
| `numerical_correctness_test` | [numerical_correctness/](numerical_correctness) | Error bounds, determinism, RMS accuracy, ratio monotonicity |
| `robustness_test` | [robustness/](robustness) | Degenerate inputs, odd dimensions, buffer/reuse safety |

## How to run

```bash
# Build first (from repo root)
cmake --build build -j$(nproc)

# Run one specific test case (ctest -R matches the GoogleTest "Suite.Case" name
# that gtest_discover_tests registers, NOT the executable name)
ctest --test-dir build -R WaveletStage -V

# Run all tests in one executable by matching its suite names
ctest --test-dir build -R "^(WaveletStage|QuantizationStage|RleStage)\."          # stage_isolation_test
ctest --test-dir build -R "^(ErrorBound|CompressionRatio|DeviceRms|CompressedLength|Determinism)\."  # numerical_correctness_test
ctest --test-dir build -R "^Robustness"                                           # robustness_test
```

To run an entire executable directly (bypassing CTest), invoke the built binary, optionally with a GoogleTest filter:

```bash
./build/stage_isolation_test                                 # all tests in the executable
./build/numerical_correctness_test
./build/robustness_test
./build/stage_isolation_test --gtest_filter='WaveletStage.*' # a subset
```

## What each test verifies

### Stage isolation: [stage_isolation/](stage_isolation)

**Wavelet** ([wavelet_stage_test.cu](stage_isolation/wavelet_stage_test.cu))
- `WaveletStage.ForwardInverseIsIdentity`: forward then inverse transform reproduces the input to float precision.
- `WaveletStage.ConstantBlockPreserved`: a constant (DC) field survives a roundtrip exactly.
- `WaveletStage.ForwardIsLinear`: the forward transform is linear: `F(a·x) == a·F(x)`.

**Quantization** ([quantization_stage_test.cu](stage_isolation/quantization_stage_test.cu))
- `QuantizationStage.MeanRmsMatchesHost`: the device mean/RMS reduction matches a double-precision host reference.
- `QuantizationStage.GlobalNormalizeScalesByMultiplier`: global normalization applies a fixed `1/(rms·scale)` multiplier with zero mean.
- `QuantizationStage.GlobalNormalizeFlagsNaN`: a NaN-containing block is flagged (negative multiplier) so the caller stores it raw.
- `QuantizationStage.LocalNormalizeZeroCentersConstant`: local normalization subtracts the block mean, so a constant block quantizes to all zeros.

**Run-length encoding** ([rle_stage_test.cu](stage_isolation/rle_stage_test.cu))
- `RleStage.RoundtripAllZeros`: an all-zero block collapses to a single run and decodes back to all zeros.
- `RleStage.RoundtripDenseSmallInts`: dense small integers encode as direct single-byte escape codes.
- `RleStage.RoundtripSparseWithZeroRuns`: interleaved zero runs and direct values roundtrip losslessly.
- `RleStage.RoundtripWideValues`: values outside the direct range use multi-byte value-escape payloads.

### Numerical correctness: [numerical_correctness/](numerical_correctness)

These tests validate the quantitative guarantees of the full compress/decompress path: the size/accuracy trade-off, determinism, and the accuracy of the device RMS reduction that drives global-RMS scaling. Shared helpers (`GpuRoundtrip`, `HostRmsAboutZero`) live in [numerical_test_utils.cuh](numerical_correctness/numerical_test_utils.cuh).

**Error bound** ([error_bound_test.cu](numerical_correctness/error_bound_test.cu))
- `ErrorBound.MaxErrorWithinScaledRmsBound`: global-RMS quantization uses a step `~ rms·scale`, so the reconstructed max absolute error must stay within `(C·scale + floor)·rms`, where `floor ≈ 5e-3` of `rms` accounts for the wavelet transform's float-precision reconstruction limit. Checked across scales `{1e-1, 5e-2, 1e-2, 1e-3}`.
- `ErrorBound.ErrorDecreasesAsScaleTightens`: sweeping the same scales, tightening the scale must not increase reconstruction error (allowing ~5% slack for quantization noise).

**Compression ratio monotonicity** ([cr_vs_scale_monotonicity_test.cu](numerical_correctness/cr_vs_scale_monotonicity_test.cu))
- `CompressionRatio.DecreasesAsScaleTightens`: a smaller scale retains more wavelet coefficients, so the container grows and the compression ratio is monotonically non-increasing as the scale tightens (scales `{1e-1, 5e-2, 1e-2, 5e-3, 1e-3}`, each ratio `> 1`, with a 2% slack to absorb ties).

**Compressed length accuracy** ([compressed_length_accuracy_test.cu](numerical_correctness/compressed_length_accuracy_test.cu))
- `CompressedLength.ReportedLengthIsAccurate`: the length returned by `Compress` must equal the length independently recovered from the container header, be strictly positive, stay within `GetMaxCompressedSize`, be reproducible across calls, be consistent with the reported compression ratio, and yield a roundtrip within the RMS-scaled error bound.

**Determinism** ([determinism_test.cu](numerical_correctness/determinism_test.cu))
- `Determinism.RepeatedCompressReconstructsIdentically`: compressing identical input twice yields an identical container length and a bit-for-bit identical decompressed volume. (Stream-byte identity is not asserted because per-block metadata carries a small amount of order-dependent content.)

**Device RMS accuracy** ([device_rms_accuracy_test.cu](numerical_correctness/device_rms_accuracy_test.cu))
- `DeviceRms.MatchesHostReference`: the Thrust reduction behind `CUDA_ComputeGlobalMeanRMS` agrees with a double-precision host reference to within float accumulation error (~2e-3 relative).
- `DeviceRms.ZeroAndConstantVolumes`: degenerate volumes stay finite (no NaN/Inf): a zero field reports `rms == 0` and `mean == 0`, and a constant field `c` reports `rms == |c|` and `mean == c`.

### Robustness: [robustness/](robustness)

**Degenerate input** ([degenerate_input_test.cu](robustness/degenerate_input_test.cu))
- `Robustness.ZeroVolume`: a zero volume roundtrips exactly and compresses heavily.
- `Robustness.ConstantVolume`: a non-zero constant volume is essentially lossless.
- `Robustness.NearZeroInput`: tiny magnitudes do not trigger division blow-ups; output stays finite.
- `Robustness.LargeMagnitudeRoundtrips`: moderately large magnitudes still roundtrip accurately.
- `Robustness.ExtremeMagnitudeDegradesGracefully`: extreme magnitudes lose signal but the library stays finite, does not crash, and respects the buffer.
- `Robustness.NaNInfInputDoesNotCrash`: NaN/Inf inputs neither crash nor overflow the output buffer.

**Dimensions** ([dimensions_test.cu](robustness/dimensions_test.cu))
- `RobustnessDims.NonMultipleOf16`: non-block-aligned dimensions are padded internally and still roundtrip within bounds.
- `RobustnessDims.SingleVoxel`: a single voxel padded to a full block.
- `RobustnessDims.SubBlockVolume`: a volume smaller than 16 in every dimension.
- `RobustnessDims.ThinAlongX` / `RobustnessDims.ThinAlongZ`: extreme aspect ratios along each axis.

**Buffer and reuse** ([buffer_and_reuse_test.cu](robustness/buffer_and_reuse_test.cu))
- `RobustnessBuffer.IncompressibleInputFitsBuffer`: high-entropy input barely compresses (may expand) but always fits `GetMaxCompressedSize` and roundtrips within the error bound.
- `RobustnessReuse.RepeatedCyclesAreStable`: repeated compress/decompress cycles on one instance produce identical length and bit-identical reconstruction every cycle.
