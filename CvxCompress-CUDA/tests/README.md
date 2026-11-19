# Tests

This directory contains the dataset-free correctness and robustness tests for CvxCompress-CUDA. Tests use GoogleTest and are registered with CTest during the CMake build.

## Run

From the repository root:

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

The tests generate their input volumes in-process. No external datasets or CPU reference library are required.

For the complete catalog and focused test commands, see [unit/README.md](unit/README.md).

## Test areas

- `unit/stage_isolation/`: wavelet, quantization, and run-length encoding stages.
- `unit/numerical_correctness/`: error bounds, compression ratio, RMS accuracy, compressed length, and determinism.
- `unit/robustness/`: degenerate inputs, dimensions, buffer safety, and reuse.