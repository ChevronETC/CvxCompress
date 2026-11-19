
| Path | Role |
|---|---|
| [include/CvxCompress_CUDA.hxx](include/CvxCompress_CUDA.hxx) | Public API: `Compress`, `Decompress`, `CompressAsync`, `GetMaxCompressedSize`, `SetStream` |
| [src/CvxCompress_CUDA.cu](src/CvxCompress_CUDA.cu) | Top-level orchestration of the compress/decompress pipeline |
| [src/CvxCompress_CUDA_kernels.cu](src/CvxCompress_CUDA_kernels.cu) | CUDA kernels: buffer copy, subvolume compress/decompress, container packing |
| [src/Wavelet_Transform_16.cu](src/Wavelet_Transform_16.cu) | Forward/inverse 16³ wavelet transform |
| [src/Run_Length_Encoding.cu](src/Run_Length_Encoding.cu) | Run-length encoding/decoding of quantized coefficients |
| [src/CvxCompress_CUDA_helpers.cu](src/CvxCompress_CUDA_helpers.cu) | Device helpers (RMS reduction, sizing) |
| [include/Common/](include/Common) | Shared utilities (`gpuAssert.h`, `profiler.h`) |
| [include/IO/](include/IO), [io/](io) | Raw volume read/write helpers |
| [tests/](tests) | Dataset-free unit tests (see [tests/README.md](tests/README.md)) |


## Dependencies

- NVIDIA CUDA Toolkit 12.x (tested with 12.3)
- CMake 3.18+
- GCC with C++17 support

### Configure and build

```bash
cd CvxCompress-CUDA
cmake -S . -B build
cmake --build build -j$(nproc)
```

`CMAKE_CUDA_ARCHITECTURES` is not required to configure the project, so the default build can be kept simple. If you want to target a specific GPU architecture explicitly, add it during configure, for example:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=80
```

Use the architecture value that matches your GPU, for example `80` for Ampere.

### Verify the build

Run the unit tests:

```bash
ctest --test-dir build --output-on-failure
```


### Available tests

The suite contains stage-level correctness, numerical, and robustness unit tests. For the catalog and focused test commands, see:

- [tests/README.md](tests/README.md): test suite overview
- [tests/unit/README.md](tests/unit/README.md): unit tests

## API Reference

The public API lives in [include/CvxCompress_CUDA.hxx](include/CvxCompress_CUDA.hxx). All `vol` and `compressed` pointers are **device** pointers. Create one `CvxCompress_CUDA` object and reuse it.

### Synchronous compress / decompress

```cpp
#include "CvxCompress_CUDA.hxx"

CvxCompress_CUDA cvx;

// Allocate a worst-case output buffer on the device.
long long max_bytes = CvxCompress_CUDA::GetMaxCompressedSize(nx, ny, nz);
char* d_compressed; cudaMalloc(&d_compressed, max_bytes);

// Compress (blocks until done). Returns the compression ratio.
long  compressed_length = 0;
float scale = 0.01f;   // higher scale -> more compression, less accuracy
bool use_local_RMS = false;   // false = global RMS (default), true = per-block RMS
float ratio = cvx.Compress(scale, d_vol, nx, ny, nz,
                            16, 16, 16,          // block dims (bx, by, bz)
                            use_local_RMS,
                            d_compressed, compressed_length);

// Decompress back into a device volume.
float* d_out; cudaMalloc(&d_out, (size_t)nx * ny * nz * sizeof(float));
cvx.Decompress(d_out, nx, ny, nz, d_compressed, compressed_length);
```

### Asynchronous compress (on a CUDA stream)

`CompressAsync` enqueues all GPU work on the plan's stream and returns immediately. The compressed size is not valid until you synchronize the stream and read it back with `GetCompressedSize`.

```cpp
cudaStream_t stream;  cudaStreamCreate(&stream);
cvx.SetStream(stream); // route work onto your stream

cvx.CompressAsync(scale, d_vol, nx, ny, nz,
                  16, 16, 16, use_local_RMS, d_compressed);
// ... overlap other work here ...

cudaStreamSynchronize(stream); // required before reading the size
long compressed_length = 0;
float ratio = cvx.GetCompressedSize(d_compressed, compressed_length);
```

### Method summary

| Method | Description |
|---|---|
| `Compress(scale, vol, nx, ny, nz, bx, by, bz, use_local_RMS, compressed, compressed_length)` | Blocking compress; returns the compression ratio and writes `compressed_length`. |
| `Compress(scale, vol, nx, ny, nz, use_local_RMS, compressed, compressed_length, voi_pad)` | Blocking compress with a volume-of-interest padding scheme. |
| `CompressAsync(scale, vol, nx, ny, nz, bx, by, bz, use_local_RMS, compressed)` | Non-blocking compress on the plan's stream (set via `SetStream`). |
| `GetCompressedSize(compressed, compressed_length)` | Blocking; parses the container header to recover the byte length. Pair with `CompressAsync`. |
| `Decompress(vol, nx, ny, nz, compressed, compressed_length)` | Reconstruct the volume from a compressed container. |
| `GetMaxCompressedSize(nx, ny, nz)` | Static; worst-case buffer size (bytes) for allocation before compressing. |
| `SetStream(stream)` | Route all subsequent GPU work onto the given CUDA stream. |

> Block dimensions are fixed at 16³; the `bx, by, bz` parameters are kept for API symmetry. `use_local_RMS = false` selects global RMS (the default, higher ratio); see [docs/2_details.md](docs/2_details.md#normalization) for the tradeoff.