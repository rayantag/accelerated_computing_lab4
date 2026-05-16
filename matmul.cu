#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

void cuda_check(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(1);
    }
}

// hi

#define CUDA_CHECK(x) \
    do { \
        cuda_check((x), __FILE__, __LINE__); \
    } while (0)

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation (Too slow to actually run!)
//
// void matmul_cpu_naive(
//     int32_t size_i,
//     int32_t size_j,
//     int32_t size_k,
//     float const *a,
//     float const *b,
//     float *c) {
//     for (int32_t i = 0; i < size_i; ++i) {
//         for (int32_t j = 0; j < size_j; ++j) {
//             float sum = 0.0;
//             for (int32_t k = 0; k < size_k; ++k) {
//                 sum += a[i * size_k + k] * b[k * size_j + j];
//             }
//             c[i * size_j + j] = sum;
//         }
//     }
// }

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// GPU Implementation (With Reuse in L1/Shmem)

namespace matmul_l1 {

const int TILE = 32;

__global__ void matmul_l1(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
    /* TODO: your GPU code here */

    // Reminders:
    // Blocks run simultaneously across different SMs
    // Threads run simultaneously within a block (hierarchy: block --> warp --> thread)
    // C/CUDA doesn't have native 2D arrays for dynamically sized matrices.
    // Matrix accesses must be 1D

    // Declare shared memory tiles for A and B. They are logically private to each block
    // because we use the L1 scratchpad for them (the L1 cache part is per-SM, not per-block).
    // I'm told that we know only two blocks would be using this at once, so 32x32x4x2 = 16KB.
    __shared__ float a_shared[TILE][TILE];
    __shared__ float b_shared[TILE][TILE];

    // Global (i,j) this thread is responsible for.
    int global_col = blockIdx.x * blockDim.x + threadIdx.x;
    int global_row = blockIdx.y * blockDim.y + threadIdx.y;

    // Local thread within the block.
    int local_col = threadIdx.x;
    int local_row = threadIdx.y;

    // Accumulator for this thread's output elem.
    float sum = 0.0f;

    // Loop over K tiles.
    for (int k = 0; k < size_k; k += TILE) {

        // Cooperatively load a TILE-wide strip of A and B into shared memory (DRAM --> L1 via L2).
        // Each thread loads one element; together all threads fill the TILExTILE tile.
        // At one timestep, [a,b]_shared are block-specific, so therefore use local row/col.
        // But [a,b]_shared will end up spanning the whole k dim!

        // a_shared: fixed row, varies by column (k-dim).
        // get global row offset: global_row * size_k
        // get global col: k + local_col
        a_shared[local_row][local_col] = a[global_row * size_k + (k + local_col)];

        // b_shared: varied row (k_dim), fixed column.
        // get global row offset: (k + local_row) * size_j
        // get global col offset: global_col
        b_shared[local_row][local_col] = b[(k + local_row) * size_j + global_col];

        // Wait for all threads to finish loading.
        __syncthreads();

        // Accumulate dot product over the k-tile dimension (L1 --> register).
        for (int ki = 0; ki < TILE; ++ki) {
            sum += a_shared[local_row][ki] * b_shared[ki][local_col];
        }

        // Wait before overwriting shared memory in the next iteration.
        __syncthreads();
    }

    // Write result to global memory (only time we use global i and j).
    // Given row i, column j: C[i, j] = i * size_j + j (this is row-major).
    c[global_row * size_j + global_col] = sum;
}

void launch_matmul_l1(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {

    // Blocks: how many threads we are using (2D, threads must cover full dims).
    // This is what ThreadIdx.x and ThreadIdx.y are local to!! They go from (0-31).
    dim3 block(32, 32);

    // Grid: how many blocks are we using (2D, blocks must cover full dims).
    dim3 grid(size_i / TILE, size_j / TILE);

    // Launch matmul_l1 kernel.
    matmul_l1<<<grid, block>>>(size_i, size_j, size_k, a, b, c);
}

}; // namespace matmul_l1

////////////////////////////////////////////////////////////////////////////////
// GPU Implementation (With Reuse in L1/Shmem and Registers)

namespace matmul_l1_reg {

// These need to be compile-time constatns, otherwise you'll get something like:
// note: the value of variable "matmul_l1_reg::microtile_dim" cannot be used as a constant

// This is what gets loaded into L1: 128x128 block (4 bytes/elem).
// block_dim tiles the output (i and j dimensions), how much of C each block owns.
constexpr auto block_dim = 128;

// k_tile tiles the reduction (k-dimension) -- determines how much of the dot prod is computed per iteration.
constexpr auto k_tile = 64;

// Each thread owns an 8x8 microtile.
constexpr auto microtile_dim = 8;

// Threads per block.
constexpr auto threads_per_block_dim = block_dim / microtile_dim;

__global__ void matmul_l1_reg(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
    /* TODO: your GPU code here */
    // 64 KB memory per warp scheduler. 256 KB per SM (4 warp schedulers per SM)

    // Want to load 128x128 tiles at a time.
    // Want to load 128x64 and 64x128 tiles at a time.
    __shared__ float a_shared[block_dim][k_tile];
    __shared__ float b_shared[k_tile][block_dim];

    // Local thread within the block.
    int local_col = threadIdx.x;
    int local_row = threadIdx.y;

    // Global indices for the thread.
    int global_col = blockIdx.x * blockDim.x + local_col;
    int global_row = blockIdx.y * blockDim.y + local_row;

    float c_sums[8][8];

    // Advance shared tile along k-dim, increment by block size.
    for (int k = 0; k < size_k; k += k_tile) {
        
        // Load 128xk_tile of A and k_tile of B into shared mem cooperatively.
        // Here, each thread is responsible for 64 elements.
        for (int row = 0; row < microtile_dim; row += 1) {
            for (int col = 0; col < microtile_dim; col += 1) {

                // A: row is fixed (global row * size_k to actually get to the first elem), col (k-dim) varies.
                a_shared[local_row + row][local_col + col] = a[(global_row + row) * size_k + k + local_col + col];

                // B: col is fixed, row varies.
                // Get to specific row within block: k + local_row + row
                // Get to that actual element byte: (k + local_row + row) * size_j
                // Col: global_col + col
                b_shared[local_row + row][local_col + col] = b[(k + local_row + row) * size_j + global_col + col];
            }
        }
        __syncthreads();
        float a_reg[8], b_reg[8];
        for (int ki = 0; ki < k_tile; ki++) {
            for (int row = 0; row < microtile_dim; row++) {
                a_reg[row] = a_shared[local_row+row][ki];
            }
            for (int col = 0; col < microtile_dim; col++) {
                b_reg[col] = b_shared[ki][local_col + col];
            }
            for (int row = 0; row < microtile_dim; row++) {
                for (int col = 0; col < microtile_dim; col++) {
                    c_sums[row][col] += (a_reg[row] * b_reg[col]);
                }
            }
        }
        __syncthreads();
    }
    // Write register accumulators to global C.
    for (int row = 0; row < microtile_dim; row++) {
        for (int col = 0; col < microtile_dim; col++) {
            c[(global_row + row) * size_j + (global_col + col)] = c_sums[row][col];
        }
    }
}

void launch_matmul_l1_reg(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
    /* TODO: your CPU code here */
    
    // This notation specifies the thread layout within the block.
    // Note: regardless of the layout, warps are always 32 threads.
    dim3 block(threads_per_block_dim, threads_per_block_dim);

    // This specifies the block layout within the grid.
    dim3 grid(size_i / block_dim, size_j / block_dim);

    // Launch!
    matmul_l1_reg<<<grid, block>>>(size_i, size_j, size_k, a, b, c);
}

}; // namespace matmul_l1_reg

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

std::vector<float> read_data(std::string const &path, int32_t size) {
    std::ifstream file(path, std::ios::binary);
    std::vector<float> data(size);
    file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));
    if (file.fail()) {
        std::cerr << "Failed to read " << path << std::endl;
        std::abort();
    }
    return data;
}

template <typename F>
double benchmark_ms(double target_time_ms, int32_t num_iters_inner, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        for (int32_t i = 0; i < num_iters_inner; ++i) {
            f();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms / num_iters_inner);
    }
    return best_time_ms;
}

struct BenchmarkResult {
    char const *name;
    double elapsed_ms;
};

struct BenchmarkConfig {
    int32_t size_i;
    int32_t size_j;
    int32_t size_k;
    bool save_result;
};

template <typename Impl>
void run_tests_for_size(
    std::string const &test_data_dir,
    std::vector<BenchmarkResult> &saved_results,
    std::vector<BenchmarkConfig> const &configs) {
    for (auto config : configs) {
        auto size_i = config.size_i;
        auto size_j = config.size_j;
        auto size_k = config.size_k;

        auto path_prefix = test_data_dir + "/test_" + std::to_string(size_i) + "x" +
            std::to_string(size_j) + "x" + std::to_string(size_k);
        auto a = read_data(path_prefix + "_a.bin", size_i * size_k);
        auto b = read_data(path_prefix + "_b.bin", size_k * size_j);
        auto c = read_data(path_prefix + "_c.bin", size_i * size_j);

        float *a_gpu;
        float *b_gpu;
        float *c_gpu;
        CUDA_CHECK(cudaMalloc(&a_gpu, size_i * size_k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&b_gpu, size_k * size_j * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c_gpu, size_i * size_j * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(
            a_gpu,
            a.data(),
            size_i * size_k * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            b_gpu,
            b.data(),
            size_k * size_j * sizeof(float),
            cudaMemcpyHostToDevice));

        Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu);

        std::vector<float> c_out_host(size_i * size_j);
        CUDA_CHECK(cudaMemcpy(
            c_out_host.data(),
            c_gpu,
            size_i * size_j * sizeof(float),
            cudaMemcpyDeviceToHost));

        double mse = 0.0;
        double ref_mean_square = 0.0;
        for (int32_t i = 0; i < size_i; ++i) {
            for (int32_t j = 0; j < size_j; ++j) {
                float diff = c_out_host[i * size_j + j] - c[i * size_j + j];
                mse += diff * diff;
                ref_mean_square += c[i * size_j + j] * c[i * size_j + j];
            }
        }
        mse /= size_i * size_j;
        ref_mean_square /= size_i * size_j;
        float rmse = std::sqrt(mse);
        float rel_rmse = rmse / std::sqrt(ref_mean_square);

        printf("  size %4d * %4d * %4d:\n", size_i, size_j, size_k);
        printf("    correctness: %.02e relative RMSE\n", rel_rmse);

        if (rel_rmse > 1e-5) {
            printf("    skipping benchmark (incorrect)\n");
        } else {
            double elapsed_ms = benchmark_ms(1000.0, 4, [&]() {
                Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu);
            });

            printf("    run time: %6.02f ms\n", elapsed_ms);

            double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
            printf("    throughput: %5.02f TFLOP/s\n", tflop / (elapsed_ms * 1e-3));

            if (config.save_result) {
                saved_results.push_back({Impl::name, elapsed_ms});
            }
        }

        printf("\n");
    }
}

template <typename Impl>
void run_all_tests(
    std::string const &test_data_dir,
    std::vector<BenchmarkResult> &saved_results) {
    printf("%s:\n\n", Impl::name);
    run_tests_for_size<Impl>(test_data_dir, saved_results, {{256, 256, 256, false}});
    run_tests_for_size<Impl>(test_data_dir, saved_results, {{3072, 3072, 3072, true}});
}

struct MatmulL1 {
    constexpr static char const *name = "matmul_l1";
    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c) {
        matmul_l1::launch_matmul_l1(size_i, size_j, size_k, a, b, c);
    }
};

struct MatmulL1Reg {
    constexpr static char const *name = "matmul_l1_reg";
    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c) {
        matmul_l1_reg::launch_matmul_l1_reg(size_i, size_j, size_k, a, b, c);
    }
};

int main(int argc, char **argv) {
    std::string test_data_dir = ".";

    auto saved_results = std::vector<BenchmarkResult>();

    run_all_tests<MatmulL1>(test_data_dir, saved_results);
    run_all_tests<MatmulL1Reg>(test_data_dir, saved_results);

    if (saved_results.size() > 1) {
        printf("speedups on largest problem size:\n");
        for (int32_t j = 1; j < saved_results.size(); ++j) {
            printf("\n");
            for (int32_t i = j; i > 0;) {
                --i;
                auto const &first = saved_results.at(i);
                auto const &second = saved_results.at(j);
                printf(
                    "  speedup %s -> %s: %.02fx\n",
                    first.name,
                    second.name,
                    first.elapsed_ms / second.elapsed_ms);
            }
        }
    }

    return 0;
}
