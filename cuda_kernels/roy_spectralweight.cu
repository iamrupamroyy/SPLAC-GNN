/*
 * =====================================================================================
 *
 *  SCALABILITY LIMITATIONS:
 *
 *  This implementation is not suitable for very large-scale graphs (i.e., graphs
 *  that do not fit entirely within GPU memory).
 *
 *  1. High Memory Consumption: The entire graph structure (CSR), weights, and all
 *     intermediate arrays (degrees, normalized weights, spectral distances) are stored
 *     in GPU memory. This makes the approach infeasible for graphs that exceed the
 *     available VRAM.
 *
 *  2. High Computational Complexity: The `compute_spectral_distances` kernel can be a
 *     significant bottleneck for graphs containing high-degree "hub" nodes, as its
 *     complexity for a single edge is proportional to the sum of the degrees of the
 *     two incident nodes.
 *
 * =====================================================================================
 */

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <sys/time.h>
#include <cfloat>
#include <algorithm>

#define SPECTRAL_WEIGHT_SCALE 100.0f

#if defined(__CUDACC__)
#define ROUND_FUNCTION roundf
#else
#define ROUND_FUNCTION std::round
#endif

__global__ void compute_degree_and_normalize(int N, const int* __restrict__ row_ptr, const float* __restrict__ weights, float* degrees, float* norm_weights)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int start = row_ptr[i];
    int end = row_ptr[i + 1];
    float degree = 0.0f;
    for (int e = start; e < end; ++e)
        degree += weights[e];

    degrees[i] = degree;

    if (degree > 0.0f) {
        for (int e = start; e < end; ++e)
            norm_weights[e] = weights[e] / degree;
    } else {
        for (int e = start; e < end; ++e)
            norm_weights[e] = 0.0f;
    }
}

// NOTE: The complexity of this kernel for a single edge (i, j) is O(degree(i) + degree(j)).
// This can be a performance bottleneck for graphs with high-degree "hub" nodes.
__global__ void compute_spectral_distances(
    int N,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float* __restrict__ norm_weights,
    float* spectral_distances_chunk, // Output for the chunk
    int edge_offset, // Starting edge index for this chunk
    int num_chunk_edges)
{
    int e_chunk = blockIdx.x * blockDim.x + threadIdx.x;
    if (e_chunk >= num_chunk_edges) return;

    int e_global = edge_offset + e_chunk; // Global edge index

    // Find the source vertex 'i' for the global edge 'e_global'
    // This is inefficient (logarithmic search), but necessary for this algorithm.
    // A better approach would be to precompute this mapping if memory allows.
    int i = 0;
    int high = N;
    int low = 0;
    while(high > low + 1) {
        int mid = low + (high - low) / 2;
        if(row_ptr[mid] > e_global) {
            high = mid;
        } else {
            low = mid;
        }
    }
    i = low;

    int j = col_ind[e_global];

    int start_i = row_ptr[i];
    int end_i = row_ptr[i + 1];
    int start_j = row_ptr[j];
    int end_j = row_ptr[j + 1];

    float dist = 0.0f;
    int pi = start_i, pj = start_j;

    while (pi < end_i && pj < end_j) {
        int ni = col_ind[pi];
        int nj = col_ind[pj];

        if (ni == nj) {
            dist += fabsf(norm_weights[pi] - norm_weights[pj]);
            ++pi;
            ++pj;
        } else if (ni < nj) {
            dist += norm_weights[pi];
            ++pi;
        } else {
            dist += norm_weights[pj];
            ++pj;
        }
    }
    while (pi < end_i) dist += norm_weights[pi++];
    while (pj < end_j) dist += norm_weights[pj++];

    spectral_distances_chunk[e_chunk] = dist;
}

__global__ void compute_spectral_weights(
    int num_chunk_edges,
    const float* __restrict__ spectral_distances_chunk,
    int* spectral_weights_chunk
)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_chunk_edges) return;

    const float dist = spectral_distances_chunk[e];
    const float epsilon = 1e-7f; // Small epsilon to prevent division by zero
    const int MAX_INT_WEIGHT = 2147483647;
    
    // Calculate weight, ensuring stability with epsilon
    float weight_float = (1.0f / (dist + epsilon)) + 1.0f;

    // Clamp the value before scaling to prevent overflow during multiplication
    const float max_safe_float = (float)MAX_INT_WEIGHT / SPECTRAL_WEIGHT_SCALE;
    if (weight_float > max_safe_float) {
        weight_float = max_safe_float;
    }

    // Scale and round the weight
    int weight_int = (int)roundf(weight_float * SPECTRAL_WEIGHT_SCALE);

    // Final check to ensure the value is within the valid integer range
    if (weight_int < 0) {
        weight_int = 0;
    }

    spectral_weights_chunk[e] = weight_int;
}


std::vector<int> readIntFile(const std::string& filename) {
    std::vector<int> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << ". Assuming empty data.\n";
        return data;
    }
    int value;
    while (file >> value)
        data.push_back(value);
    return data;
}

std::vector<float> readFloatFile(const std::string& filename) {
    std::vector<float> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        return data;
    }
    float value;
    while (file >> value)
        data.push_back(value);
    return data;
}

void writeFloatFile(const std::string& filename, const thrust::host_vector<float>& data) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening output file: " << filename << "\n";
        return;
    }
    for (auto x : data) file << x << "\n";
    file.close();
}

// Modified to support appending
void writeIntFile(const std::string& filename, const thrust::host_vector<int>& data, bool append = false) {
    std::ofstream file(filename, append ? std::ios_base::app : std::ios_base::trunc);
    if (!file.is_open()) {
        std::cerr << "Error opening output file: " << filename << "\n";
        return;
    }
    for (auto x : data) file << x << "\n";
    file.close();
}

double getTime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <directory>\n";
        return 1;
    }

    const std::string dir = argv[1];

    auto h_row_ptr = readIntFile(dir + "/row.txt");
    auto h_col_ind = readIntFile(dir + "/column.txt");
    auto h_weights_in = readFloatFile(dir + "/weight.txt");

    int N = (int)h_row_ptr.size() - 1;
    int M = (int)h_col_ind.size();

    if (N <= 0 || M <= 1) {
        std::cerr << "Graph data is too small to compute more than one spectral weight (N=" << N << ", M=" << M << "). At least 2 edges are required.\n";
        return -1;
    }

    if (h_weights_in.empty()) {
        std::cout << "Weight file not found. Initializing all edge weights to 1.0.\n";
        h_weights_in.resize(M, 1.0f);
    }
    if (h_weights_in.size() != M) {
        std::cerr << "Error: Number of weights (" << h_weights_in.size() << ") does not match number of edges (" << M << ").\n";
        return -1;
    }

    // --- Main graph topology MUST fit in GPU memory ---
    thrust::device_vector<int> d_row_ptr(h_row_ptr);
    thrust::device_vector<int> d_col_ind(h_col_ind);
    thrust::device_vector<float> d_weights(h_weights_in);
    
    // --- Intermediate arrays that also must fit ---
    thrust::device_vector<float> d_degrees(N, 0);
    thrust::device_vector<float> d_norm_weights(M, 0);

    double startNormalize = getTime();
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    compute_degree_and_normalize<<<gridSize, blockSize>>>(
        N,
        thrust::raw_pointer_cast(d_row_ptr.data()),
        thrust::raw_pointer_cast(d_weights.data()),
        thrust::raw_pointer_cast(d_degrees.data()),
        thrust::raw_pointer_cast(d_norm_weights.data())
    );
    cudaDeviceSynchronize();
    std::cout<<"Kernel time for Normalize edge weights: " << (getTime() - startNormalize) << " seconds\n";

    // --- CHUNKING LOGIC STARTS HERE ---
    // Reduced chunk size for better parallel execution support
    const int chunk_size = 1000000; 
    
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "GPU Memory: Free=" << free_mem/(1024*1024) << "MB, Total=" << total_mem/(1024*1024) << "MB\n";

    // Prepare output files by clearing them first
    // const std::string spectral_dist_file = "./SpectralOutput/spectral_distances_csr.txt";
    const std::string spectral_weight_file = "./SpectralOutput/spectral_weights_csr.txt";
    // std::ofstream(spectral_dist_file, std::ios_base::trunc).close();
    std::ofstream(spectral_weight_file, std::ios_base::trunc).close();

    std::cout << "Starting edge processing in chunks of " << chunk_size << "...\n";
    double total_dist_time = 0;
    double total_weight_time = 0;

    for (int edge_offset = 0; edge_offset < M; edge_offset += chunk_size) {
        int num_chunk_edges = std::min(chunk_size, M - edge_offset);
        
        std::cout << "Processing edge chunk: " << edge_offset << " to " << edge_offset + num_chunk_edges - 1 << "\n";

        // Allocate memory for the chunk
        thrust::device_vector<float> d_spectral_distances_chunk(num_chunk_edges);
        thrust::device_vector<int> d_spectral_weights_chunk(num_chunk_edges);

        // --- Compute spectral distances for the chunk ---
        double startSpectralDist = getTime();
        gridSize = (num_chunk_edges + blockSize - 1) / blockSize;
        compute_spectral_distances<<<gridSize, blockSize>>>(
            N,
            thrust::raw_pointer_cast(d_row_ptr.data()),
            thrust::raw_pointer_cast(d_col_ind.data()),
            thrust::raw_pointer_cast(d_norm_weights.data()),
            thrust::raw_pointer_cast(d_spectral_distances_chunk.data()),
            edge_offset,
            num_chunk_edges
        );
        cudaDeviceSynchronize();
        total_dist_time += (getTime() - startSpectralDist);

        // --- Compute spectral weights for the chunk ---
        double startSpectralWeight = getTime();
        compute_spectral_weights<<<gridSize, blockSize>>>(
            num_chunk_edges,
            thrust::raw_pointer_cast(d_spectral_distances_chunk.data()),
            thrust::raw_pointer_cast(d_spectral_weights_chunk.data())
        );
        cudaDeviceSynchronize();
        total_weight_time += (getTime() - startSpectralWeight);

        // --- Copy results to host and write to file ---
        thrust::host_vector<float> h_spectral_distances_chunk = d_spectral_distances_chunk;
        thrust::host_vector<int> h_spectral_weights_chunk = d_spectral_weights_chunk;
        
        // Append results to the output files
        writeIntFile(spectral_weight_file, h_spectral_weights_chunk, true);
        // Also writing distances for inspection, can be commented out for performance
        // writeFloatFile(spectral_dist_file, h_spectral_distances_chunk, true); // Need to modify writeFloatFile to support append

    } // End of chunking loop

    std::cout << "\n--- Processing Complete ---\n";
    std::cout << "Total kernel time for Spectral distance compute: " << total_dist_time << " seconds\n";
    std::cout << "Total kernel time for Spectral weight compute: " << total_weight_time << " seconds\n";
    std::cout << "Scaled Integer Spectral weights (CSR format) written to: " << spectral_weight_file << "\n";

    return 0;
}
