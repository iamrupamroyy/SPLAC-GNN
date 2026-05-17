# Algorithm Description and Complexity Analysis

This document provides a formal description of the Efficient Spectral-Structural Graph Coarsening pipeline and a detailed analysis of its computational complexity.

## Algorithm: Efficient Spectral-Structural Coarsening (ESSC)

### Phase 1: CUDA-Accelerated Hybrid Weighting (rupam_file.py + CUDA kernels)
1.  **Local Structural Weighting (Triangle Counting):**
    *   For every edge $(u, v) \in E$, compute the intersection of adjacency lists $\Gamma(u) \cap \Gamma(v)$.
2.  **Global Spectral Weighting:**
    *   Compute normalized transition matrix $P = D^{-1}A$.
    *   Calculate spectral distances based on $L_2$ norm of localized diffusion patterns.
3.  **Hybrid Integration:** Generate $W_{uv} = \alpha \cdot \text{Triangle}(u,v) + (1-\alpha) \cdot \text{Spectral}(u,v)$.

### Phase 2: Parallel Graph Coarsening (rupam_file.py)
4.  **Heavy Edge Matching:** Find a maximal matching $M$ in $G$ by greedily selecting edges with highest $W_{uv}$.
5.  **Graph Contraction:**
    *   Merge each pair $(u, v) \in M$ into a super-node $S$.
    *   **Feature Aggregation:** $X_S = \text{AGG}(X_u, X_v)$ (e.g., mean).
    *   **Edge Reduction:** $E_{S_1, S_2}$ exists if any $u \in S_1$ is connected to $v \in S_2$ in original $G$.

### Phase 3: GNN Training & Inference (node_classification.py)
6.  **Training:** Optimize GNN weights $\theta$ on Coarsened Graph $G'$ for $T$ epochs.
7.  **Strict Testing:** Evaluate $\theta$ on the full Original Graph $G$ using global message passing.

---

## Time Complexity Analysis

Let $n = |V|$ be the number of nodes, $m = |E|$ be the number of edges, $d$ be the average degree, and $f$ be the feature dimensionality.

### 1. Hybrid Weighting (Phase 1)
*   **Triangle Counting:** 
    *   In the CUDA implementation, for each edge, we intersect two sorted adjacency lists.
    *   Complexity: $O(m \cdot d)$. On GPU, this is parallelized across $m$ threads, resulting in an effective runtime of $O(d)$.
*   **Spectral Distance:**
    *   Requires neighbor-of-neighbor traversal for normalization and distance calculation.
    *   Complexity: $O(m \cdot d)$.
*   **Total Phase 1:** $O(m \cdot d)$.

### 2. Graph Coarsening (Phase 2)
*   **Matching:** 
    *   The greedy matching iterates over nodes and their neighbors.
    *   Complexity: $O(n + m)$.
*   **Feature Aggregation:**
    *   Iterates over all nodes once.
    *   Complexity: $O(n \cdot f)$.
*   **Edge Contraction:**
    *   Iterates over all original edges to map them to super-node edges.
    *   Complexity: $O(m)$.
*   **Total Phase 2:** $O(m + nf)$.

### 3. GNN Training & Inference (Phase 3)
*   **Training (Coarsened):**
    *   Let $n'$ and $m'$ be the reduced node/edge counts (e.g., $n' = 0.5n$).
    *   Complexity per epoch: $O(L \cdot (m' \cdot f + n' \cdot f^2))$, where $L$ is number of layers.
*   **Inference (Original):**
    *   Complexity: $O(L \cdot (m \cdot f + n \cdot f^2))$.
*   **Total Training Savings:** Since $m' \ll m$, the training phase speedup is proportional to the coarsening ratio $r$.

---

## Summary Table: Theoretical vs. Practical Efficiency

| Phase | Complexity | Main Bottleneck | Optimization |
| :--- | :--- | :--- | :--- |
| **Weighting** | $O(m \cdot d)$ | Edge-wise intersection | CUDA Parallelism |
| **Coarsening**| $O(m + nf)$ | Feature data movement | Vectorized NumPy/Torch |
| **Training**  | $O(T \cdot m'f)$| Message passing | Reduced Edge Count ($m'$) |

**Conclusion:** The total complexity is dominated by $O(m \cdot d)$ during a one-time pre-processing, which is offset by the $O(T \cdot (m-m') \cdot f)$ savings during the iterative training phase (where $T$ is the number of epochs/HPO trials).
