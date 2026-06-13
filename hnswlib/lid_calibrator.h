#pragma once

#include "hnswalg.h"
#include "hnsw_lid_search.h"
#include "lid_estimator.h"
#include "../hnswlib/adaptive_ef.h"

#include <vector>
#include <map>
#include <random>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <cmath>
#include <iostream>
#include <unordered_set>

namespace hnswlid {

// ─── Proxy query matrix ───────────────────────────────────────────────────────
//
// Three populations, N_EACH rows each:
//   [0,        N_EACH)   Layer-1 nodes  — easy / core queries
//   [N_EACH,   2*N_EACH) Edge midpoints — hard / boundary queries
//   [2*N_EACH, 3*N_EACH) Noisy vectors  — outlier queries
//
// Total rows = 3 * N_EACH (clamped to whatever is available in the index).

static constexpr int N_EACH = 100;

struct ProxySet {
    hnswdis::MatrixXf queries;
    int n_easy{0};
    int n_hard{0};
    int n_outlier{0};
};

// ─── generate_proxy_queries ───────────────────────────────────────────────────
//
// Walks the loaded HNSW graph to generate representative proxy queries.
//
// Easy (layer >= 1 nodes):
//   Nodes at layer 1+ are the "hub" nodes that form the coarse navigation
//   graph. A query at such a node's location is easy — many paths reach it.
//
// Hard (edge midpoints, layer >= 2):
//   The midpoint of an edge between two layer-2 neighbors lies equidistant
//   from both cluster centres — a boundary region where routing is uncertain.
//   These are the hardest queries for HNSW because the greedy descent may
//   commit to the wrong side.
//
// Outlier (random Gaussian noise):
//   Vectors drawn from N(0, sigma^2) with sigma matched to the average
//   magnitude of data vectors. These have no nearby neighbours and force
//   HNSW to explore widely.

inline ProxySet generate_proxy_queries(
    const hnswlib::HierarchicalNSW<float>& hnsw,
    unsigned seed = 42)
{
    using tableint = hnswlib::tableint;

    const int n       = static_cast<int>(hnsw.cur_element_count);
    const int dim     = static_cast<int>(hnsw.data_size_ / sizeof(float));
    const int n_each  = N_EACH;

    std::mt19937 rng(seed);

    // ── 1. Easy: sample up to N_EACH nodes with element_levels_[id] >= 1 ──

    std::vector<tableint> layer1_nodes;
    layer1_nodes.reserve(512);
    for (int id = 0; id < n; ++id) {
        if (hnsw.element_levels_[id] >= 1)
            layer1_nodes.push_back(static_cast<tableint>(id));
    }
    std::shuffle(layer1_nodes.begin(), layer1_nodes.end(), rng);
    int n_easy = std::min(n_each, static_cast<int>(layer1_nodes.size()));

    // ── 2. Hard: edge midpoints from layer >= 2 ───────────────────────────
    //
    // For each node at level >= 2, iterate its layer-2 neighbour list.
    // Compute midpoint = (vec_a + vec_b) / 2 for each edge (a, b).
    // Sample N_EACH midpoints.

    struct EdgeMid { std::vector<float> vec; };
    std::vector<EdgeMid> midpoints;
    midpoints.reserve(n_each * 4);

    for (int id = 0; id < n && static_cast<int>(midpoints.size()) < n_each * 4; ++id) {
        if (hnsw.element_levels_[id] < 2) continue;

        auto* ll = hnsw.get_linklist(static_cast<tableint>(id), 2);
        int sz = static_cast<int>(hnsw.getListCount(ll));
        auto* nbrs = reinterpret_cast<const tableint*>(
            reinterpret_cast<const char*>(ll) + sizeof(hnswlib::linklistsizeint));

        const float* va = reinterpret_cast<const float*>(
            hnsw.getDataByInternalId(static_cast<tableint>(id)));

        for (int j = 0; j < sz; ++j) {
            tableint nb = nbrs[j];
            const float* vb = reinterpret_cast<const float*>(
                hnsw.getDataByInternalId(nb));

            EdgeMid em;
            em.vec.resize(dim);
            for (int d = 0; d < dim; ++d)
                em.vec[d] = 0.5f * (va[d] + vb[d]);
            midpoints.push_back(std::move(em));
        }
    }
    std::shuffle(midpoints.begin(), midpoints.end(), rng);
    int n_hard = std::min(n_each, static_cast<int>(midpoints.size()));

    // ── 3. Outlier: Gaussian noise scaled to avg data vector magnitude ────
    //
    // Estimate average L2 norm from a small sample of data vectors.

    float avg_norm = 0.0f;
    int norm_samples = std::min(200, n);
    for (int i = 0; i < norm_samples; ++i) {
        const float* v = reinterpret_cast<const float*>(
            hnsw.getDataByInternalId(static_cast<tableint>(i)));
        float s = 0.0f;
        for (int d = 0; d < dim; ++d) s += v[d] * v[d];
        avg_norm += std::sqrt(s);
    }
    avg_norm /= static_cast<float>(norm_samples);
    float sigma = avg_norm / std::sqrt(static_cast<float>(dim));

    std::normal_distribution<float> gauss(0.0f, sigma);
    int n_outlier = n_each;

    // ── Assemble into MatrixXf ────────────────────────────────────────────

    int total = n_easy + n_hard + n_outlier;
    ProxySet ps;
    ps.queries.resize(total, dim);
    ps.n_easy    = n_easy;
    ps.n_hard    = n_hard;
    ps.n_outlier = n_outlier;

    int row = 0;
    for (int i = 0; i < n_easy; ++i, ++row) {
        const float* v = reinterpret_cast<const float*>(
            hnsw.getDataByInternalId(layer1_nodes[i]));
        std::copy(v, v + dim, ps.queries.row(row).data());
    }
    for (int i = 0; i < n_hard; ++i, ++row) {
        std::copy(midpoints[i].vec.begin(), midpoints[i].vec.end(),
                  ps.queries.row(row).data());
    }
    for (int i = 0; i < n_outlier; ++i, ++row) {
        for (int d = 0; d < dim; ++d)
            ps.queries(row, d) = gauss(rng);
    }

    std::cout << "Proxy queries: " << n_easy << " easy, "
              << n_hard << " hard, " << n_outlier << " outlier. "
              << "Total=" << total << ", dim=" << dim << std::endl;

    return ps;
}

// ─── calibrate_lid_table ──────────────────────────────────────────────────────
//
// Full calibration pipeline:
//
//   1. Generate proxy queries (easy / hard / outlier).
//   2. Brute-force exact top-K ground truth for every proxy query.
//   3. Run Phase 1+2 of instrumented_lid_search on each proxy to get its Score.
//      Group proxies into score buckets.
//   4. For each bucket, sweep ef from K upward in steps of EF_STEP until the
//      bucket's average recall reaches target_recall.
//   5. Return a calibrated, monotone LidEfTable.
//
// Parameters
//   hnsw           : loaded index (read-only)
//   data           : full corpus MatrixXf (needed for brute-force GT)
//   k              : number of neighbours
//   metric         : "cd" (cosine distance) — same as run.cpp
//   target_recall  : e.g. 0.95
//   ef_max         : upper bound on ef (sweep stops here)
//   ef_step        : increment per sweep iteration (default 10)

inline LidEfTable calibrate_lid_table(
    hnswlib::HierarchicalNSW<float>&  hnsw,
    const hnswdis::MatrixXf&          data,
    int                               k,
    const std::string&                metric,
    float                             target_recall = 0.95f,
    int                               ef_max        = 5000,
    int                               ef_step       = 10,
    unsigned                          seed          = 42)
{
    // ── Step 1: Generate proxy queries ────────────────────────────────────
    ProxySet ps = generate_proxy_queries(hnsw, seed);
    const int nq  = static_cast<int>(ps.queries.rows());
    const int dim = static_cast<int>(ps.queries.cols());

    // ── Step 2: Brute-force ground truth ──────────────────────────────────
    //
    // compute_ground_truth_batch_parallel4 does a full matrix multiply
    // (queries × data^T) then selects top-K per row. For 300 queries this
    // is fast even on large corpora.
    //
    // NOTE: the function lives in an hnswdis::Estimator instance context
    // in adaptive_ef.h, but it is also a free function within the
    // hnswdis namespace when called from there. We replicate the same
    // logic here inline to avoid instantiating an Estimator.

    std::cout << "Computing brute-force ground truth for " << nq
              << " proxy queries..." << std::endl;

    hnswdis::MatrixXi ground_truth(nq, k);
    {
        // MatrixXf dot product: (nq x dim) * (dim x N) = (nq x N)
        // Uses inner-product similarity (cosine after normalization).
        // adaptive_ef.h's compute_ground_truth_batch_parallel4 assumes
        // normalized vectors and metric="cd" → similarity = dot product,
        // distance = 1 - dot. We do the same inline.

        const int N = static_cast<int>(data.rows());
        hnswdis::MatrixXf sims = ps.queries * data.transpose(); // (nq x N)

        int num_threads = static_cast<int>(
            std::max(1u, std::thread::hardware_concurrency() / 4));
        omp_set_num_threads(num_threads);

#pragma omp parallel for schedule(static)
        for (int qi = 0; qi < nq; ++qi) {
            std::priority_queue<std::pair<float, int>> heap;
            for (int di = 0; di < std::min(k, N); ++di)
                heap.emplace(1.0f - sims(qi, di), di);
            float worst = heap.top().first;
            for (int di = k; di < N; ++di) {
                float dist = 1.0f - sims(qi, di);
                if (dist < worst) {
                    heap.emplace(dist, di);
                    if (static_cast<int>(heap.size()) > k) heap.pop();
                    worst = heap.top().first;
                }
            }
            int cnt = k;
            while (!heap.empty()) {
                ground_truth(qi, --cnt) = heap.top().second;
                heap.pop();
            }
        }
        std::cout << "Ground truth done." << std::endl;
    }

    // ── Step 3: Score each proxy via Phase 1+2, group into buckets ────────
    //
    // We use a dummy table (linear prior) just to run the instrumented
    // search and extract the Score — the ef returned by the dummy table
    // is not used for the calibration sweep.

    LidEfTable dummy_table;
    dummy_table.build_linear(k, ef_max, 200);

    std::map<int, std::vector<int>> bucket_to_proxy_indices;

    std::cout << "Scoring proxy queries..." << std::endl;
    for (int qi = 0; qi < nq; ++qi) {
        auto r = instrumented_lid_search(
            hnsw, ps.queries.row(qi).data(), k, dummy_table, k);
        bucket_to_proxy_indices[r.score].push_back(qi);
    }

    std::cout << "Score buckets found: " << bucket_to_proxy_indices.size()
              << std::endl;
    for (const auto& [sc, idxs] : bucket_to_proxy_indices)
        std::cout << "  score=" << sc << " -> " << idxs.size()
                  << " proxies" << std::endl;

    // ── Step 4: Sweep ef per bucket until bucket avg recall >= target ─────

    LidTableCalibrator calibrator(target_recall, ef_max, k);

    for (const auto& [score, idxs] : bucket_to_proxy_indices) {
        // Build per-bucket ground truth subset.
        int bsz = static_cast<int>(idxs.size());
        hnswdis::MatrixXi gt_bucket(bsz, k);
        for (int bi = 0; bi < bsz; ++bi)
            gt_bucket.row(bi) = ground_truth.row(idxs[bi]);

        int optimal_ef = k;
        for (int ef = k; ef <= ef_max; ef += ef_step) {
            hnsw.setEf(ef);

            // Run standard HNSW search for this ef on all bucket proxies.
            std::vector<std::vector<std::size_t>> results(bsz);
            for (int bi = 0; bi < bsz; ++bi) {
                auto ret = hnsw.searchKnn(
                    ps.queries.row(idxs[bi]).data(), k);
                int cnt = static_cast<int>(ret.size());
                results[bi].resize(cnt);
                while (!ret.empty()) {
                    results[bi][--cnt] = ret.top().second;
                    ret.pop();
                }
            }

            // Compute average recall for this bucket at this ef.
            float avg = 0.0f;
            for (int bi = 0; bi < bsz; ++bi) {
                std::unordered_set<int> gt_set;
                for (int j = 0; j < k; ++j) gt_set.insert(gt_bucket(bi, j));
                int correct = 0;
                for (std::size_t lbl : results[bi])
                    if (gt_set.count(static_cast<int>(lbl))) ++correct;
                avg += static_cast<float>(correct) / static_cast<float>(k);
            }
            avg /= static_cast<float>(bsz);

            optimal_ef = ef;
            if (avg >= target_recall) break;
        }

        std::cout << "  score=" << score
                  << " optimal_ef=" << optimal_ef << std::endl;
        calibrator.add_observation(score, optimal_ef);
    }

    LidEfTable table = calibrator.build();
    std::cout << "Calibration complete." << std::endl;
    table.print();
    return table;
}

} // namespace hnswlid
