#pragma once

#include "hnswalg.h"
#include "hnsw_lid_search.h"
#include "lid_2d_table.h"
#include "../hnswlib/adaptive_ef.h"

#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <unordered_set>
#include <queue>
#include <omp.h>
#include <memory>
#include <thread>

namespace hnswlid {

// ─── smart_proxy_sample ───────────────────────────────────────────────────────
//
// Builds a stratified calibration query set of size n_easy + n_hard + n_random:
//
//   Easy  (n_easy)   — nodes that appear at layer ≥ 1 in the HNSW graph.
//                      These are cluster centres; the greedy descent converges
//                      fast → low macro-LID → anchor the low-ef corner.
//
//   Hard  (n_hard)   — midpoints of uniformly-sampled layer-0 edges.
//                      These sit exactly on Voronoi boundaries where greedy
//                      routing stalls → high macro-LID → anchor the high-ef corner.
//
//   Random (n_random) — easy vectors perturbed by Gaussian noise (σ = noise_std).
//                      Fills the transition region between easy and hard.
//
// All returned vectors are L2-normalised when metric == "cd" (cosine distance)
// to match the normalised index vectors.

inline hnswdis::MatrixXf smart_proxy_sample(
    const hnswlib::HierarchicalNSW<float>& hnsw,
    const hnswdis::MatrixXf&               data,
    const std::string&                     metric,
    int                                    n_easy   = 100,
    int                                    n_hard   = 100,
    int                                    n_random = 100,
    float                                  noise_std = 0.05f,
    unsigned                               seed      = 42)
{
    using tableint = hnswlib::tableint;
    const int dim  = static_cast<int>(data.cols());
    const int N    = static_cast<int>(hnsw.cur_element_count);

    std::mt19937 rng(seed);

    // ── Collect L1+ nodes ─────────────────────────────────────────────────────
    std::vector<tableint> l1_nodes;
    l1_nodes.reserve(N / 4);
    for (tableint i = 0; i < static_cast<tableint>(N); ++i) {
        if (hnsw.element_levels_[i] >= 1)
            l1_nodes.push_back(i);
    }

    // ── Easy: sample n_easy L1 nodes ─────────────────────────────────────────
    std::shuffle(l1_nodes.begin(), l1_nodes.end(), rng);
    int actual_easy = std::min(n_easy, static_cast<int>(l1_nodes.size()));

    // ── Hard: sample n_hard layer-0 edge midpoints ────────────────────────────
    //
    // We randomly pick nodes, then randomly pick one of their L0 neighbours.
    // The midpoint of that edge sits on the Voronoi boundary between the two
    // nodes' Voronoi cells — exactly where greedy routing gets confused.
    std::uniform_int_distribution<int> node_dist(0, N - 1);
    std::vector<std::pair<tableint, tableint>> edge_pairs;
    edge_pairs.reserve(n_hard * 4);

    std::unordered_set<long long> seen_edges;
    int attempts = 0;
    while (static_cast<int>(edge_pairs.size()) < n_hard * 4 && attempts < n_hard * 40) {
        ++attempts;
        tableint u = static_cast<tableint>(node_dist(rng));
        auto* ll   = reinterpret_cast<int*>(hnsw.get_linklist0(u));
        int   sz   = static_cast<int>(hnsw.getListCount(
                         reinterpret_cast<hnswlib::linklistsizeint*>(ll)));
        if (sz == 0) continue;

        std::uniform_int_distribution<int> nbr_dist(0, sz - 1);
        tableint v = reinterpret_cast<tableint*>(ll + 1)[nbr_dist(rng)];

        long long key = static_cast<long long>(std::min(u, v)) * N
                      + static_cast<long long>(std::max(u, v));
        if (!seen_edges.insert(key).second) continue;
        edge_pairs.emplace_back(u, v);
    }
    std::shuffle(edge_pairs.begin(), edge_pairs.end(), rng);
    int actual_hard = std::min(n_hard, static_cast<int>(edge_pairs.size()));

    // ── Allocate output matrix ─────────────────────────────────────────────────
    const int total = actual_easy + actual_hard + n_random;
    hnswdis::MatrixXf out(total, dim);
    int row = 0;

    // Fill easy rows
    for (int i = 0; i < actual_easy; ++i) {
        tableint nid = l1_nodes[i];
        const float* vec = reinterpret_cast<const float*>(hnsw.getDataByInternalId(nid));
        for (int d = 0; d < dim; ++d)
            out(row, d) = vec[d];
        ++row;
    }

    // Fill hard rows (edge midpoints)
    for (int i = 0; i < actual_hard; ++i) {
        auto [u, v] = edge_pairs[i];
        const float* vu = reinterpret_cast<const float*>(hnsw.getDataByInternalId(u));
        const float* vv = reinterpret_cast<const float*>(hnsw.getDataByInternalId(v));
        for (int d = 0; d < dim; ++d)
            out(row, d) = 0.5f * (vu[d] + vv[d]);
        ++row;
    }

    // Fill random rows (easy + Gaussian noise)
    {
        std::normal_distribution<float> gauss(0.0f, noise_std);
        std::uniform_int_distribution<int> easy_pick(0, actual_easy - 1);
        for (int i = 0; i < n_random; ++i) {
            int src = easy_pick(rng);
            for (int d = 0; d < dim; ++d)
                out(row, d) = out(src, d) + gauss(rng);
            ++row;
        }
    }

    // L2-normalise if cosine distance
    if (metric == "cd") {
        for (int i = 0; i < total; ++i) {
            float norm = out.row(i).norm();
            if (norm > 1e-10f)
                out.row(i) /= norm;
        }
    }

    std::cout << "smart_proxy_sample: "
              << actual_easy << " easy, "
              << actual_hard << " hard, "
              << n_random    << " random  (total=" << total << ")" << std::endl;
    return out;
}

// ─── calibrate_lid_2d_table ───────────────────────────────────────────────────
//
// Control variables matched to ada-ef for fair comparison:
//   sample_size = 300  (100 easy + 100 hard + 100 random — same order of
//                       magnitude as ada-ef's 200 uniform random samples)
//   ef_max      = 5000 (identical to ada-ef)
//   ef_step     = 10   (identical to ada-ef's sweep granularity)
//   target_recall       identical to whatever expected_recall is passed
//
// The only differing variable is the sampling strategy (stratified vs. uniform).

inline Lid2DTable calibrate_lid_2d_table(
    hnswlib::HierarchicalNSW<float>&  hnsw,
    const hnswdis::MatrixXf&          data,
    int                               k,
    const std::string&                metric,
    float                             target_recall = 0.95f,
    int                               ef_max        = 5000,
    int                               ef_step       = 10,
    int                               n_easy        = 100,
    int                               n_hard        = 100,
    int                               n_random      = 100)
{
    // ── Step 1: Smart stratified proxy queries ─────────────────────────────────
    const hnswdis::MatrixXf queries = smart_proxy_sample(
        hnsw, data, metric, n_easy, n_hard, n_random);
    const int nq = static_cast<int>(queries.rows());

    // ── Step 2: Brute-force ground truth ──────────────────────────────────────
    std::cout << "Computing brute-force ground truth for "
              << nq << " calibration queries..." << std::endl;
    hnswdis::MatrixXi ground_truth(nq, k);
    {
        const int N = static_cast<int>(data.rows());
        hnswdis::MatrixXf sims = queries * data.transpose();

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

    // ── Step 3: Collect (macro_lid, micro_lid) for each proxy query ───────────
    std::cout << "Collecting Macro/Micro LID features..." << std::endl;
    Lid2DTable dummy_table;
    std::vector<float> macro_samples(nq);
    std::vector<float> micro_samples(nq);

    for (int qi = 0; qi < nq; ++qi) {
        auto r = instrumented_lid_search(hnsw, queries.row(qi).data(), k, dummy_table, k);
        macro_samples[qi] = r.triad.macro_lid;
        micro_samples[qi] = r.triad.micro_lid;
    }

    // ── Step 4: Sweep ef per query to find minimum ef hitting target recall ────
    std::cout << "Sweeping ef (step=" << ef_step
              << ", max=" << ef_max << ")..." << std::endl;
    std::vector<Lid2DTable::Obs> observations;
    observations.reserve(nq);

    for (int qi = 0; qi < nq; ++qi) {
        std::unordered_set<int> gt_set;
        for (int j = 0; j < k; ++j)
            gt_set.insert(ground_truth(qi, j));

        int optimal_ef = k;
        for (int ef = k; ef <= ef_max; ef += ef_step) {
            hnsw.setEf(ef);
            auto ret = hnsw.searchKnn(queries.row(qi).data(), k);

            int correct = 0;
            while (!ret.empty()) {
                if (gt_set.count(static_cast<int>(ret.top().second)))
                    ++correct;
                ret.pop();
            }

            optimal_ef = ef;
            if (static_cast<float>(correct) / static_cast<float>(k) >= target_recall)
                break;
        }
        observations.push_back({macro_samples[qi], micro_samples[qi], optimal_ef});
    }

    // ── Step 5+6: Build 2D table ───────────────────────────────────────────────
    std::cout << "Building 2D grid..." << std::endl;
    Lid2DTable table;
    table.build_from_observations(observations, k, ef_max);
    std::cout << "2D calibration complete." << std::endl;
    table.print();
    return table;
}

} // namespace hnswlid
