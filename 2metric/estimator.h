#pragma once

#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>

#include "../hnswlib/hnswlib.h"

namespace hnsw_2metric {

struct EstimatorResult {
    float entry_point_dist;
    float revisit_rank;
};

// Carries the full Layer-0 search state from probe_with_state() so the caller
// can resume at a higher ef without redoing any distance computations.
//
//   top_candidates    = W   (max-heap: top() is farthest, size <= ef_probe_cap)
//   candidate_frontier = W_d (min-heap: nodes queued but not expanded when probe stopped)
//   vl / vl_tag       = VisitedList borrowed from alg_hnsw's pool; caller MUST pass
//                       it back to searchKnnFromProbeState so the pool can reclaim it.
struct ProbeState {
    float entry_point_dist;
    float revisit_rank;

    std::priority_queue<std::pair<float, hnswlib::tableint>> top_candidates;

    std::priority_queue<
        std::pair<float, hnswlib::tableint>,
        std::vector<std::pair<float, hnswlib::tableint>>,
        std::greater<std::pair<float, hnswlib::tableint>>
    > candidate_frontier;

    // Nodes evaluated during probe but discarded because they were worse than ef_probe_cap
    std::vector<std::pair<float, hnswlib::tableint>> discarded_nodes;

    // Borrowed from visited_list_pool_. Ownership transfers to searchKnnFromProbeState.
    hnswlib::VisitedList* vl{nullptr};
    hnswlib::vl_type vl_tag{0};
};

class Estimator2Metric {
public:

    // Stateless probe — kept intact for table_generator.h (omp parallel for).
    static EstimatorResult probe_query(
        hnswlib::HierarchicalNSW<float>* alg_hnsw,
        const float* query,
        const Eigen::RowVectorXf& global_mean,
        int ef_probe_cap = 32,
        float gamma = 16.0f
    ) {
        int L = alg_hnsw->maxlevel_;

        hnswlib::tableint currObj = alg_hnsw->enterpoint_node_;
        float curdist = alg_hnsw->fstdistfunc_(query, alg_hnsw->getDataByInternalId(currObj), alg_hnsw->dist_func_param_);

        for (int level = L; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                auto* data = reinterpret_cast<unsigned int*>(alg_hnsw->get_linklist(currObj, level));
                int sz = alg_hnsw->getListCount(reinterpret_cast<hnswlib::linklistsizeint*>(data));
                auto* nbrs = reinterpret_cast<hnswlib::tableint*>(data + 1);
                for (int j = 0; j < sz; j++) {
                    hnswlib::tableint cand = nbrs[j];
                    float d = alg_hnsw->fstdistfunc_(query, alg_hnsw->getDataByInternalId(cand), alg_hnsw->dist_func_param_);
                    if (d < curdist) { curdist = d; currObj = cand; changed = true; }
                }
            }
        }

        const size_t max_elements = alg_hnsw->max_elements_;
        std::vector<bool> visited(max_elements, false);

        std::priority_queue<std::pair<float, hnswlib::tableint>> top_candidates;
        std::priority_queue<
            std::pair<float, hnswlib::tableint>,
            std::vector<std::pair<float, hnswlib::tableint>>,
            std::greater<std::pair<float, hnswlib::tableint>>
        > candidate_set;

        visited[currObj] = true;
        candidate_set.emplace(curdist, currObj);
        top_candidates.emplace(curdist, currObj);

        float lower_bound = curdist;

        std::vector<std::pair<float, bool>> edges;
        edges.reserve(ef_probe_cap * 16);

        while (!candidate_set.empty()) {
            auto [cur_d, cur_node] = candidate_set.top();
            if (cur_d > lower_bound && (int)top_candidates.size() == ef_probe_cap) break;
            candidate_set.pop();

            auto* data = reinterpret_cast<int*>(alg_hnsw->get_linklist0(cur_node));
            int sz = alg_hnsw->getListCount(reinterpret_cast<hnswlib::linklistsizeint*>(data));
            auto* nbrs = reinterpret_cast<hnswlib::tableint*>(data + 1);

            for (int j = 0; j < sz; j++) {
                hnswlib::tableint cand = nbrs[j];
                float d = alg_hnsw->fstdistfunc_(query, alg_hnsw->getDataByInternalId(cand), alg_hnsw->dist_func_param_);

                bool is_revisit = visited[cand];
                edges.emplace_back(d, is_revisit);

                if (!is_revisit) {
                    visited[cand] = true;
                    if ((int)top_candidates.size() < ef_probe_cap || lower_bound > d) {
                        candidate_set.emplace(d, cand);
                        top_candidates.emplace(d, cand);
                        if ((int)top_candidates.size() > ef_probe_cap) top_candidates.pop();
                        lower_bound = top_candidates.top().first;
                    }
                }
            }
        }

        std::sort(edges.begin(), edges.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        int N = edges.size();
        float sum_tot = 0.0f, sum_vis = 0.0f;
        const float inv_N = N > 0 ? 1.0f / N : 1.0f;
        for (int i = 0; i < N; i++) {
            float w = std::exp(-gamma * (i + 1) * inv_N);
            sum_tot += w;
            if (edges[i].second) sum_vis += w;
        }

        EstimatorResult res;
        res.entry_point_dist = curdist;
        res.revisit_rank = sum_vis / std::max(1e-5f, sum_tot);
        return res;
    }

    // Stateful variant: same computation as probe_query but returns ProbeState
    // (W, W_d frontier, VisitedList from pool) so searchKnnFromProbeState can resume
    // without redoing greedy descent or the probe neighborhood.
    // NOT thread-safe for parallel calls on the same alg_hnsw (pool mutex serializes,
    // but the returned vl* must be consumed before the next probe on the same alg_hnsw).
    static ProbeState probe_with_state(
        hnswlib::HierarchicalNSW<float>* alg_hnsw,
        const float* query,
        const Eigen::RowVectorXf& global_mean,
        int ef_probe_cap = 32,
        float gamma = 16.0f
    ) {
        int L = alg_hnsw->maxlevel_;

        hnswlib::tableint currObj = alg_hnsw->enterpoint_node_;
        float curdist = alg_hnsw->fstdistfunc_(query, alg_hnsw->getDataByInternalId(currObj), alg_hnsw->dist_func_param_);

        for (int level = L; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                auto* data = reinterpret_cast<unsigned int*>(alg_hnsw->get_linklist(currObj, level));
                int sz = alg_hnsw->getListCount(reinterpret_cast<hnswlib::linklistsizeint*>(data));
                auto* nbrs = reinterpret_cast<hnswlib::tableint*>(data + 1);
                for (int j = 0; j < sz; j++) {
                    hnswlib::tableint cand = nbrs[j];
                    float d = alg_hnsw->fstdistfunc_(query, alg_hnsw->getDataByInternalId(cand), alg_hnsw->dist_func_param_);
                    if (d < curdist) { curdist = d; currObj = cand; changed = true; }
                }
            }
        }

        // Borrow a VisitedList from the pool — O(1) reset, no memset per query.
        hnswlib::VisitedList* vl = alg_hnsw->visited_list_pool_->getFreeVisitedList();
        hnswlib::vl_type* visited_array = vl->mass;
        hnswlib::vl_type tag = vl->curV;

        // W: max-heap, top() is farthest (worst) among the best ef_probe_cap seen
        std::priority_queue<std::pair<float, hnswlib::tableint>> top_candidates;
        // candidate_set: min-heap, top() is nearest (best) node to expand next
        std::priority_queue<
            std::pair<float, hnswlib::tableint>,
            std::vector<std::pair<float, hnswlib::tableint>>,
            std::greater<std::pair<float, hnswlib::tableint>>
        > candidate_set;

        visited_array[currObj] = tag;
        candidate_set.emplace(curdist, currObj);
        top_candidates.emplace(curdist, currObj);
        float lower_bound = curdist;

        std::vector<std::pair<float, bool>> edges;
        edges.reserve(ef_probe_cap * 16);

        while (!candidate_set.empty()) {
            auto [cur_d, cur_node] = candidate_set.top();
            if (cur_d > lower_bound && (int)top_candidates.size() == ef_probe_cap) break;
            candidate_set.pop();

            auto* data = reinterpret_cast<int*>(alg_hnsw->get_linklist0(cur_node));
            int sz = alg_hnsw->getListCount(reinterpret_cast<hnswlib::linklistsizeint*>(data));
            auto* nbrs = reinterpret_cast<hnswlib::tableint*>(data + 1);

            for (int j = 0; j < sz; j++) {
                hnswlib::tableint cand = nbrs[j];
                float d = alg_hnsw->fstdistfunc_(query, alg_hnsw->getDataByInternalId(cand), alg_hnsw->dist_func_param_);

                bool is_revisit = (visited_array[cand] == tag);
                edges.emplace_back(d, is_revisit);

                if (!is_revisit) {
                    visited_array[cand] = tag;
                    if ((int)top_candidates.size() < ef_probe_cap || lower_bound > d) {
                        candidate_set.emplace(d, cand);
                        top_candidates.emplace(d, cand);
                        if ((int)top_candidates.size() > ef_probe_cap) top_candidates.pop();
                        lower_bound = top_candidates.top().first;
                    }
                }
            }
        }
        // candidate_set is now the unexpanded frontier W_d.

        std::sort(edges.begin(), edges.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        int N = edges.size();
        float sum_tot = 0.0f, sum_vis = 0.0f;
        const float inv_N = N > 0 ? 1.0f / N : 1.0f;
        for (int i = 0; i < N; i++) {
            float w = std::exp(-gamma * (i + 1) * inv_N);
            sum_tot += w;
            if (edges[i].second) sum_vis += w;
        }

        ProbeState state;
        state.entry_point_dist   = curdist;
        state.revisit_rank       = sum_vis / std::max(1e-5f, sum_tot);
        state.top_candidates     = std::move(top_candidates);
        state.candidate_frontier = std::move(candidate_set);
        state.vl                 = vl;
        state.vl_tag             = tag;
        return state;
    }
};

} // namespace hnsw_2metric
