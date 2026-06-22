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

class Estimator2Metric {
public:

    static EstimatorResult probe_query(
        hnswlib::HierarchicalNSW<float>* alg_hnsw,
        const float* query,
        const Eigen::RowVectorXf& global_mean,
        int ef_probe_cap = 100,
        float gamma = 15.0f
    ) {
        int L = alg_hnsw->maxlevel_;

        // 2. Greedily drop down from the top layer to layer 0 to find the entry point.
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

        // 3. Bounded probe on Layer 0. Use a flat visited bitvector for O(1) lookup.
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

        float lowerBound = curdist;

        // Collect (dist, is_revisit) pairs for revisit_rank. Pre-reserve to avoid realloc.
        std::vector<std::pair<float, bool>> edges;
        edges.reserve(ef_probe_cap * 16);

        while (!candidate_set.empty()) {
            auto [cur_d, cur_node] = candidate_set.top();
            if (cur_d > lowerBound && (int)top_candidates.size() == ef_probe_cap) break;
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
                    if ((int)top_candidates.size() < ef_probe_cap || lowerBound > d) {
                        candidate_set.emplace(d, cand);
                        top_candidates.emplace(d, cand);
                        if ((int)top_candidates.size() > ef_probe_cap) top_candidates.pop();
                        lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        // 4. Sort edges by distance, then accumulate revisit_rank with exp-decay weights.
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
        res.entry_point_dist    = curdist;
        res.revisit_rank = sum_vis / std::max(1e-6f, sum_tot);
        return res;
    }
};

} // namespace hnsw_2metric
