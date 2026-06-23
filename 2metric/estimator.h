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
    hnswlib::tableint ep_id; // <-- 2METRIC EXTENSION: Cached Layer 0 enterpoint
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
        res.ep_id = currObj; // Save for fast-path resume
        return res;
    }
};

} // namespace hnsw_2metric
