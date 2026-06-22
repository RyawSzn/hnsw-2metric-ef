#pragma once

#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>

#include "../hnswlib/hnswlib.h"

namespace hnsw_2metric {

/**
 * @struct EstimatorResult
 * @brief Holds the calculated hardness metrics for a given query.
 * 
 * RC (Relative Contrast): Measures how close the query is to the global mean versus its 
 * nearest neighbors. High RC = easy query (high contrast).
 * 
 * RV_rank (Topological Rank Entrapment): Measures the ratio of revisited nodes during 
 * a short probe, weighted by their distance rank. High RV = hard query (trapped in a hub).
 */
struct EstimatorResult {
    float d_ep;
    float RV_rank;
};

/**
 * @struct EdgeEval
 * @brief Represents an evaluated edge during the probe phase.
 */
struct ResearchEstimatorResult {
    float RC;
    float RV_rank;
    float d_mean;
    float d_ep;
    float m_LID;
};

struct EdgeEval {
    float dist;
    bool is_revisit;
};

class Estimator2Metric {
public:
    static ResearchEstimatorResult probe_query_research(
        hnswlib::HierarchicalNSW<float>* alg_hnsw, 
        const float* query, 
        const Eigen::RowVectorXf& global_mean, 
        int ef_probe_cap = 100,
        float gamma = 15.0f
    ) {
        int L = alg_hnsw->maxlevel_;
        float d_mean = alg_hnsw->fstdistfunc_(query, global_mean.data(), alg_hnsw->dist_func_param_);
        d_mean = std::max(1e-6f, d_mean);

        hnswlib::tableint currObj = alg_hnsw->enterpoint_node_;
        float curdist = alg_hnsw->fstdistfunc_(query, alg_hnsw->getDataByInternalId(currObj), alg_hnsw->dist_func_param_);

        std::vector<float> upper_dists;
        upper_dists.push_back(curdist);

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
                    upper_dists.push_back(d);
                    if (d < curdist) { curdist = d; currObj = cand; changed = true; }
                }
            }
        }
        
        float d_ep = curdist; // entry point distance to layer 0

        // Compute m_LID
        float m_LID = 0.0f;
        if (!upper_dists.empty()) {
            float d_max = *std::max_element(upper_dists.begin(), upper_dists.end());
            float sum_log = 0.0f;
            for (float d : upper_dists) {
                sum_log += std::log((d + 1e-10f) / (d_max + 1e-10f));
            }
            if (-sum_log > 1e-6f) m_LID = upper_dists.size() / -sum_log;
        }

        std::priority_queue<std::pair<float, hnswlib::tableint>> top_candidates;
        std::priority_queue<std::pair<float, hnswlib::tableint>, std::vector<std::pair<float, hnswlib::tableint>>, std::greater<std::pair<float, hnswlib::tableint>>> candidate_set;
        std::unordered_set<hnswlib::tableint> visited;

        visited.insert(currObj);
        candidate_set.emplace(curdist, currObj);
        top_candidates.emplace(curdist, currObj);

        float lowerBound = curdist;
        float best_d = curdist;
        std::vector<EdgeEval> edges;

        while (!candidate_set.empty()) {
            auto current_node_pair = candidate_set.top();
            if (current_node_pair.first > lowerBound && top_candidates.size() == ef_probe_cap) break;
            candidate_set.pop();
            hnswlib::tableint cur_node = current_node_pair.second;

            auto* data = reinterpret_cast<int*>(alg_hnsw->get_linklist0(cur_node));
            int sz = alg_hnsw->getListCount(reinterpret_cast<hnswlib::linklistsizeint*>(data));
            auto* nbrs = reinterpret_cast<hnswlib::tableint*>(data + 1);

            for (int j = 0; j < sz; j++) {
                hnswlib::tableint cand = nbrs[j];
                float d = alg_hnsw->fstdistfunc_(query, alg_hnsw->getDataByInternalId(cand), alg_hnsw->dist_func_param_);
                
                if (d < best_d) best_d = d;
                bool is_revisit = (visited.find(cand) != visited.end());
                edges.push_back({d, is_revisit});

                if (!is_revisit) {
                    visited.insert(cand);
                    if (top_candidates.size() < ef_probe_cap || lowerBound > d) {
                        candidate_set.emplace(d, cand);
                        top_candidates.emplace(d, cand);
                        if (top_candidates.size() > ef_probe_cap) top_candidates.pop();
                        lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        std::sort(edges.begin(), edges.end(), [](const EdgeEval& a, const EdgeEval& b) {
            return a.dist < b.dist;
        });

        int N = edges.size();
        float sum_tot = 0, sum_vis = 0;
        for (int i = 0; i < N; i++) {
            float rank_ratio = (float)(i + 1) / N;
            float w = std::exp(-gamma * rank_ratio);
            sum_tot += w;
            if (edges[i].is_revisit) sum_vis += w;
        }

        ResearchEstimatorResult res;
        res.d_ep = curdist;
        res.RV_rank = sum_vis / std::max(1e-6f, sum_tot);
        res.d_mean = d_mean;
        res.d_ep = d_ep;
        res.m_LID = m_LID;
        
        return res;
    }


    /**
     * @brief Probes the HNSW graph to estimate the difficulty of a query.
     * 
     * This function simulates a short HNSW search (up to ef_probe_cap) to gather statistics
     * about the local graph topology around the query.
     * 
     * @param alg_hnsw Pointer to the initialized HNSW index.
     * @param query The query vector data.
     * @param global_mean The precomputed global centroid of the entire dataset.
     * @param ef_probe_cap The maximum number of candidates to evaluate during the probe (default 100).
     * @param gamma The decay factor for rank weighting (universally optimal at 15.0f).
     * @return EstimatorResult containing RC and RV_rank.
     */
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

        // Collect (dist, is_revisit) pairs for RV_rank. Pre-reserve to avoid realloc.
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

        // 4. Sort edges by distance, then accumulate RV_rank with exp-decay weights.
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
        res.d_ep    = curdist;
        res.RV_rank = sum_vis / std::max(1e-6f, sum_tot);
        return res;
    }
};

} // namespace hnsw_2metric
