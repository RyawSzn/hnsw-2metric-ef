#pragma once

#include <vector>
#include <queue>
#include <unordered_set>
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
    float RC;
    float RV_rank;
};

/**
 * @struct EdgeEval
 * @brief Represents an evaluated edge during the probe phase.
 */
struct EdgeEval {
    float dist;
    bool is_revisit;
};

class Estimator2Metric {
public:
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
        
        // 1. Calculate the distance from the query to the global dataset mean.
        // We use the space's native distance function to remain metric-agnostic (L2 or InnerProduct).
        float d_mean = alg_hnsw->fstdistfunc_(query, global_mean.data(), alg_hnsw->dist_func_param_);
        d_mean = std::max(1e-6f, d_mean); // Prevent division by zero

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

        // 3. Perform a bounded search on Layer 0 to track revisited nodes (Hubness/Entrapment).
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
            // Stop probing if we exceed the probe cap and are further than the worst candidate
            if (current_node_pair.first > lowerBound && top_candidates.size() == ef_probe_cap) break;
            candidate_set.pop();
            hnswlib::tableint cur_node = current_node_pair.second;

            auto* data = reinterpret_cast<int*>(alg_hnsw->get_linklist0(cur_node));
            int sz = alg_hnsw->getListCount(reinterpret_cast<hnswlib::linklistsizeint*>(data));
            auto* nbrs = reinterpret_cast<hnswlib::tableint*>(data + 1);

            for (int j = 0; j < sz; j++) {
                hnswlib::tableint cand = nbrs[j];
                float d = alg_hnsw->fstdistfunc_(query, alg_hnsw->getDataByInternalId(cand), alg_hnsw->dist_func_param_);
                
                // Track the absolute closest distance found so far for the RC metric
                if (d < best_d) best_d = d;
                
                // Track whether this edge points to a node we've already evaluated
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

        // 4. Sort all traversed edges by distance to apply rank-based weighting
        std::sort(edges.begin(), edges.end(), [](const EdgeEval& a, const EdgeEval& b) {
            return a.dist < b.dist;
        });

        // 5. Calculate Topological Rank Entrapment (RV_rank) using exponential decay
        int N = edges.size();
        float sum_tot = 0, sum_vis = 0;
        for (int i = 0; i < N; i++) {
            float rank_ratio = (float)(i + 1) / N; // Normalized rank in (0, 1]
            float w = std::exp(-gamma * rank_ratio); // Heavy weight to close nodes
            sum_tot += w;
            if (edges[i].is_revisit) sum_vis += w;
        }

        EstimatorResult res;
        // RC = Global Distance / Local Distance. High contrast = distinct cluster = easy.
        res.RC = d_mean / std::max(best_d, 1e-6f);
        
        // RV = Weighted Revisit Ratio. High revisit = trapped in a hub = hard.
        res.RV_rank = sum_vis / std::max(1e-6f, sum_tot);
        
        return res;
    }
};

} // namespace hnsw_2metric
