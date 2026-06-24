#pragma once
#include "../hnswlib/hnswlib.h"
#include <queue>
#include <vector>
#include <cmath>
#include <algorithm>

namespace hnswlib {

struct Estimator2Metric {
    static std::pair<float, float> probe_query(
        const HierarchicalNSW<float>* alg_hnsw,
        const void* query_data,
        tableint v_ep,
        size_t ef_cap,
        float gamma = 16.0f)
    {
        float d_ep = alg_hnsw->fstdistfunc_(query_data, alg_hnsw->getDataByInternalId(v_ep), alg_hnsw->dist_func_param_);

        std::vector<bool> visited(alg_hnsw->cur_element_count, false);

        // Cand = MinQueue, Top = MaxQueue
        std::priority_queue<std::pair<float, tableint>, std::vector<std::pair<float, tableint>>, std::greater<std::pair<float, tableint>>> Cand;
        std::priority_queue<std::pair<float, tableint>> Top;

        visited[v_ep] = true;
        Cand.emplace(d_ep, v_ep);
        Top.emplace(d_ep, v_ep);

        struct Edge {
            float dist;
            bool is_revisit;
            bool operator<(const Edge& other) const {
                return dist < other.dist;
            }
        };
        std::vector<Edge> Edges;

        while (!Cand.empty()) {
            auto c = Cand.top();
            Cand.pop();

            if (c.first > Top.top().first && Top.size() == ef_cap) {
                break;
            }

            // For n in N(c)
            tableint* data = (tableint*)alg_hnsw->get_linklist_at_level(c.second, 0);
            size_t size = *data;
            tableint* links = data + 1;

            for (size_t i = 0; i < size; i++) {
                tableint n = links[i];
                float d_n = alg_hnsw->fstdistfunc_(query_data, alg_hnsw->getDataByInternalId(n), alg_hnsw->dist_func_param_);
                bool is_revisit = visited[n];

                Edges.push_back({d_n, is_revisit});

                if (!is_revisit) {
                    visited[n] = true;
                    if (Top.size() < ef_cap || d_n < Top.top().first) {
                        Cand.emplace(d_n, n);
                        Top.emplace(d_n, n);
                        if (Top.size() > ef_cap) {
                            Top.pop();
                        }
                    }
                }
            }
        }

        std::sort(Edges.begin(), Edges.end());

        float W = 0.0f;
        float W_rev = 0.0f;
        float N = Edges.size();

        for (size_t i = 0; i < N; i++) {
            float w = std::exp(-gamma * (i + 1) / N);
            W += w;
            if (Edges[i].is_revisit) {
                W_rev += w;
            }
        }

        float epsilon = 1e-5f;
        float R_v = W_rev / std::max(epsilon, W);

        return {d_ep, R_v};
    }
};

} // namespace hnswlib
