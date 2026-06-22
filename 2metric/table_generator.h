#pragma once

#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <string>

#include "estimator.h"
#include "lookuptable.h"

namespace hnsw_2metric {

class TableGenerator2Metric {
public:
    static LookupTable2D generate(
        hnswlib::HierarchicalNSW<float>* alg_hnsw,
        const hnswdis::MatrixXf& full_query_vectors,
        const hnswdis::MatrixXi& full_ground_truth,
        const Eigen::RowVectorXf& global_mean,
        float target_recall = 0.95f,
        int k_val = 32,
        int entry_point_bins = 32,
        int revisit_bins = 32,
        int max_ef = 5000,
        int sample_size = 5000,
        const std::string& save_csv_path = ""
    ) {
        std::cout << "=== Generating " << entry_point_bins << "x" << revisit_bins << " Dynamic Ada-EF Curve Table ===" << std::endl;

        int total_queries = full_query_vectors.rows();
        int nq = total_queries;
        std::vector<int> query_indices(total_queries);
        std::iota(query_indices.begin(), query_indices.end(), 0);

        if (sample_size > 0 && sample_size < total_queries) {
            std::srand(42);
            std::random_shuffle(query_indices.begin(), query_indices.end());
            query_indices.resize(sample_size);
            nq = sample_size;
            std::cout << "Randomly sampled " << nq << " queries for table generation." << std::endl;
        }

        std::vector<float> ep_scores(nq, 0.0f);
        std::vector<float> rv_scores(nq, 0.0f);

        std::cout << "Phase 1: Probing queries to calculate EP and RV..." << std::endl;
        #pragma omp parallel for
        for (int i = 0; i < nq; i++) {
            int q_idx = query_indices[i];
            Eigen::RowVectorXf q = full_query_vectors.row(q_idx);
            auto est = Estimator2Metric::probe_query(alg_hnsw, q.data(), global_mean, 50, 15.0f);
            ep_scores[i] = est.entry_point_dist;
            rv_scores[i] = est.revisit_rank;
        }

        std::cout << "Phase 2: Calculating Quantile Boundaries..." << std::endl;
        std::vector<float> sorted_EP = ep_scores;
        std::vector<float> sorted_RV = rv_scores;
        std::sort(sorted_EP.begin(), sorted_EP.end());
        std::sort(sorted_RV.begin(), sorted_RV.end());

        std::vector<float> ep_bounds(entry_point_bins + 1);
        std::vector<float> rv_bounds(revisit_bins + 1);
        for(int i = 0; i <= entry_point_bins; i++) {
            int idx = std::min((int)(nq - 1), (int)(i * nq / (float)entry_point_bins));
            ep_bounds[i] = sorted_EP[idx];
        }
        for(int j = 0; j <= revisit_bins; j++) {
            int idx = std::min((int)(nq - 1), (int)(j * nq / (float)revisit_bins));
            rv_bounds[j] = sorted_RV[idx];
        }

        std::vector<std::vector<std::vector<int>>> buckets(entry_point_bins, std::vector<std::vector<int>>(revisit_bins));
        for (int i = 0; i < nq; i++) {
            int bin_ep = 0;
            while(bin_ep < entry_point_bins - 1 && ep_scores[i] > ep_bounds[bin_ep+1]) bin_ep++;
            int bin_rv = 0;
            while(bin_rv < revisit_bins - 1 && rv_scores[i] > rv_bounds[bin_rv+1]) bin_rv++;
            buckets[bin_ep][bin_rv].push_back(query_indices[i]);
        }

        std::cout << "Phase 3: Measuring dynamic 2metric-EF per bin..." << std::endl;

        std::vector<std::vector<std::vector<std::pair<int, float>>>> bin_curves(
            entry_point_bins, std::vector<std::vector<std::pair<int, float>>>(revisit_bins)
        );


        float expected_recall = target_recall;

        auto calc_avg_recall = [&](const std::vector<int>& q_list, int ef) {
            alg_hnsw->setEf(ef);
            std::vector<float> local_recs(q_list.size(), 0.0f);

            #pragma omp parallel for
            for (size_t idx = 0; idx < q_list.size(); idx++) {
                int q_idx = q_list[idx];
                const float* query = full_query_vectors.row(q_idx).data();
                auto pq = alg_hnsw->searchKnn(query, k_val);
                int hits = 0;
                while (!pq.empty()) {
                    size_t r = pq.top().second;
                    pq.pop();
                    for (size_t col = 0; col < k_val; col++) {
                        if (r == full_ground_truth(q_idx, col)) { hits++; break; }
                    }
                }
                local_recs[idx] = (float)hits / k_val;
            }
            float sum = 0;
            for(float r : local_recs) sum += r;
            return sum / q_list.size();
        };

        for (int i = 0; i < entry_point_bins; i++) {
            for (int j = 0; j < revisit_bins; j++) {
                const auto& q_list = buckets[i][j];
                if (q_list.empty()) continue;

                std::vector<std::pair<int, float>> curve;

                // First seed
                int ef1 = k_val;
                float rec1 = calc_avg_recall(q_list, ef1);
                curve.push_back({ef1, rec1});

                if (rec1 >= expected_recall) {
                    bin_curves[i][j] = curve;
                    continue;
                }

                // Second seed
                int ef2 = std::max((int)(k_val * 1.5), ef1 + 1);
                if (ef2 > max_ef) ef2 = max_ef;
                float rec2 = calc_avg_recall(q_list, ef2);
                curve.push_back({ef2, rec2});

                int latest_ef = ef2;
                float latest_agg_recall = rec2;
                int prev_ef = ef1;
                float prev_agg_recall = rec1;
                int ef_diff = latest_ef - prev_ef;
                float epsilon = 1e-5f;
                float alpha = 0.1f; // 5% growth factor

                while (latest_agg_recall < expected_recall && latest_ef < max_ef) {
                    float recall_diff = latest_agg_recall - prev_agg_recall;

                    if (recall_diff < epsilon) break;

                    // Large geometric step to jump over micro-plateaus and bracket the target
                    ef_diff = std::max(
                        (int)(ef_diff * (expected_recall - latest_agg_recall) / recall_diff),
                        (int)std::max(k_val * 0.5f, latest_ef * alpha)
                    );

                    int next_ef = latest_ef + ef_diff;
                    if (next_ef > max_ef) next_ef = max_ef;

                    float agg_recall = calc_avg_recall(q_list, next_ef);
                    curve.push_back({next_ef, agg_recall});

                    float step_improvement = agg_recall - latest_agg_recall;
                    recall_diff = step_improvement;
                    latest_ef = next_ef;
                    latest_agg_recall = agg_recall;
                }

                bin_curves[i][j] = curve;
            }
        }

        std::vector<LookupBin> final_bins;
        std::ofstream out;
        if (!save_csv_path.empty()) {
            out.open(save_csv_path);
            if(out.is_open()) out << "entry_point_dist_bin,revisit_bin,query_count,curve\n";
        }

        for (int i = 0; i < entry_point_bins; i++) {
            for (int j = 0; j < revisit_bins; j++) {
                if (buckets[i][j].empty()) continue;

                LookupBin bin;
                bin.ep_lower_bound = ep_bounds[i]; bin.ep_upper_bound = ep_bounds[i+1];
                bin.rv_lower_bound = rv_bounds[j]; bin.rv_upper_bound = rv_bounds[j+1];
                bin.curve = bin_curves[i][j];
                bin.query_count = buckets[i][j].size();
                final_bins.push_back(bin);

                if (out.is_open()) {
                    out << "\"(" << bin.ep_lower_bound << "," << bin.ep_upper_bound << "]\",\"("
                        << bin.rv_lower_bound << "," << bin.rv_upper_bound << "]\",\"" << buckets[i][j].size() << "\",\"";

                    for (size_t c = 0; c < bin.curve.size(); c++) {
                        out << bin.curve[c].first << ":" << bin.curve[c].second;
                        if (c + 1 < bin.curve.size()) out << ",";
                    }
                    out << "\"\n";
                }
            }
        }

        if (out.is_open()) {
            out.close();
            std::cout << "Lookup table generated and saved to " << save_csv_path << "\n";
        }

        return LookupTable2D(final_bins, 50, target_recall);
    }
};

} // namespace hnsw_2metric
