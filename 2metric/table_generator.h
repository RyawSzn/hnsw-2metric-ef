#pragma once

#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <string>

#include "../experiments_driver/util.h"
#include "../hnswlib/hnswlib.h"
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
        int RC_BINS = 20,
        int RV_BINS = 20,
        int max_ef = 5000,
        int sample_size = 2000,
        const std::string& save_csv_path = ""
    ) {
        std::cout << "=== Generating " << RC_BINS << "x" << RV_BINS << " Adaptive EF Lookup Table (In-Memory) ===" << std::endl;
        
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

        std::vector<float> RC_scores(nq, 0.0f);
        std::vector<float> RV_scores(nq, 0.0f);

        std::cout << "Phase 1: Probing queries to calculate RC and RV..." << std::endl;
        #pragma omp parallel for
        for (int i = 0; i < nq; i++) {
            int q_idx = query_indices[i];
            Eigen::RowVectorXf q = full_query_vectors.row(q_idx);
            auto est = Estimator2Metric::probe_query(alg_hnsw, q.data(), global_mean, 50, 15.0f);
            RC_scores[i] = est.RC;
            RV_scores[i] = est.RV_rank;
        }

        std::cout << "Phase 1.5: Computing true EF for correlation diagnostic..." << std::endl;
        std::vector<int> true_ef_scores(nq, 50);
        size_t k_val = full_ground_truth.cols();
        for (int i = 0; i < nq; i++) {
            int q_idx = query_indices[i];
            const float* query = full_query_vectors.row(q_idx).data();
            for (int ef = 50; ef <= max_ef; ef += 50) {
                alg_hnsw->setEf(ef);
                auto sres = alg_hnsw->searchKnn(query, k_val);
                int hits = 0;
                while (!sres.empty()) {
                    hnswlib::tableint id = sres.top().second;
                    sres.pop();
                    for (size_t c = 0; c < k_val; c++) {
                        if (full_ground_truth(q_idx, c) == (int)id) { hits++; break; }
                    }
                }
                float recall = (float)hits / k_val;
                true_ef_scores[i] = ef;
                if (recall >= target_recall) break;
            }
        }

        if (!save_csv_path.empty()) {
            std::string diag_path = save_csv_path;
            size_t pos = diag_path.find("lookup_table_");
            if (pos != std::string::npos) {
                diag_path.replace(pos, 13, "diagnostic_");
            } else {
                diag_path = save_csv_path + ".diag.csv";
            }
            std::ofstream dout(diag_path);
            if (dout.is_open()) {
                dout << "query_idx,RC,RV_rank,ef_true\n";
                for (int i = 0; i < nq; i++) {
                    dout << query_indices[i] << "," << RC_scores[i] << "," << RV_scores[i] << "," << true_ef_scores[i] << "\n";
                }
                dout.close();
            }
        }

        std::cout << "Phase 2: Calculating Quantile Boundaries..." << std::endl;
        std::vector<float> sorted_M = RC_scores;
        std::vector<float> sorted_RV = RV_scores;
        std::sort(sorted_M.begin(), sorted_M.end());
        std::sort(sorted_RV.begin(), sorted_RV.end());

        std::vector<float> RC_bounds(RC_BINS + 1);
        std::vector<float> RV_bounds(RV_BINS + 1);
        for(int i = 0; i <= RC_BINS; i++) {
            int idx = std::min((int)(nq - 1), (int)(i * nq / (float)RC_BINS));
            RC_bounds[i] = sorted_M[idx];
        }
        for(int j = 0; j <= RV_BINS; j++) {
            int idx = std::min((int)(nq - 1), (int)(j * nq / (float)RV_BINS));
            RV_bounds[j] = sorted_RV[idx];
        }

        std::vector<std::vector<std::vector<int>>> buckets(RC_BINS, std::vector<std::vector<int>>(RV_BINS));
        for (int i = 0; i < nq; i++) {
            int bRC = 0;
            while(bRC < RC_BINS - 1 && RC_scores[i] > RC_bounds[bRC+1]) bRC++;
            int bRV = 0;
            while(bRV < RV_BINS - 1 && RV_scores[i] > RV_bounds[bRV+1]) bRV++;
            buckets[bRC][bRV].push_back(query_indices[i]);
        }

        std::cout << "Phase 3: Stepping ef by +50 to hit Target Recall (" << target_recall << ")..." << std::endl;
        
        std::vector<LookupBin> final_bins;
        std::ofstream out;
        if (!save_csv_path.empty()) {
            out.open(save_csv_path);
            if(out.is_open()) out << "RC_bin,RV_bin,ef,actual_recall\n";
        }

        int col_width = 13;
        std::string separator(6 + RV_BINS * (col_width + 1), '-');
        std::cout << separator << "\n     |";
        for(int j=0; j<RV_BINS; j++) std::cout << "     RV" << std::setw(2) << std::left << (j+1) << "   |";
        std::cout << "\n" << separator << std::endl;

        for (int i = 0; i < RC_BINS; i++) {
            std::cout << " RC" << std::setw(2) << std::left << (i+1) << " |";
            for (int j = 0; j < RV_BINS; j++) {
                auto& query_list = buckets[i][j];
                if (query_list.empty()) {
                    std::cout << "    ---      |";
                    continue;
                }

                int best_ef = 50;
                float best_recall = 0.0f;
                size_t k_val = full_ground_truth.cols();

                for (int ef = 50; ef <= max_ef; ef += 50) {
                    alg_hnsw->setEf(ef);
                    int total_hits = 0;

                    for (int q_idx : query_list) {
                        const float* query = full_query_vectors.row(q_idx).data();
                        auto pq = alg_hnsw->searchKnn(query, k_val);
                        
                        std::vector<size_t> res(pq.size());
                        int count = pq.size();
                        while (!pq.empty()) {
                            res[--count] = pq.top().second;
                            pq.pop();
                        }

                        int hits = 0;
                        for (size_t r : res) {
                            for (size_t col = 0; col < k_val; col++) {
                                if (r == full_ground_truth(q_idx, col)) { hits++; break; }
                            }
                        }
                        total_hits += hits;
                    }

                    float avg_recall = (float)total_hits / (query_list.size() * k_val);
                    if (avg_recall > best_recall) {
                        best_recall = avg_recall;
                        best_ef = ef;
                    }
                    if (avg_recall >= target_recall) break; 
                }

                LookupBin bin;
                bin.RC_lower = RC_bounds[i]; bin.RC_upper = RC_bounds[i+1];
                bin.RV_lower = RV_bounds[j]; bin.RV_upper = RV_bounds[j+1];
                bin.ef = best_ef;
                final_bins.push_back(bin);

                if (out.is_open()) {
                    out << "\"(" << bin.RC_lower << "," << bin.RC_upper << "]\",\"(" 
                        << bin.RV_lower << "," << bin.RV_upper << "]\",\"" 
                        << best_ef << "\",\"" << best_recall << "\"\n";
                    out.flush();
                }
                
                char buf[32];
                snprintf(buf, sizeof(buf), "%4d,%.4f", best_ef, best_recall);
                std::cout << " " << std::setw(col_width-1) << std::left << buf << "|";
            }
            std::cout << "\n";
        }
        std::cout << separator << "\n" << std::endl;
        
        if (out.is_open()) {
            out.close();
            std::cout << "Lookup table generated and saved to " << save_csv_path << "\n";
        }

        return LookupTable2D(final_bins, 100);
    }
};

} // namespace hnsw_2metric