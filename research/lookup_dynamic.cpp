#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <queue>
#include <unordered_set>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstdlib>
#include <string>

#include "../experiments_driver/util.h"
#include "../hnswlib/hnswlib.h"

using namespace hnswlib;

// Simple argument parser
std::string get_cmd_option(char ** begin, char ** end, const std::string & option, const std::string & default_value = "") {
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return default_value;
}

bool cmd_option_exists(char** begin, char** end, const std::string& option) {
    return std::find(begin, end, option) != end;
}

int main(int argc, char** argv) {
    if (cmd_option_exists(argv, argv + argc, "-h") || cmd_option_exists(argv, argv + argc, "--help")) {
        std::cerr << "Usage: " << argv[0] << " [options]\n"
                  << "Options:\n"
                  << "  --dataset <name>        Dataset name (default: glove-100-angular)\n"
                  << "  --target_recall <float> Target recall (default: 0.95)\n"
                  << "  --max_ef <int>          Maximum EF allowed (default: 5000)\n"
                  << "  --RC_bins <int>         Number of RC bins (default: 20)\n"
                  << "  --RV_bins <int>         Number of RV bins (default: 20)\n"
                  << "  --sample_size <int>     Number of queries to sample (default: 2000)\n";
        return 0;
    }

    std::string dataset = get_cmd_option(argv, argv + argc, "--dataset", "glove-100-angular");
    float target_recall = std::stof(get_cmd_option(argv, argv + argc, "--target_recall", "0.95"));
    int max_ef = std::stoi(get_cmd_option(argv, argv + argc, "--max_ef", "5000"));
    int RC_BINS = std::stoi(get_cmd_option(argv, argv + argc, "--RC_bins", "20"));
    int RV_BINS = std::stoi(get_cmd_option(argv, argv + argc, "--RV_bins", "20"));
    int sample_size = std::stoi(get_cmd_option(argv, argv + argc, "--sample_size", "2000"));
    
    int ef_probe = 50; 
    float gamma = 15.0f; // Universally optimal gamma for Rank RV

    std::cout << "=== Generating " << RC_BINS << "x" << RV_BINS << " Adaptive EF Lookup Table (RC x Rank RV) ===" << std::endl;
    std::cout << "Target Recall: " << target_recall << std::endl;
    std::cout << "Max 'ef' Cap:  " << max_ef << std::endl;
    if (sample_size > 0) {
        std::cout << "Sample Size:   " << sample_size << " queries" << std::endl;
    } else {
        std::cout << "Sample Size:   All queries" << std::endl;
    }
    std::cout << "Matrix Size:   " << RC_BINS << " (M) x " << RV_BINS << " (RV)" << std::endl;
    std::cout << "Dataset:       " << dataset << std::endl;
    std::cout << "----------------------------------------------------------------\n" << std::endl;

    const char *experiments_root = std::getenv("EXPERIMENTS_ROOT");
    std::filesystem::path root = experiments_root ? std::filesystem::path(experiments_root)
                                                  : std::filesystem::current_path();
    std::string hdf5_path = (root / "data" / (dataset + ".hdf5")).string();
    if (!std::filesystem::exists(hdf5_path)) {
        hdf5_path = "/home/ryawszn/experiments/data/" + dataset + ".hdf5";
    }

    std::cout << "Loading dataset: " << hdf5_path << std::endl;
    hnswdis::MatrixXf full_data;
    hnswdis::MatrixXf query_vectors;
    hnswdis::MatrixXi ground_truth;

    load_hdf5(hdf5_path, query_vectors, full_data, ground_truth);
    
    // Normalize if dataset is cosine/angular
    if (dataset.find("angular") != std::string::npos) {
        normalize_matrix(full_data);
        normalize_matrix(query_vectors);
    }

    std::string index_path = (root / "index" / (dataset + "-M16-efc-500-parallel.hnsw")).string();
    if (!std::filesystem::exists(index_path)) {
        index_path = "/home/ryawszn/experiments/index/" + dataset + "-M16-efc-500-parallel.hnsw";
    }

    std::cout << "Loading HNSW index from " << index_path << "..." << std::endl;
    SpaceInterface<float>* space;
    if (dataset.find("euclidean") != std::string::npos) {
        space = new L2Space(full_data.cols());
    } else {
        space = new InnerProductSpace(full_data.cols());
    }
    HierarchicalNSW<float>* alg_hnsw = new HierarchicalNSW<float>(space, index_path, false, full_data.rows());

    int total_queries = query_vectors.rows();
    int nq = total_queries;
    std::vector<int> query_indices(total_queries);
    std::iota(query_indices.begin(), query_indices.end(), 0);
    
    if (sample_size > 0 && sample_size < total_queries) {
        std::srand(42); 
        std::random_shuffle(query_indices.begin(), query_indices.end());
        query_indices.resize(sample_size);
        nq = sample_size;
        std::cout << "Randomly sampled " << nq << " queries out of " << total_queries << "." << std::endl;
    }

    int L = alg_hnsw->maxlevel_;
    
    std::vector<float> RC_scores(nq, 0.0f);
    std::vector<float> RV_scores(nq, 0.0f);

    std::cout << "Computing global centroid independently..." << std::endl;
    Eigen::RowVectorXf global_mean = full_data.colwise().mean();

    std::cout << "\nPhase 1: Probing all queries to calculate RC (Contrast) and RV (Rank Topology)..." << std::endl;
    alg_hnsw->setEf(ef_probe); 

    #pragma omp parallel for
    for (int i = 0; i < nq; i++) {
        int original_idx = query_indices[i];
        Eigen::RowVectorXf q = query_vectors.row(original_idx);
        const float* query = q.data();
        
        float d_mean = alg_hnsw->fstdistfunc_(query, global_mean.data(), alg_hnsw->dist_func_param_);
        d_mean = std::max(1e-6f, d_mean);

        tableint currObj = alg_hnsw->enterpoint_node_;
        float curdist = alg_hnsw->fstdistfunc_(query, alg_hnsw->getDataByInternalId(currObj), alg_hnsw->dist_func_param_);

        for (int level = L; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                auto* data = reinterpret_cast<unsigned int*>(alg_hnsw->get_linklist(currObj, level));
                int sz = alg_hnsw->getListCount(reinterpret_cast<hnswlib::linklistsizeint*>(data));
                auto* nbrs = reinterpret_cast<tableint*>(data + 1);

                for (int j = 0; j < sz; j++) {
                    tableint cand = nbrs[j];
                    float d = alg_hnsw->fstdistfunc_(query, alg_hnsw->getDataByInternalId(cand), alg_hnsw->dist_func_param_);
                    if (d < curdist) { curdist = d; currObj = cand; changed = true; }
                }
            }
        }

        std::priority_queue<std::pair<float, tableint>> top_candidates;
        std::priority_queue<std::pair<float, tableint>, std::vector<std::pair<float, tableint>>, std::greater<std::pair<float, tableint>>> candidate_set;
        std::unordered_set<tableint> visited;

        visited.insert(currObj);
        candidate_set.emplace(curdist, currObj);
        top_candidates.emplace(curdist, currObj);

        float lowerBound = curdist;
        float best_d = curdist;
        std::vector<std::pair<float, bool>> evaluated_edges;

        while (!candidate_set.empty()) {
            auto current_node_pair = candidate_set.top();
            if (current_node_pair.first > lowerBound && top_candidates.size() == ef_probe) break;
            candidate_set.pop();
            tableint cur_node = current_node_pair.second;

            auto* data = reinterpret_cast<int*>(alg_hnsw->get_linklist0(cur_node));
            int sz = alg_hnsw->getListCount(reinterpret_cast<hnswlib::linklistsizeint*>(data));
            auto* nbrs = reinterpret_cast<tableint*>(data + 1);

            for (int j = 0; j < sz; j++) {
                tableint cand = nbrs[j];
                float d = alg_hnsw->fstdistfunc_(query, alg_hnsw->getDataByInternalId(cand), alg_hnsw->dist_func_param_);
                if (d < best_d) best_d = d;

                bool is_revisit = (visited.find(cand) != visited.end());
                evaluated_edges.push_back({d, is_revisit});

                if (!is_revisit) {
                    visited.insert(cand);
                    if (top_candidates.size() < ef_probe || lowerBound > d) {
                        candidate_set.emplace(d, cand);
                        top_candidates.emplace(d, cand);
                        if (top_candidates.size() > ef_probe) top_candidates.pop();
                        lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        RC_scores[i] = d_mean / std::max(best_d, 1e-6f);

        std::sort(evaluated_edges.begin(), evaluated_edges.end(), 
            [](const std::pair<float, bool>& a, const std::pair<float, bool>& b) { return a.first < b.first; });

        int num_edges = evaluated_edges.size();
        float sum_r_tot = 0, sum_r_vis = 0;
        for (int e = 0; e < num_edges; e++) {
            float rank_ratio = (float)(e + 1) / num_edges;
            float w = std::exp(-gamma * rank_ratio);
            sum_r_tot += w;
            if (evaluated_edges[e].second) sum_r_vis += w;
        }
        
        RV_scores[i] = sum_r_vis / std::max(1e-6f, sum_r_tot);
    }

    std::cout << "Phase 2: Calculating Quantile Boundaries for " << RC_BINS << "x" << RV_BINS << " grid..." << std::endl;
    std::vector<float> sorted_M = RC_scores;
    std::vector<float> sorted_RV = RV_scores;
    std::sort(sorted_M.begin(), sorted_M.end());
    std::sort(sorted_RV.begin(), sorted_RV.end());

    std::vector<float> RC_bounds(RC_BINS + 1);
    std::vector<float> RV_bounds(RV_BINS + 1);
    for(int i=0; i<=RC_BINS; i++) {
        int idx = std::min((int)(nq - 1), (int)(i * nq / (float)RC_BINS));
        RC_bounds[i] = sorted_M[idx];
    }
    for(int j=0; j<=RV_BINS; j++) {
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

    std::cout << "Phase 3: Stepping ef by +50 to hit Target Recall (" << target_recall << ") for each bucket...\n" << std::endl;

    std::vector<std::vector<std::pair<int, float>>> lookup_table(RC_BINS, std::vector<std::pair<int, float>>(RV_BINS, {0, 0.0f}));

    int col_width = 13;
    std::string separator(6 + RV_BINS * (col_width + 1), '-');
    
    std::cout << separator << std::endl;
    std::cout << "             Adaptive EF Lookup Table (" << RC_BINS << "x" << RV_BINS << ")" << std::endl;
    std::cout << separator << std::endl;
    
    std::cout << "     |";
    for(int j=0; j<RV_BINS; j++) std::cout << "     RV" << std::setw(2) << std::left << (j+1) << "   |";
    std::cout << "\n" << separator << std::endl;

    char filename[512];
    snprintf(filename, sizeof(filename), "/home/ryawszn/experiments/2metric/lookup/lookup_table_%s_%dx%d.csv", dataset.c_str(), RC_BINS, RV_BINS);
    std::ofstream out(filename);
    out << "RC_bin,RV_bin,ef,actual_recall\n";

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

            for (int ef = 50; ef <= max_ef; ef += 50) {
                alg_hnsw->setEf(ef);
                int total_hits = 0;

                for (int q_idx : query_list) {
                    const float* query = query_vectors.row(q_idx).data();
                    auto pq = alg_hnsw->searchKnn(query, ground_truth.cols());
                    
                    std::vector<size_t> res(pq.size());
                    int count = pq.size();
                    while (!pq.empty()) {
                        res[--count] = pq.top().second;
                        pq.pop();
                    }

                    int hits = 0;
                    for (size_t r : res) {
                        for (int k = 0; k < ground_truth.cols(); k++) {
                            if (r == ground_truth(q_idx, k)) { hits++; break; }
                        }
                    }
                    total_hits += hits;
                }

                float avg_recall = (float)total_hits / (query_list.size() * ground_truth.cols());
                if (avg_recall > best_recall) {
                    best_recall = avg_recall;
                    best_ef = ef;
                }

                if (avg_recall >= target_recall) break; 
            }

            lookup_table[i][j] = {best_ef, best_recall};
            out << "\"(" << RC_bounds[i] << "," << RC_bounds[i+1] << "]\",\"(" 
                << RV_bounds[j] << "," << RV_bounds[j+1] << "]\",\"" 
                << best_ef << "\",\"" << best_recall << "\"\n";
            out.flush();
            
            char buf[32];
            snprintf(buf, sizeof(buf), "%4d,%.4f", best_ef, best_recall);
            std::cout << " " << std::setw(col_width-1) << std::left << buf << "|";
        }
        std::cout << "\n";
    }
    std::cout << separator << "\n" << std::endl;

    out.close();
    std::cout << "Lookup table saved to " << filename << std::endl;

    delete alg_hnsw;
    delete space;
    return 0;
}
