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
#include <limits>

#include "../experiments_driver/util.h"
#include "../hnswlib/hnswlib.h"

using namespace hnswlib;

struct ProbeResult {
    float M = 0.0f;
    float m = 0.0f;
    float RV = 0.0f; // Gaussian-Weighted Revisit Ratio
    std::vector<float> base_dists; 
};

ProbeResult probe_query(HierarchicalNSW<float>* alg_hnsw,
                         const float* query,
                         const Eigen::RowVectorXf& global_mean,
                         int L,
                         int K_probe,
                         int ef_probe_cap) {
    ProbeResult res;

    Eigen::Map<const Eigen::RowVectorXf> q(query, global_mean.size());
    float th_mean = q.dot(global_mean);
    float d_mean = std::max(0.01f, 1.0f - th_mean);

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
                if (d < curdist) {
                    curdist = d;
                    currObj = cand;
                    changed = true;
                }
            }
        }
    }

    std::priority_queue<std::pair<float, tableint>> top_candidates;
    std::priority_queue<std::pair<float, tableint>,
                         std::vector<std::pair<float, tableint>>,
                         std::greater<std::pair<float, tableint>>> candidate_set;
    std::unordered_set<tableint> visited;

    visited.insert(currObj);
    candidate_set.emplace(curdist, currObj);
    top_candidates.emplace(curdist, currObj);

    std::vector<float> base_dists;
    base_dists.push_back(curdist);
    float lowerBound = curdist;

    float sum_gau_tot = 0.0f;
    float sum_gau_vis = 0.0f;

    while (!candidate_set.empty()) {
        auto current_node_pair = candidate_set.top();
        if (current_node_pair.first > lowerBound && top_candidates.size() == ef_probe_cap) break;
        candidate_set.pop();
        tableint cur_node = current_node_pair.second;

        auto* data = reinterpret_cast<int*>(alg_hnsw->get_linklist0(cur_node));
        int sz = alg_hnsw->getListCount(reinterpret_cast<hnswlib::linklistsizeint*>(data));
        auto* nbrs = reinterpret_cast<tableint*>(data + 1);

        for (int j = 0; j < sz; j++) {
            tableint cand = nbrs[j];
            float d = alg_hnsw->fstdistfunc_(query, alg_hnsw->getDataByInternalId(cand), alg_hnsw->dist_func_param_);
            
            // Gaussian Weighting (gamma = 2.0)
            float w_gau = std::exp(-2.0f * d); 
            
            sum_gau_tot += w_gau;
            
            if (visited.find(cand) != visited.end()) {
                sum_gau_vis += w_gau;
            } else {
                visited.insert(cand);
                base_dists.push_back(d);

                if (top_candidates.size() < ef_probe_cap || lowerBound > d) {
                    candidate_set.emplace(d, cand);
                    top_candidates.emplace(d, cand);
                    if (top_candidates.size() > ef_probe_cap) {
                        top_candidates.pop();
                    }
                    lowerBound = top_candidates.top().first;
                }
            }
        }
    }

    std::sort(base_dists.begin(), base_dists.end());
    float b_0 = base_dists[0];

    // 1. M (Macro / Relative Contrast)
    res.M = b_0 / d_mean;
    
    // 2. RV (Gaussian-Weighted Revisit Ratio)
    res.RV = sum_gau_vis / std::max(1e-6f, sum_gau_tot);

    // 3. m (Micro / LID Hill Estimator)
    int K = std::min(K_probe, (int)base_dists.size());
    float m_q = 0.0f;
    if (K > 1) {
        float d_K = base_dists[K - 1] + 1e-6f;
        float sum_log = 0.0f;
        for (int k = 0; k < K - 1; k++) {
            float d_i = base_dists[k] + 1e-6f;
            sum_log += std::log(d_K / d_i);
        }
        if (sum_log > 0) m_q = (K - 1) / sum_log;
    }
    res.m = m_q;

    return res;
}

std::pair<int, float> find_true_ef_for_query(HierarchicalNSW<float>* alg_hnsw,
                                              const float* query,
                                              const hnswdis::MatrixXi& ground_truth,
                                              int original_idx,
                                              float target_recall,
                                              int max_ef,
                                              int k_results = 10) {
    int best_ef = 50;
    float best_recall = 0.0f;

    for (int ef = 50; ef <= max_ef; ef += 50) {
        alg_hnsw->setEf(ef);
        auto sres = alg_hnsw->searchKnn(query, k_results);

        int hits = 0;
        while (!sres.empty()) {
            tableint id = sres.top().second;
            sres.pop();
            for (int c = 0; c < k_results; c++) {
                if (ground_truth(original_idx, c) == (int)id) {
                    hits++;
                    break;
                }
            }
        }
        float recall = (float)hits / k_results;
        best_ef = ef;
        best_recall = recall;
        if (recall >= target_recall) break;
    }
    return {best_ef, best_recall};
}

int main(int argc, char** argv) {
    float target_recall = 0.95f;
    std::string dataset = "glove-100-angular";
    int max_ef = 5000;  
    int sample_size = 2000; 
    int K_probe = 32;
    int ef_probe = 100; 

    const char* experiments_root = std::getenv("EXPERIMENTS_ROOT");
    std::filesystem::path root = experiments_root ? std::filesystem::path(experiments_root)
                                                   : std::filesystem::current_path();
    std::string hdf5_path = (root / "data" / (dataset + ".hdf5")).string();
    if (!std::filesystem::exists(hdf5_path)) {
        hdf5_path = (root / "experiments/data" / (dataset + ".hdf5")).string();
    }

    hnswdis::MatrixXf full_data;
    hnswdis::MatrixXf query_vectors;
    hnswdis::MatrixXi ground_truth;
    load_hdf5(hdf5_path, query_vectors, full_data, ground_truth);
    normalize_matrix(full_data);
    normalize_matrix(query_vectors);

    std::string index_path = (root / "index" / (dataset + "-M16-efc-500-parallel.hnsw")).string();
    L2Space space(full_data.cols());
    HierarchicalNSW<float>* alg_hnsw = new HierarchicalNSW<float>(&space, index_path, false, full_data.rows());

    int total_queries = query_vectors.rows();
    int nq = sample_size;
    std::vector<int> query_indices(total_queries);
    std::iota(query_indices.begin(), query_indices.end(), 0);
    std::srand(42);
    std::random_shuffle(query_indices.begin(), query_indices.end());
    query_indices.resize(nq);

    int L = alg_hnsw->maxlevel_;
    Eigen::RowVectorXf global_mean = full_data.colwise().mean();

    std::vector<ProbeResult> pr_scores(nq);
    std::vector<int> ef_true(nq);
    std::vector<float> recall_true(nq);

    alg_hnsw->setEf(ef_probe); 
    #pragma omp parallel for
    for (int i = 0; i < nq; i++) {
        int original_idx = query_indices[i];
        Eigen::RowVectorXf q = query_vectors.row(original_idx);
        pr_scores[i] = probe_query(alg_hnsw, q.data(), global_mean, L, K_probe, ef_probe);
    }

    for (int i = 0; i < nq; i++) {
        int original_idx = query_indices[i];
        Eigen::RowVectorXf q = query_vectors.row(original_idx);
        auto [ef_star, r_star] = find_true_ef_for_query(alg_hnsw, q.data(), ground_truth, original_idx, target_recall, max_ef);
        ef_true[i] = ef_star;
        recall_true[i] = r_star;
    }

    std::string out_path = "research/hardness_diagnostic_" + dataset + ".csv";
    std::ofstream out(out_path);
    out << "query_idx,M,m,RV,ef_true,recall_at_ef_true\n";
    for (int i = 0; i < nq; i++) {
        out << query_indices[i] << ","
            << pr_scores[i].M << ","
            << pr_scores[i].m << ","
            << pr_scores[i].RV << ","
            << ef_true[i] << ","
            << recall_true[i] << "\n";
    }
    out.close();

    delete alg_hnsw;
    return 0;
}
