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

#include "../experiments_driver/util.h"
#include "../hnswlib/hnswlib.h"

using namespace hnswlib;

const std::vector<float> GAMMAS = {5.0f, 10.0f, 20.0f, 50.0f, 100.0f};

struct ProbeRes {
    std::vector<float> rv_norm_gauss;
};

ProbeRes probe_query_best(HierarchicalNSW<float>* alg_hnsw, const float* query, int L, int ef_probe_cap) {
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
    
    std::vector<float> sum_tot(GAMMAS.size(), 0.0f);
    std::vector<float> sum_vis(GAMMAS.size(), 0.0f);

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
            if (d < best_d) best_d = d;
            
            bool is_revisit = (visited.find(cand) != visited.end());
            
            for (size_t g = 0; g < GAMMAS.size(); g++) {
                // Scale-invariant Gaussian
                float ratio = (best_d > 0.0f) ? (d / best_d) - 1.0f : 0.0f;
                // For L2 squared, distance ratio is squared. To normalize linearly we could use sqrt(d)/sqrt(best_d), 
                // but let's test the raw ratio first.
                float w = std::exp(-GAMMAS[g] * std::max(0.0f, ratio));
                
                sum_tot[g] += w;
                if (is_revisit) sum_vis[g] += w;
            }

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

    ProbeRes res;
    res.rv_norm_gauss.resize(GAMMAS.size(), 0.0f);
    for (size_t g = 0; g < GAMMAS.size(); g++) {
        res.rv_norm_gauss[g] = sum_vis[g] / std::max(1e-6f, sum_tot[g]);
    }
    return res;
}

std::pair<int, float> find_true_ef_for_query(HierarchicalNSW<float>* alg_hnsw, const float* query, const hnswdis::MatrixXi& ground_truth, int original_idx, float target_recall, int max_ef) {
    int best_ef = 50;
    float best_recall = 0.0f;
    for (int ef = 50; ef <= max_ef; ef += 50) {
        alg_hnsw->setEf(ef);
        auto sres = alg_hnsw->searchKnn(query, 10);
        int hits = 0;
        while (!sres.empty()) {
            tableint id = sres.top().second;
            sres.pop();
            for (int c = 0; c < 10; c++) if (ground_truth(original_idx, c) == (int)id) { hits++; break; }
        }
        float recall = hits / 10.0f;
        best_ef = ef;
        best_recall = recall;
        if (recall >= target_recall) break;
    }
    return {best_ef, best_recall};
}

int main(int argc, char** argv) {
    std::string dataset = "glove-100-angular";
    if (argc > 1) dataset = argv[1];

    const char* root_env = std::getenv("EXPERIMENTS_ROOT");
    std::filesystem::path root = root_env ? std::filesystem::path(root_env) : std::filesystem::current_path();
    std::string hdf5_path = (root / "data" / (dataset + ".hdf5")).string();
    if (!std::filesystem::exists(hdf5_path)) hdf5_path = (root / "experiments/data" / (dataset + ".hdf5")).string();

    hnswdis::MatrixXf full_data, query_vectors;
    hnswdis::MatrixXi ground_truth;
    load_hdf5(hdf5_path, query_vectors, full_data, ground_truth);
    
    if (dataset.find("angular") != std::string::npos) {
        normalize_matrix(full_data); 
        normalize_matrix(query_vectors);
    }

    std::string index_path = (root / "index" / (dataset + "-M16-efc-500-parallel.hnsw")).string();
    if (!std::filesystem::exists(index_path)) {
        index_path = "/home/ryawszn/experiments/index/" + dataset + "-M16-efc-500-parallel.hnsw";
    }

    SpaceInterface<float>* space;
    if (dataset.find("euclidean") != std::string::npos) {
        space = new L2Space(full_data.cols());
    } else {
        space = new InnerProductSpace(full_data.cols());
    }
    
    auto* alg_hnsw = new HierarchicalNSW<float>(space, index_path, false, full_data.rows());

    int nq = 250;
    
    std::vector<int> q_idx(query_vectors.rows());
    std::iota(q_idx.begin(), q_idx.end(), 0);
    std::srand(42); std::random_shuffle(q_idx.begin(), q_idx.end());
    q_idx.resize(nq);

    std::vector<ProbeRes> pr(nq);
    std::vector<int> ef_true(nq);

    alg_hnsw->setEf(100); 
    #pragma omp parallel for
    for (int i = 0; i < nq; i++) {
        Eigen::RowVectorXf q = query_vectors.row(q_idx[i]);
        pr[i] = probe_query_best(alg_hnsw, q.data(), alg_hnsw->maxlevel_, 100);
    }

    for (int i = 0; i < nq; i++) {
        Eigen::RowVectorXf q = query_vectors.row(q_idx[i]);
        ef_true[i] = find_true_ef_for_query(alg_hnsw, q.data(), ground_truth, q_idx[i], 0.95f, 2000).first;
    }

    std::string out_path = "research/best_rv_sweep_" + dataset + ".csv";
    std::ofstream out(out_path);
    out << "query_idx,ef_true";
    for (float g : GAMMAS) out << ",RV_normgauss_" << std::fixed << std::setprecision(1) << g;
    out << "\n";

    for (int i = 0; i < nq; i++) {
        out << q_idx[i] << "," << ef_true[i];
        for (float rv : pr[i].rv_norm_gauss) out << "," << rv;
        out << "\n";
    }
    out.close();

    delete alg_hnsw;
    delete space;
    return 0;
}
