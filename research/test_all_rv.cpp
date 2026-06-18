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

struct EdgeEval {
    float dist;
    bool is_revisit;
    tableint id;
};

struct ProbeRes {
    float rv_gauss = 0.0f;
    float rv_rank = 0.0f;
    float rv_hub = 0.0f;
};

ProbeRes probe_query_all(HierarchicalNSW<float>* alg_hnsw, const float* query, int L, int ef_probe_cap, float opt_gamma, const std::vector<int>& in_degree) {
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
    std::vector<EdgeEval> edges;

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
            bool is_revisit = (visited.find(cand) != visited.end());
            
            edges.push_back({d, is_revisit, cand});

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

    // 1 & 3: Gauss and Hub
    float sum_g_tot = 0, sum_g_vis = 0;
    float sum_h_tot = 0, sum_h_vis = 0;
    for (auto& e : edges) {
        float w_g = std::exp(-opt_gamma * e.dist);
        float hc = std::max(1.0f, (float)in_degree[e.id]);
        float w_h = w_g / std::pow(hc, 0.5f);
        
        sum_g_tot += w_g;
        sum_h_tot += w_h;
        if (e.is_revisit) {
            sum_g_vis += w_g;
            sum_h_vis += w_h;
        }
    }

    // 2. Rank
    std::sort(edges.begin(), edges.end(), [](const EdgeEval& a, const EdgeEval& b) { return a.dist < b.dist; });
    int N = edges.size();
    float sum_r_tot = 0, sum_r_vis = 0;
    for (int i = 0; i < N; i++) {
        float w_r = std::exp(-10.0f * ((float)(i + 1) / N));
        sum_r_tot += w_r;
        if (edges[i].is_revisit) sum_r_vis += w_r;
    }

    ProbeRes res;
    res.rv_gauss = sum_g_vis / std::max(1e-6f, sum_g_tot);
    res.rv_rank = sum_r_vis / std::max(1e-6f, sum_r_tot);
    res.rv_hub = sum_h_vis / std::max(1e-6f, sum_h_tot);
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
    float opt_gamma = 5.0f;
    if (dataset.find("euclidean") != std::string::npos) {
        space = new L2Space(full_data.cols());
        opt_gamma = 0.001f; // Euclidean needs tiny gamma to prevent underflow
    } else {
        space = new InnerProductSpace(full_data.cols());
        if (dataset.find("deep") != std::string::npos) opt_gamma = 0.1f;
    }
    
    auto* alg_hnsw = new HierarchicalNSW<float>(space, index_path, false, full_data.rows());

    std::vector<int> in_degree(alg_hnsw->max_elements_, 0);
    for (tableint i = 0; i < alg_hnsw->cur_element_count; i++) {
        auto* data = reinterpret_cast<int*>(alg_hnsw->get_linklist0(i));
        int sz = alg_hnsw->getListCount(reinterpret_cast<hnswlib::linklistsizeint*>(data));
        auto* nbrs = reinterpret_cast<tableint*>(data + 1);
        for (int j = 0; j < sz; j++) in_degree[nbrs[j]]++;
    }

    int nq = 100;
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
        pr[i] = probe_query_all(alg_hnsw, q.data(), alg_hnsw->maxlevel_, 100, opt_gamma, in_degree);
    }

    for (int i = 0; i < nq; i++) {
        Eigen::RowVectorXf q = query_vectors.row(q_idx[i]);
        ef_true[i] = find_true_ef_for_query(alg_hnsw, q.data(), ground_truth, q_idx[i], 0.95f, 2000).first;
    }

    std::string out_path = "research/final_rv_" + dataset + ".csv";
    std::ofstream out(out_path);
    out << "query_idx,ef_true,RV_gauss,RV_rank,RV_hub\n";

    for (int i = 0; i < nq; i++) {
        out << q_idx[i] << "," << ef_true[i] << "," 
            << pr[i].rv_gauss << "," 
            << pr[i].rv_rank << "," 
            << pr[i].rv_hub << "\n";
    }
    out.close();

    delete alg_hnsw;
    delete space;
    return 0;
}
