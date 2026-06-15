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

int main(int argc, char** argv) {
    std::string dataset = "deep-image-96-angular";
    if (argc > 1) dataset = argv[1];

    std::cout << "=== Benchmarking Metric Computation Overhead ===" << std::endl;
    std::cout << "Dataset: " << dataset << std::endl;

    const char *experiments_root = std::getenv("EXPERIMENTS_ROOT");
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
    HierarchicalNSW<float> alg_hnsw(&space, index_path, false, full_data.rows());

    int nq = query_vectors.rows();
    int L = alg_hnsw.maxlevel_;

    Eigen::RowVectorXf global_mean = full_data.colwise().mean();

    std::vector<float> w(L + 1, 0.0f);
    float denom = 0.0f;
    for (int j = 0; j <= L; j++) denom += std::exp((float)j);
    for (int l = 0; l <= L; l++) w[l] = std::exp(-l + L + 1.0f) / denom;

    // Dummy arrays to prevent compiler from optimizing out the math
    std::vector<float> M_out(nq, 0.0f);
    std::vector<float> m_out(nq, 0.0f);

    std::cout << "\nRunning benchmark on " << nq << " queries (averaged over 3 runs)..." << std::endl;
    
    // Warmup
    alg_hnsw.setEf(50);

    // ---------------------------------------------------------
    // Method A: M_Graph + m_Spear
    // ---------------------------------------------------------
    long long time_A = 0;
    for(int run = 0; run < 3; run++) {
        auto start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for (int i = 0; i < nq; i++) {
            const float* query = query_vectors.row(i).data();
            tableint ep = alg_hnsw.enterpoint_node_;
            float curdist = alg_hnsw.fstdistfunc_(query, alg_hnsw.getDataByInternalId(ep), alg_hnsw.dist_func_param_);
            
            std::vector<int> n_evals(L + 1, 0);
            std::vector<float> b_best(L + 1, 0.0f);
            float p_b = curdist;
            
            // Track routing
            for (int level = L; level > 0; level--) {
                int evals = 0;
                bool changed = true;
                while (changed) {
                    changed = false;
                    auto* data = reinterpret_cast<unsigned int*>(alg_hnsw.get_linklist(ep, level));
                    int sz = alg_hnsw.getListCount(reinterpret_cast<hnswlib::linklistsizeint*>(data));
                    auto* nbrs = reinterpret_cast<tableint*>(data + 1);
                    for (int j = 0; j < sz; j++) {
                        float d = alg_hnsw.fstdistfunc_(query, alg_hnsw.getDataByInternalId(nbrs[j]), alg_hnsw.dist_func_param_);
                        evals++;
                        if (d < curdist) { curdist = d; ep = nbrs[j]; changed = true; }
                    }
                }
                n_evals[level] = evals;
                b_best[level] = curdist;
            }
            
            // ef=50 greedy
            std::priority_queue<std::pair<float, tableint>> top_c;
            std::priority_queue<std::pair<float, tableint>, std::vector<std::pair<float, tableint>>, std::greater<std::pair<float, tableint>>> cand_s;
            std::unordered_set<tableint> vis;
            vis.insert(ep); cand_s.emplace(curdist, ep); top_c.emplace(curdist, ep);
            std::vector<float> dists; dists.push_back(curdist); float lb = curdist;
            
            while (!cand_s.empty()) {
                auto pair = cand_s.top();
                if (pair.first > lb && top_c.size() == 50) break;
                cand_s.pop();
                auto* data = reinterpret_cast<int*>(alg_hnsw.get_linklist0(pair.second));
                int sz = alg_hnsw.getListCount(reinterpret_cast<hnswlib::linklistsizeint*>(data));
                auto* nbrs = reinterpret_cast<tableint*>(data + 1);
                for (int j = 0; j < sz; j++) {
                    if (vis.find(nbrs[j]) == vis.end()) {
                        vis.insert(nbrs[j]);
                        float d = alg_hnsw.fstdistfunc_(query, alg_hnsw.getDataByInternalId(nbrs[j]), alg_hnsw.dist_func_param_);
                        dists.push_back(d);
                        if (top_c.size() < 50 || lb > d) {
                            cand_s.emplace(d, nbrs[j]); top_c.emplace(d, nbrs[j]);
                            if (top_c.size() > 50) top_c.pop();
                            lb = top_c.top().first;
                        }
                    }
                }
            }
            
            float M = 0;
            for (int l = L; l >= 1; l--) {
                float rho = (p_b - b_best[l]) / (p_b + 1e-6f);
                M += w[l] * (std::log1p(n_evals[l]) / (1.0f + std::max(0.0f, rho)));
                p_b = b_best[l];
            }
            M_out[i] = M;
            
            std::sort(dists.begin(), dists.end());
            int K = std::min(32, (int)dists.size());
            float m = 0;
            if (K > 1) {
                float d_K = dists[K-1] + 1e-6f; float sum_log = 0;
                for (int k = 0; k < K - 1; k++) sum_log += std::log(d_K / (dists[k] + 1e-6f));
                if (sum_log > 0) m = (K - 1) / sum_log;
            }
            m_out[i] = m;
        }
        auto end = std::chrono::high_resolution_clock::now();
        time_A += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }

    // ---------------------------------------------------------
    // Method B: M_RC + m_Spear
    // ---------------------------------------------------------
    long long time_B = 0;
    for(int run = 0; run < 3; run++) {
        auto start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for (int i = 0; i < nq; i++) {
            Eigen::RowVectorXf q = query_vectors.row(i);
            const float* query = q.data();
            
            float d_mean = std::max(0.01f, 1.0f - (float)q.dot(global_mean));
            tableint ep = alg_hnsw.enterpoint_node_;
            float curdist = alg_hnsw.fstdistfunc_(query, alg_hnsw.getDataByInternalId(ep), alg_hnsw.dist_func_param_);
            
            // Fast routing (no tracking)
            for (int level = L; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    auto* data = reinterpret_cast<unsigned int*>(alg_hnsw.get_linklist(ep, level));
                    int sz = alg_hnsw.getListCount(reinterpret_cast<hnswlib::linklistsizeint*>(data));
                    auto* nbrs = reinterpret_cast<tableint*>(data + 1);
                    for (int j = 0; j < sz; j++) {
                        float d = alg_hnsw.fstdistfunc_(query, alg_hnsw.getDataByInternalId(nbrs[j]), alg_hnsw.dist_func_param_);
                        if (d < curdist) { curdist = d; ep = nbrs[j]; changed = true; }
                    }
                }
            }
            
            // ef=50 greedy
            std::priority_queue<std::pair<float, tableint>> top_c;
            std::priority_queue<std::pair<float, tableint>, std::vector<std::pair<float, tableint>>, std::greater<std::pair<float, tableint>>> cand_s;
            std::unordered_set<tableint> vis;
            vis.insert(ep); cand_s.emplace(curdist, ep); top_c.emplace(curdist, ep);
            std::vector<float> dists; dists.push_back(curdist); float lb = curdist;
            
            while (!cand_s.empty()) {
                auto pair = cand_s.top();
                if (pair.first > lb && top_c.size() == 50) break;
                cand_s.pop();
                auto* data = reinterpret_cast<int*>(alg_hnsw.get_linklist0(pair.second));
                int sz = alg_hnsw.getListCount(reinterpret_cast<hnswlib::linklistsizeint*>(data));
                auto* nbrs = reinterpret_cast<tableint*>(data + 1);
                for (int j = 0; j < sz; j++) {
                    if (vis.find(nbrs[j]) == vis.end()) {
                        vis.insert(nbrs[j]);
                        float d = alg_hnsw.fstdistfunc_(query, alg_hnsw.getDataByInternalId(nbrs[j]), alg_hnsw.dist_func_param_);
                        dists.push_back(d);
                        if (top_c.size() < 50 || lb > d) {
                            cand_s.emplace(d, nbrs[j]); top_c.emplace(d, nbrs[j]);
                            if (top_c.size() > 50) top_c.pop();
                            lb = top_c.top().first;
                        }
                    }
                }
            }
            
            std::sort(dists.begin(), dists.end());
            M_out[i] = dists[0] / d_mean;
            
            int K = std::min(32, (int)dists.size());
            float m = 0;
            if (K > 1) {
                float d_K = dists[K-1] + 1e-6f; float sum_log = 0;
                for (int k = 0; k < K - 1; k++) sum_log += std::log(d_K / (dists[k] + 1e-6f));
                if (sum_log > 0) m = (K - 1) / sum_log;
            }
            m_out[i] = m;
        }
        auto end = std::chrono::high_resolution_clock::now();
        time_B += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }

    // ---------------------------------------------------------
    // Method C: M_RC + m_Sphere (2-Hop)
    // ---------------------------------------------------------
    long long time_C = 0;
    for(int run = 0; run < 3; run++) {
        auto start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for (int i = 0; i < nq; i++) {
            Eigen::RowVectorXf q = query_vectors.row(i);
            const float* query = q.data();
            
            float d_mean = std::max(0.01f, 1.0f - (float)q.dot(global_mean));
            tableint ep = alg_hnsw.enterpoint_node_;
            float curdist = alg_hnsw.fstdistfunc_(query, alg_hnsw.getDataByInternalId(ep), alg_hnsw.dist_func_param_);
            
            // Fast routing (no tracking)
            for (int level = L; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    auto* data = reinterpret_cast<unsigned int*>(alg_hnsw.get_linklist(ep, level));
                    int sz = alg_hnsw.getListCount(reinterpret_cast<hnswlib::linklistsizeint*>(data));
                    auto* nbrs = reinterpret_cast<tableint*>(data + 1);
                    for (int j = 0; j < sz; j++) {
                        float d = alg_hnsw.fstdistfunc_(query, alg_hnsw.getDataByInternalId(nbrs[j]), alg_hnsw.dist_func_param_);
                        if (d < curdist) { curdist = d; ep = nbrs[j]; changed = true; }
                    }
                }
            }
            
            // 2-Hop Sphere Probe
            std::vector<float> dists; dists.reserve(1050);
            std::unordered_set<tableint> vis;
            dists.push_back(curdist); vis.insert(ep);
            std::vector<tableint> hop1;
            
            auto* data_ep = reinterpret_cast<int*>(alg_hnsw.get_linklist0(ep));
            int sz_ep = alg_hnsw.getListCount(reinterpret_cast<hnswlib::linklistsizeint*>(data_ep));
            auto* nbrs_ep = reinterpret_cast<tableint*>(data_ep + 1);
            for (int j = 0; j < sz_ep; j++) {
                if (vis.find(nbrs_ep[j]) == vis.end()) {
                    vis.insert(nbrs_ep[j]); hop1.push_back(nbrs_ep[j]);
                    dists.push_back(alg_hnsw.fstdistfunc_(query, alg_hnsw.getDataByInternalId(nbrs_ep[j]), alg_hnsw.dist_func_param_));
                }
            }
            
            for (tableint h1 : hop1) {
                auto* data_h1 = reinterpret_cast<int*>(alg_hnsw.get_linklist0(h1));
                int sz_h1 = alg_hnsw.getListCount(reinterpret_cast<hnswlib::linklistsizeint*>(data_h1));
                auto* nbrs_h1 = reinterpret_cast<tableint*>(data_h1 + 1);
                for (int j = 0; j < sz_h1; j++) {
                    if (vis.find(nbrs_h1[j]) == vis.end()) {
                        vis.insert(nbrs_h1[j]);
                        dists.push_back(alg_hnsw.fstdistfunc_(query, alg_hnsw.getDataByInternalId(nbrs_h1[j]), alg_hnsw.dist_func_param_));
                    }
                }
            }
            
            std::sort(dists.begin(), dists.end());
            M_out[i] = dists[0] / d_mean;
            
            int K = std::min(32, (int)dists.size());
            float m = 0;
            if (K > 1) {
                float d_K = dists[K-1] + 1e-6f; float sum_log = 0;
                for (int k = 0; k < K - 1; k++) sum_log += std::log(d_K / (dists[k] + 1e-6f));
                if (sum_log > 0) m = (K - 1) / sum_log;
            }
            m_out[i] = m;
        }
        auto end = std::chrono::high_resolution_clock::now();
        time_C += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }

    std::cout << "\n=======================================================" << std::endl;
    std::cout << "          Metric Computation Overhead (10,000 queries) " << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout << std::left << std::setw(38) << "Method" << " | " << "Average Time (ms)" << std::endl;
    std::cout << "-------------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(38) << "Config A: Original (M_Graph + Spear)" << " | " << (time_A / 3.0f) << " ms" << std::endl;
    std::cout << std::left << std::setw(38) << "Config B: Fast RC (M_RC + Spear)" << " | " << (time_B / 3.0f) << " ms" << std::endl;
    std::cout << std::left << std::setw(38) << "Config C: Pure Sphere (M_RC + 2-Hop)" << " | " << (time_C / 3.0f) << " ms" << std::endl;
    std::cout << "=======================================================\n" << std::endl;

    return 0;
}
