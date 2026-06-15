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

#include "../experiments_driver/util.h"
#include "../hnswlib/hnswlib.h"

using namespace hnswlib;

int main(int argc, char** argv) {
    float target_recall = 0.95f;
    int M_BINS = 10;
    int m_BINS = 10;
    std::string dataset = "glove-100-angular";
    int max_ef = 3000;
    int sample_size = 2000;

    if (argc > 1) target_recall = std::stof(argv[1]);
    if (argc > 2) M_BINS = std::stoi(argv[2]);
    if (argc > 3) m_BINS = std::stoi(argv[3]);
    if (argc > 4) dataset = argv[4];
    if (argc > 5) max_ef = std::stoi(argv[5]);
    if (argc > 6) sample_size = std::stoi(argv[6]);

    std::cout << "=== Generating Fast-Probe 2D Adaptive Matrix ===" << std::endl;
    std::cout << "Target Recall: " << target_recall << std::endl;
    std::cout << "Probe limit:   W = 50 evaluations" << std::endl;
    std::cout << "Matrix Size:   " << M_BINS << " (M) x " << m_BINS << " (m)" << std::endl;
    std::cout << "Dataset:       " << dataset << std::endl;
    std::cout << "------------------------------------------------\n" << std::endl;

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
    HierarchicalNSW<float>* alg_hnsw = new HierarchicalNSW<float>(&space, index_path, false, full_data.rows());

    int total_queries = query_vectors.rows();
    int nq = total_queries;
    std::vector<int> query_indices(total_queries);
    std::iota(query_indices.begin(), query_indices.end(), 0);

    if (sample_size > 0 && sample_size < total_queries) {
        std::srand(42);
        std::random_shuffle(query_indices.begin(), query_indices.end());
        query_indices.resize(sample_size);
        nq = sample_size;
    }

    int L = alg_hnsw->maxlevel_;
    std::vector<float> M_scores(nq, 0.0f);
    std::vector<float> m_scores(nq, 0.0f);

    Eigen::RowVectorXf global_mean = full_data.colwise().mean();

    std::cout << "Phase 1: Fast Probe (W <= 50 elements) to calculate M_RC and m_LID..." << std::endl;

    #pragma omp parallel for
    for (int i = 0; i < nq; i++) {
        int original_idx = query_indices[i];
        Eigen::RowVectorXf q = query_vectors.row(original_idx);
        const float* query = q.data();

        float th_mean = q.dot(global_mean);
        float d_mean = std::max(0.01f, 1.0f - th_mean);

        tableint currObj = alg_hnsw->enterpoint_node_;
        float curdist = alg_hnsw->fstdistfunc_(query, alg_hnsw->getDataByInternalId(currObj), alg_hnsw->dist_func_param_);

        // Upper layers
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

        // Base layer (0) Fast Probe: Stop exactly at |W| = 50
        std::priority_queue<std::pair<float, tableint>, std::vector<std::pair<float, tableint>>, std::greater<std::pair<float, tableint>>> candidate_set;
        std::unordered_set<tableint> visited;

        visited.insert(currObj);
        candidate_set.emplace(curdist, currObj);

        std::vector<float> W;
        W.push_back(curdist);

        while (!candidate_set.empty() && W.size() < 50) {
            auto current_node_pair = candidate_set.top();
            candidate_set.pop();
            tableint cur_node = current_node_pair.second;

            auto* data = reinterpret_cast<int*>(alg_hnsw->get_linklist0(cur_node));
            int sz = alg_hnsw->getListCount(reinterpret_cast<hnswlib::linklistsizeint*>(data));
            auto* nbrs = reinterpret_cast<tableint*>(data + 1);

            for (int j = 0; j < sz; j++) {
                tableint cand = nbrs[j];
                if (visited.find(cand) == visited.end()) {
                    visited.insert(cand);
                    float d = alg_hnsw->fstdistfunc_(query, alg_hnsw->getDataByInternalId(cand), alg_hnsw->dist_func_param_);
                    W.push_back(d);
                    candidate_set.emplace(d, cand);
                    if (W.size() >= 50) break;
                }
            }
        }

        std::sort(W.begin(), W.end());
        float b_0 = W[0]; // Shortest distance found in W

        M_scores[i] = b_0 / d_mean;

        int K = (int)W.size();
        float m_q = 0;
        if (K > 1) {
            float d_K = W[K-1] + 1e-6f;
            float sum_log = 0;
            for (int k = 0; k < K - 1; k++) {
                float d_i = W[k] + 1e-6f;
                sum_log += std::log(d_K / d_i);
            }
            if (sum_log > 0) m_q = (K - 1) / sum_log;
        }
        m_scores[i] = m_q;
    }

    std::cout << "Phase 2: Calculating Quantile Boundaries..." << std::endl;
    std::vector<float> sorted_M = M_scores;
    std::vector<float> sorted_m = m_scores;
    std::sort(sorted_M.begin(), sorted_M.end());
    std::sort(sorted_m.begin(), sorted_m.end());

    std::vector<float> M_bounds(M_BINS + 1);
    std::vector<float> m_bounds(m_BINS + 1);
    for(int i=0; i<=M_BINS; i++) {
        int idx = std::min((int)(nq - 1), (int)(i * nq / (float)M_BINS));
        M_bounds[i] = sorted_M[idx];
    }
    for(int i=0; i<=m_BINS; i++) {
        int idx = std::min((int)(nq - 1), (int)(i * nq / (float)m_BINS));
        m_bounds[i] = sorted_m[idx];
    }
    M_bounds[M_BINS] = std::numeric_limits<float>::max();
    m_bounds[m_BINS] = std::numeric_limits<float>::max();

    std::vector<std::vector<std::vector<int>>> buckets(M_BINS, std::vector<std::vector<int>>(m_BINS));
    for (int i = 0; i < nq; i++) {
        int bM = 0;
        while(bM < M_BINS - 1 && M_scores[i] > M_bounds[bM+1]) bM++;
        int bm = 0;
        while(bm < m_BINS - 1 && m_scores[i] > m_bounds[bm+1]) bm++;
        buckets[bM][bm].push_back(i);
    }

    std::cout << "Phase 3: Stepping ef by +50 to hit Target Recall (" << target_recall << ") for each bucket...\n" << std::endl;

    std::vector<std::vector<std::pair<int, float>>> lookup_table(M_BINS, std::vector<std::pair<int, float>>(m_BINS, {0, 0.0f}));

    int col_width = 13;
    std::string separator(6 + m_BINS * (col_width + 1), '-');
    std::cout << separator << std::endl;
    std::cout << "             Adaptive EF Lookup Table (" << M_BINS << "x" << m_BINS << ")" << std::endl;
    std::cout << separator << std::endl;

    std::cout << "     |";
    for(int j=0; j<m_BINS; j++) std::cout << "      m" << std::setw(2) << std::left << (j+1) << "    |";
    std::cout << "\n" << separator << std::endl;

    char filename[256];
    snprintf(filename, sizeof(filename), "research/lookup_table_fast_%dx%d.csv", M_BINS, m_BINS);
    std::ofstream out(filename);
    out << "M_bin,m_bin,ef,actual_recall\n";

    for (int i = 0; i < M_BINS; i++) {
        std::cout << " M" << std::setw(2) << std::left << (i+1) << " |";
        for (int j = 0; j < m_BINS; j++) {
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

                for (int i_idx : query_list) {
                    int original_idx = query_indices[i_idx];
                    auto res = alg_hnsw->searchKnn(query_vectors.row(original_idx).data(), 10);
                    while (!res.empty()) {
                        tableint id = res.top().second;
                        res.pop();
                        for (int c = 0; c < 10; c++) {
                            if (ground_truth(original_idx, c) == id) {
                                total_hits++;
                                break;
                            }
                        }
                    }
                }

                float avg_recall = (float)total_hits / (query_list.size() * 10);
                best_ef = ef;
                best_recall = avg_recall;

                if (avg_recall >= target_recall) break;
            }

            lookup_table[i][j] = {best_ef, best_recall};
            out << (i+1) << "," << (j+1) << "," << best_ef << "," << best_recall << "\n";
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
    return 0;
}
