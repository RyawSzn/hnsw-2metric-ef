#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <string>
#include <fstream>
#include <iomanip>

#include "../../experiments_driver/util.h"
#include "../../hnswlib/hnswlib.h"
#include "../estimator.h"
#include "../lookuptable.h"

using namespace hnswlib;
using namespace hnsw_2metric;
namespace fs = std::filesystem;

// Find exact EF required to hit target recall for a single query
std::pair<int, float> find_true_ef_for_query(HierarchicalNSW<float>* alg_hnsw, const float* query, const hnswdis::MatrixXi& ground_truth, int original_idx, float target_recall, int max_ef, int SEARCH_K) {
    int best_ef = 50;
    float best_recall = 0.0f;
    size_t k_val = ground_truth.cols();
    if (k_val > SEARCH_K) k_val = SEARCH_K;

    for (int ef = 50; ef <= max_ef; ef += 50) {
        alg_hnsw->setEf(ef);
        auto sres = alg_hnsw->searchKnn(query, k_val);
        int hits = 0;

        std::vector<size_t> res(sres.size());
        int count = sres.size();
        while (!sres.empty()) {
            res[--count] = sres.top().second;
            sres.pop();
        }

        for (size_t r : res) {
            for (size_t c = 0; c < k_val; c++) {
                if (ground_truth(original_idx, c) == (int)r) { hits++; break; }
            }
        }

        float recall = (float)hits / k_val;
        best_ef = ef;
        best_recall = recall;
        if (recall >= target_recall) break;
    }
    return {best_ef, best_recall};
}

int main() {
    std::string dataset = "glove-100-angular";
    float target_recall = 0.95f;
    int max_ef = 5000;
    int sample_size = 5000;
    int SEARCH_K = 32;

    const char* root_env = std::getenv("EXPERIMENTS_ROOT");
    fs::path root = root_env ? fs::path(root_env) : fs::current_path();

    std::string hdf5_path = (root / "data" / (dataset + ".hdf5")).string();
    if (!fs::exists(hdf5_path)) {
        hdf5_path = "/home/ryawszn/experiments/data/" + dataset + ".hdf5";
    }

    std::cout << "Loading dataset: " << dataset << "\n";
    hnswdis::MatrixXf full_data, query_vectors;
    hnswdis::MatrixXi ground_truth;
    load_hdf5(hdf5_path, query_vectors, full_data, ground_truth);

    if (dataset.find("angular") != std::string::npos) {
        normalize_matrix(full_data);
        normalize_matrix(query_vectors);
    }

    std::string index_path = (root / "index" / (dataset + "-M16-efc-500-parallel.hnsw")).string();
    if (!fs::exists(index_path)) {
        index_path = "/home/ryawszn/experiments/index/" + dataset + "-M16-efc-500-parallel.hnsw";
    }

    std::cout << "Loading HNSW...\n";
    SpaceInterface<float>* space;
    if (dataset.find("euclidean") != std::string::npos) {
        space = new L2Space(full_data.cols());
    } else {
        space = new InnerProductSpace(full_data.cols());
    }
    auto* alg_hnsw = new HierarchicalNSW<float>(space, index_path, false, full_data.rows());

    // --- ADA-EF Setup ---
    std::cout << "Setting up Ada-EF dependencies...\n";
    size_t ada_k = 100;

    std::string estimator_path = (root / "statistics" / (dataset + "-estimator--k-" + std::to_string(ada_k) + ".bin")).string();
    if (!fs::exists(estimator_path)) {
        estimator_path = "/home/ryawszn/experiments/statistics/" + dataset + "-estimator--k-" + std::to_string(ada_k) + ".bin";
    }

    std::string ef_adaptor_path = (root / "estimation_table" / (dataset + "-ef_adaptor--k" + std::to_string(ada_k) + "-ef.bin")).string();
    if (!fs::exists(ef_adaptor_path)) {
        ef_adaptor_path = "/home/ryawszn/experiments/estimation_table/" + dataset + "-ef_adaptor--k" + std::to_string(ada_k) + "-ef.bin";
    }

    std::shared_ptr<hnswdis::Estimator> ada_estimator = hnswdis::load_estimator_from_file(estimator_path);
    hnswdis::ApproximatedScoreCalculator score_cal(ada_estimator, 1e-3);

    hnswdis::EfAdapter ef_adapter(ef_adaptor_path);
    auto ef_adapter_ptr = std::make_shared<hnswdis::EfAdapter>(ef_adapter);
    hnswdis::Sketch sketch(ef_adapter_ptr->get_ef_recall_estimators(), target_recall);
    size_t statics_length = 1 + 32 + 31 * 32;

    // --- 2METRIC Setup ---
    std::cout << "Setting up 2Metric dependencies...\n";
    Eigen::RowVectorXf global_mean = full_data.colwise().mean();
    std::string lookup_csv = (root / "2metric/lookup" / ("lookup_table_" + dataset + "_32x32.csv")).string();
    if (!fs::exists(lookup_csv)) {
        lookup_csv = "/home/ryawszn/experiments/2metric/lookup/lookup_table_" + dataset + "_32x32.csv";
    }
    LookupTable2D lookup(lookup_csv, 50, target_recall);

    // Random sample of queries
    std::vector<int> q_idx(query_vectors.rows());
    std::iota(q_idx.begin(), q_idx.end(), 0);
    std::srand(42);
    std::random_shuffle(q_idx.begin(), q_idx.end());
    q_idx.resize(sample_size);

    std::cout << "Starting comparison loop for " << sample_size << " sampled queries...\n";

    std::string out_path = (root / "2metric/compare" / ("comparison_" + dataset + ".csv")).string();
    if (!fs::exists(root / "2metric/compare")) {
        out_path = "/home/ryawszn/experiments/2metric/compare/comparison_" + dataset + ".csv";
        fs::create_directories("/home/ryawszn/experiments/2metric/compare");
    }
    std::ofstream out(out_path);
    out << "query_idx,true_ef,ada_ef,2metric_ef,entry_point_dist,revisit_rank\n";

    double total_err_ada = 0;
    double total_err_2metric = 0;
    int ada_under = 0, metric2_under = 0;

    #pragma omp parallel for reduction(+:total_err_ada, total_err_2metric, ada_under, metric2_under)
    for (int i = 0; i < sample_size; i++) {
        int idx = q_idx[i];
        Eigen::RowVectorXf q = query_vectors.row(idx);

        // 1. True EF
        auto [true_ef, recall] = find_true_ef_for_query(alg_hnsw, q.data(), ground_truth, idx, target_recall, max_ef, SEARCH_K);

        // 2. Ada-EF
        auto ada_res = alg_hnsw->adaptiveSearchKnn(q.data(), SEARCH_K, statics_length, score_cal, &sketch);
        float score = ada_res.second;
        size_t ada_ef = sketch.estimate_ef2(score);
        if (ada_ef < SEARCH_K) ada_ef = SEARCH_K;
        if (ada_ef > max_ef) ada_ef = max_ef;

        // 3. 2Metric EF
        auto est = Estimator2Metric::probe_query(alg_hnsw, q.data(), global_mean, 50, 15.0f);
        int m2_ef = lookup.get_ef(est.entry_point_dist, est.revisit_rank);
        if (m2_ef < SEARCH_K) m2_ef = SEARCH_K;
        if (m2_ef > max_ef) m2_ef = max_ef;

        #pragma omp critical
        {
            out << idx << "," << true_ef << "," << ada_ef << "," << m2_ef << "," << est.entry_point_dist << "," << est.revisit_rank << "\n";
            if ((i + 1) % 50 == 0) {
                std::cout << "Processed " << (i + 1) << " / " << sample_size << " queries...\n";
            }
        }

        // Stats tracking
        total_err_ada += std::abs((double)ada_ef - true_ef);
        total_err_2metric += std::abs((double)m2_ef - true_ef);
        if (ada_ef < true_ef) ada_under++;
        if (m2_ef < true_ef) metric2_under++;
    }
    out.close();

    std::cout << "\n================ COMPARISON RESULTS ================\n";
    std::cout << "Sample Size:       " << sample_size << "\n";
    std::cout << "Target Recall:     " << target_recall << "\n";
    std::cout << "----------------------------------------------------\n";
    std::cout << "Ada-EF MAE:        " << (total_err_ada / sample_size) << " (Under-estimated: " << ada_under << ")\n";
    std::cout << "2Metric MAE:       " << (total_err_2metric / sample_size) << " (Under-estimated: " << metric2_under << ")\n";
    std::cout << "----------------------------------------------------\n";
    std::cout << "Full results saved to " << out_path << "\n";

    delete alg_hnsw;
    delete space;
    return 0;
}
