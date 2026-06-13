#include "util.h"
#include "../hnswlib/lid_2d_calibrator.h"
#include <filesystem>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>

const char *experiments_root_lid = std::getenv("EXPERIMENTS_ROOT");
const auto root_lid = experiments_root_lid
    ? std::filesystem::path(experiments_root_lid)
    : std::filesystem::current_path();

static std::string target_dataset_lid = "";

struct DatasetConfig {
    std::string name;
    std::string metric;
    int         k;
};

static const std::vector<DatasetConfig> DATASETS = {
    {"deep-image-96-angular",   "cd",  100},
    {"glove-100-angular",       "cd",  100},
    {"msmarco",                 "cd", 1000},
    {"cohere",                  "cd", 1000},
    {"laion_image",             "cd", 1000},
    {"cluster_mg_uniform_100d", "cd", 1000},
    {"cluster_mg_zipf_100d",    "cd", 1000},
};

static std::vector<float> compute_recall_vec(
    const hnswdis::MatrixXi&                        ground_truth,
    const std::vector<std::vector<std::size_t>>&    results,
    int                                             k)
{
    int nq = static_cast<int>(results.size());
    std::vector<float> recalls(nq, 0.0f);
    for (int i = 0; i < nq; ++i) {
        int gt_cols = static_cast<int>(ground_truth.cols());
        int limit   = std::min(k, gt_cols);
        int found   = 0;
        for (std::size_t label : results[i]) {
            for (int j = 0; j < limit; ++j) {
                if (static_cast<std::size_t>(ground_truth(i, j)) == label) {
                    ++found;
                    break;
                }
            }
        }
        recalls[i] = static_cast<float>(found) / static_cast<float>(limit);
    }
    return recalls;
}

static void run_lid_experiment(const DatasetConfig& cfg)
{
    std::string hdf5_path  = (root_lid / "data"  / (cfg.name + ".hdf5")).string();
    std::string index_path = (root_lid / "index" / (cfg.name + "-M16-efc-500-parallel.hnsw")).string();
    std::string table_path = (root_lid / "estimation_table" / (cfg.name + "-lid-2d-table-k" + std::to_string(cfg.k) + ".bin")).string();

    if (!std::filesystem::exists(hdf5_path)) {
        std::cerr << "Missing dataset: " << hdf5_path << ", skipping." << std::endl;
        return;
    }
    if (!std::filesystem::exists(index_path)) {
        std::cerr << "Missing index: " << index_path << ", skipping." << std::endl;
        return;
    }

    std::cout << "\n=== LID-Adaptive-EF: " << cfg.name << " k=" << cfg.k << " ===" << std::endl;

    auto [hnsw, query, data, ground_truth, space] =
        load_index_and_data(hdf5_path, index_path, cfg.metric);

    const int nq = static_cast<int>(query->rows());
    const int k  = cfg.k;

    hnswlid::Lid2DTable table;

    if (std::filesystem::exists(table_path)) {
        std::cout << "Loading LID 2D ef-table from " << table_path << std::endl;
        table.load(table_path);
    } else {
        table = hnswlid::calibrate_lid_2d_table(*hnsw, *data, k, cfg.metric);
        std::filesystem::create_directories(std::filesystem::path(table_path).parent_path());
        table.save(table_path);
        std::cout << "LID 2D ef-table saved to " << table_path << std::endl;
    }

    std::cout << "\n--- LID-Adaptive-EF search ---" << std::endl;
    auto batch_stats = hnswlid::lid_batch_search(*hnsw, *query, k, table, k);

    auto recalls = compute_recall_vec(*ground_truth, batch_stats.labels, k);
    float avg_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0f) / static_cast<float>(recalls.size());

    std::vector<float> sorted_recalls = recalls;
    std::sort(sorted_recalls.begin(), sorted_recalls.end());
    float p5  = sorted_recalls[static_cast<std::size_t>(sorted_recalls.size() * 0.05)];
    float p1  = sorted_recalls[static_cast<std::size_t>(sorted_recalls.size() * 0.01)];

    float avg_ef = 0.0f;
    for (int v : batch_stats.ef_used) avg_ef += static_cast<float>(v);
    avg_ef /= static_cast<float>(batch_stats.ef_used.size());

    std::cout << "Dataset:          " << cfg.name           << std::endl;
    std::cout << "Queries:          " << nq                  << std::endl;
    std::cout << "k:                " << k                   << std::endl;
    std::cout << "Search time (ms): " << batch_stats.search_ms << std::endl;
    std::cout << "Avg recall:       " << avg_recall           << std::endl;
    std::cout << "5th pct recall:   " << p5                   << std::endl;
    std::cout << "1st pct recall:   " << p1                   << std::endl;
    std::cout << "Avg ef used:      " << avg_ef               << std::endl;

    std::cout << "\n--- Baseline HNSW (ef=avg_ef) ---" << std::endl;
    int baseline_ef = static_cast<int>(avg_ef);
    hnsw->setEf(baseline_ef);
    std::vector<std::vector<std::size_t>> baseline_results;
    baseline_results.reserve(nq);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nq; ++i) {
        auto ret = hnsw->searchKnn(query->row(i).data(), k);
        int cnt  = static_cast<int>(ret.size());
        std::vector<std::size_t> labels(cnt);
        while (!ret.empty()) {
            labels[--cnt] = ret.top().second;
            ret.pop();
        }
        baseline_results.push_back(std::move(labels));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    long long baseline_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    auto base_recalls = compute_recall_vec(*ground_truth, baseline_results, k);
    float base_avg = std::accumulate(base_recalls.begin(), base_recalls.end(), 0.0f) / static_cast<float>(base_recalls.size());

    std::cout << "Baseline ef:       " << baseline_ef  << std::endl;
    std::cout << "Search time (ms):  " << baseline_ms  << std::endl;
    std::cout << "Avg recall:        " << base_avg     << std::endl;

    std::string csv_path = (root_lid / "estimation_table" / (cfg.name + "-lid-results-k" + std::to_string(k) + ".csv")).string();
    std::ofstream csv(csv_path);
    if (csv.is_open()) {
        csv << "method,avg_recall,p5_recall,p1_recall,avg_ef,search_ms\n";
        csv << "lid_adaptive," << avg_recall << "," << p5 << "," << p1 << "," << avg_ef << "," << batch_stats.search_ms << "\n";
        csv << "baseline_hnsw," << base_avg << ",,,," << baseline_ms << "\n";
        std::cout << "Results saved to " << csv_path << std::endl;
    }
}

int main(int argc, char** argv)
{
    if (argc >= 2) {
        target_dataset_lid = argv[1];
        std::cout << "Target dataset: " << target_dataset_lid << std::endl;
    }

    for (const auto& cfg : DATASETS) {
        if (!target_dataset_lid.empty() && cfg.name != target_dataset_lid) continue;
        run_lid_experiment(cfg);
    }

    return 0;
}
