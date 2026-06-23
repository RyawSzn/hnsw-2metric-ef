#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <string>

#include "../experiments_driver/util.h"
#include "../hnswlib/hnswlib.h"
#include "estimator.h"
#include "lookuptable.h"
#include "statistics.h"
#include "table_generator.h"

using namespace hnswlib;
using namespace hnsw_2metric;
namespace fs = std::filesystem;

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
                  << "  --repeat <int>          Number of repeats for search (default: 1)\n"
                  << "  --generate_table        Force re-generation of the lookup table\n"
                  << "  --ep_bins <int>         Number of relative_contrast bins (default: 32)\n"
                  << "  --rv_bins <int>         Number of RV bins (default: 32)\n"
                  << "  --sample_size <int>     Queries to sample for table generation (default: 5000)\n";
        return 0;
    }

    std::string dataset = get_cmd_option(argv, argv + argc, "--dataset", "glove-100-angular");
    float expected_recall = std::stof(get_cmd_option(argv, argv + argc, "--target_recall", "0.95"));
    int k = std::stoi(get_cmd_option(argv, argv + argc, "--k", "32"));
    int max_ef = std::stoi(get_cmd_option(argv, argv + argc, "--max_ef", "5000"));
    int repeat = std::stoi(get_cmd_option(argv, argv + argc, "--repeat", "1"));
    bool force_generate = cmd_option_exists(argv, argv + argc, "--generate_table");
    int entry_point_bins = std::stoi(get_cmd_option(argv, argv + argc, "--ep_bins", "32"));
    int revisit_bins = std::stoi(get_cmd_option(argv, argv + argc, "--rv_bins", "32"));
    int sample_size = std::stoi(get_cmd_option(argv, argv + argc, "--sample_size", "5000"));

    // Auto-resolve lookup CSV name to match Goal 1
    const char* root_env = std::getenv("EXPERIMENTS_ROOT");
    fs::path root = root_env ? fs::path(root_env) : fs::current_path();

    // We expect the CSV to have the dataset name in it
    std::string lookup_csv_name = "lookup_table_" + dataset + "_" + std::to_string(entry_point_bins) + "x" + std::to_string(revisit_bins) + ".csv";
    std::string lookup_csv = (root / "2metric/lookup" / lookup_csv_name).string();
    if (!fs::exists(lookup_csv)) {
        lookup_csv = "/home/ryawszn/experiments/2metric/lookup/" + lookup_csv_name;
    }

    std::string hdf5_path = (root / "data" / (dataset + ".hdf5")).string();
    if (!fs::exists(hdf5_path)) {
        hdf5_path = "/home/ryawszn/experiments/data/" + dataset + ".hdf5";
    }

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

    SpaceInterface<float>* space;
    if (dataset.find("euclidean") != std::string::npos) {
        space = new L2Space(full_data.cols());
    } else {
        space = new InnerProductSpace(full_data.cols());
    }

    auto* alg_hnsw = new HierarchicalNSW<float>(space, index_path, false, full_data.rows());

    Eigen::RowVectorXf global_mean = full_data.colwise().mean();

    LookupTable2D lookup;
    if (force_generate || !fs::exists(lookup_csv)) {
        fs::create_directories(fs::path(lookup_csv).parent_path());
        lookup = TableGenerator2Metric::generate(
            alg_hnsw, query_vectors, ground_truth, global_mean,
            expected_recall, k, entry_point_bins, revisit_bins, max_ef, sample_size, lookup_csv
        );
    } else {
        std::cout << "Loading existing lookup table from " << lookup_csv << "\n";
        lookup = LookupTable2D(lookup_csv, 32, expected_recall);
    }

    int table_avg_ef = lookup.get_average_ef();
    std::cout << "Lookup Table Static Average EF: " << table_avg_ef << "\n";

    int nq = query_vectors.rows();
    std::vector<std::vector<size_t>> result(nq, std::vector<size_t>(k, 0));
    std::vector<int> efs_used(nq);

    std::cout << "Starting 2metric adaptive search...\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < nq; ++i) {
        const float* q_ptr = query_vectors.row(i).data();

        auto res = Estimator2Metric::probe_query(alg_hnsw, q_ptr, global_mean, 32, 15.0f);

        int dyn_ef = std::max(lookup.get_ef(res.entry_point_dist, res.revisit_rank), table_avg_ef);
        if (dyn_ef < static_cast<int>(k)) dyn_ef = static_cast<int>(k);
        if (dyn_ef > max_ef) dyn_ef = max_ef;
        efs_used[i] = dyn_ef;

        // Use the fast-path stateless resume
        std::priority_queue<std::pair<float, labeltype>> pq = alg_hnsw->searchKnnFromEp(res.ep_id, q_ptr, k, static_cast<size_t>(dyn_ef));

        int count = pq.size();
        while (!pq.empty()) {
            result[i][--count] = pq.top().second;
            pq.pop();
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "\nSearch time: " << duration.count() << " ms\n";

    auto recalls = hnswdis::compute_recall(ground_truth, result, k, false);
    float avg_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0f) / recalls.size();
    float avg_ef = std::accumulate(efs_used.begin(), efs_used.end(), 0.0f) / efs_used.size();

    std::sort(recalls.begin(), recalls.end());
    float percentile_5 = recalls[static_cast<size_t>(recalls.size() * 0.05)];
    float percentile_1 = recalls[static_cast<size_t>(recalls.size() * 0.01)];

    std::cout << "------------------------------------------\n";
    std::cout << "Dataset:          " << dataset << "\n";
    std::cout << "Lookup Table:     " << lookup_csv << "\n";
    std::cout << "Target Recall:    " << expected_recall << "\n";
    std::cout << "Max EF Limit:     " << max_ef << "\n";
    std::cout << "Average Recall:   " << avg_recall << "\n";
    std::cout << "Average EF Used:  " << avg_ef << "\n";
    std::cout << "5th %ile Recall:  " << percentile_5 << "\n";
    std::cout << "1st %ile Recall:  " << percentile_1 << "\n";
    std::cout << "------------------------------------------\n";

    std::string stats_json = root.string() + "/2metric/lookup/run_stats_" + dataset + "_" + std::to_string(entry_point_bins) + "x" + std::to_string(revisit_bins) + ".json";
    std::ofstream out_json(stats_json);
    if (out_json) {
        out_json << "{\n";
        out_json << "  \"search_ms\": " << duration.count() << ",\n";
        out_json << "  \"avg_recall\": " << avg_recall << ",\n";
        out_json << "  \"avg_ef\": " << avg_ef << ",\n";
        out_json << "  \"pct5_recall\": " << percentile_5 << ",\n";
        out_json << "  \"pct1_recall\": " << percentile_1 << ",\n";
        out_json << "  \"target_recall\": " << expected_recall << ",\n";
        out_json << "  \"max_ef\": " << max_ef << ",\n";
        out_json << "  \"nq\": " << nq << ",\n";

        std::map<int, int> ef_dist;
        for (int ef : efs_used) ef_dist[ef]++;

        out_json << "  \"ef_dist\": {\n";
        int count_dist = 0;
        for (auto const& [ef, count] : ef_dist) {
            out_json << "    \"" << ef << "\": " << count;
            if (++count_dist < ef_dist.size()) out_json << ",";
            out_json << "\n";
        }
        out_json << "  }\n";
        out_json << "}\n";
        out_json.close();
        std::cout << "Saved run statistics to " << stats_json << "\n";
    }

    // Call the statistics function similar to adaptive_ef_analysis
    adaptive_ef_analysis_2metric(dataset, efs_used);

    // Call baseline search to compare directly to constant EF curves
    // std::cout << "\nRunning Baseline Search for Comparison (up to max_ef=" << max_ef << ")...\n";
    // baseline_search(dataset, repeat, *alg_hnsw, query_vectors, ground_truth, k, max_ef);

    delete alg_hnsw;
    delete space;

    return 0;
}
