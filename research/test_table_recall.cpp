#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "../experiments_driver/util.h"
#include "../hnswlib/hnswlib.h"

using namespace hnswlib;

int main(int argc, char** argv) {
    std::string dataset = "glove-100-angular";
    if (argc > 1) dataset = argv[1];

    std::cout << "=== Testing Actual Recall of 2D Table (M + M_R) ===\n";
    std::cout << "Dataset: " << dataset << "\n\n";

    // 1. Load the dataset and ground truth
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

    // 2. Load the HNSW Index
    std::string index_path = (root / "index" / (dataset + "-M16-efc-500-parallel.hnsw")).string();
    L2Space space(full_data.cols());
    HierarchicalNSW<float>* alg_hnsw = new HierarchicalNSW<float>(&space, index_path, false, full_data.rows());

    // 3. Load the assigned ef predictions
    std::string assignments_path = "research/query_ef_assignments.csv";
    std::ifstream in(assignments_path);
    if (!in.is_open()) {
        std::cerr << "Failed to open " << assignments_path << "\n";
        return 1;
    }

    struct QueryTask {
        int idx;
        float M;
        float M_R;
        int ef_pred;
        float actual_recall = 0.0f;
    };

    std::vector<QueryTask> tasks;
    std::string line;
    std::getline(in, line); // skip header
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string token;
        QueryTask t;
        
        std::getline(ss, token, ','); t.idx = std::stoi(token);
        std::getline(ss, token, ','); t.M = std::stof(token);
        std::getline(ss, token, ','); t.M_R = std::stof(token);
        std::getline(ss, token, ','); t.ef_pred = std::stoi(token);
        
        tasks.push_back(t);
    }
    in.close();

    int nq = tasks.size();
    std::cout << "Loaded " << nq << " query tasks with table-assigned ef values.\n";

    // 4. Run the queries using the assigned ef
    int k_results = 10;
    
    // We cannot easily parallelize because setEf is an index-level mutation.
    // We must run it sequentially to ensure correct ef per query.
    for (int i = 0; i < nq; i++) {
        alg_hnsw->setEf(tasks[i].ef_pred);
        
        const float* query = query_vectors.row(tasks[i].idx).data();
        auto sres = alg_hnsw->searchKnn(query, k_results);

        int hits = 0;
        while (!sres.empty()) {
            tableint id = sres.top().second;
            sres.pop();
            for (int c = 0; c < k_results; c++) {
                if (ground_truth(tasks[i].idx, c) == (int)id) {
                    hits++;
                    break;
                }
            }
        }
        tasks[i].actual_recall = (float)hits / k_results;
        
        if ((i + 1) % 500 == 0) {
            std::cout << "Processed " << (i + 1) << " / " << nq << " queries...\n";
        }
    }

    // 5. Compute global statistics and save output
    double total_recall = 0.0;
    long long total_ef = 0;
    for (const auto& t : tasks) {
        total_recall += t.actual_recall;
        total_ef += t.ef_pred;
    }
    
    double avg_recall = total_recall / nq;
    double avg_ef = (double)total_ef / nq;

    std::string out_path = "research/final_table_results_" + dataset + ".csv";
    std::ofstream out(out_path);
    out << "query_idx,M,M_R,ef_assigned,actual_recall\n";
    for (const auto& t : tasks) {
        out << t.idx << "," << t.M << "," << t.M_R << "," << t.ef_pred << "," << t.actual_recall << "\n";
    }
    out.close();

    std::cout << "\n======================================================\n";
    std::cout << "FINAL RESULTS (M + M_R 20x20 Table @ 90th percentile safety)\n";
    std::cout << "======================================================\n";
    std::cout << "Average EF assigned: " << avg_ef << "\n";
    std::cout << "Average True Recall: " << (avg_recall * 100.0) << "%\n";
    std::cout << "======================================================\n";
    std::cout << "Detailed results written to: " << out_path << "\n";

    delete alg_hnsw;
    return 0;
}
