#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <filesystem>
#include "../hnswlib/hnswlib.h"
#include "../experiments_driver/util.h"
#include "../2metric/estimator.h"

using namespace hnswlib;
using namespace hnsw_2metric;
namespace fs = std::filesystem;

int main() {
    std::string dataset = "glove-100-angular";
    std::string root = "/home/ryawszn/experiments";
    
    std::string data_path = root + "/data/" + dataset + ".hdf5";
    std::string index_path = root + "/index/" + dataset + "-M16-efc-500-parallel.hnsw";
    
    hnswdis::MatrixXf data_vectors, query_vectors;
    hnswdis::MatrixXi ground_truth;
    load_hdf5(data_path, query_vectors, data_vectors, ground_truth);
    
    int dim = data_vectors.cols();
    size_t k = 32; // Use standard 32
    
    InnerProductSpace space(dim);
    HierarchicalNSW<float>* alg_hnsw = new HierarchicalNSW<float>(&space, index_path);
    
    normalize_matrix(data_vectors);
normalize_matrix(query_vectors);
Eigen::RowVectorXf global_mean = data_vectors.colwise().mean();
global_mean.normalize();
    
    int nq = 2000; // sample size
    std::vector<int> queries(nq);
    std::iota(queries.begin(), queries.end(), 0);
    std::srand(42);
    std::random_shuffle(queries.begin(), queries.end());
    
    std::string out_csv = root + "/2metric/lookup/research_categorizers_" + dataset + ".csv";
    std::ofstream out(out_csv);
    out << "query_idx,ef_true,RC,RV_rank,d_mean,d_ep,m_LID\n";
    
    std::cout << "Testing " << nq << " queries for categorizer research...\n";
    
    #pragma omp parallel for
    for (int i = 0; i < nq; i++) {
        int q_idx = queries[i];
        const float* query = query_vectors.row(q_idx).data();
        
        auto est = Estimator2Metric::probe_query(alg_hnsw, query, global_mean, 50, 15.0f);
        
        int ef_true = 5000;
        for (int ef = 50; ef <= 5000; ef += 50) {
            alg_hnsw->setEf(ef);
            auto pq = alg_hnsw->searchKnn(query, k);
            int hits = 0;
            while (!pq.empty()) {
                size_t id = pq.top().second; pq.pop();
                for (size_t c = 0; c < k; c++) {
                    if (id == ground_truth(q_idx, c)) { hits++; break; }
                }
            }
            if ((float)hits / k >= 0.95f) {
                ef_true = ef;
                break;
            }
        }
        
        #pragma omp critical
        {
            out << q_idx << "," << ef_true << "," 
                << 0 << "," << est.revisit_rank << "," 
                << 0 << "," << est.entry_point_dist << "," << 0 << "\n";
            if (i % 200 == 0) std::cout << "Processed " << i << " queries...\n";
        }
    }
    
    out.close();
    delete alg_hnsw;
    std::cout << "Saved to " << out_csv << "\n";
    return 0;
}
