#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <filesystem>
#include <chrono>

#include "../experiments_driver/util.h"
#include "../hnswlib/distribution.h"
#include <boost/math/distributions/normal.hpp>

int main() {
    // 1. Path handling
    const char *experiments_root = std::getenv("EXPERIMENTS_ROOT");
    std::filesystem::path root = experiments_root ? std::filesystem::path(experiments_root)
                                                  : std::filesystem::current_path();

    std::string dataset = "deep-image-96-angular";
    std::string hdf5_path = (root / "data" / (dataset + ".hdf5")).string();

    if (!std::filesystem::exists(hdf5_path)) {
        // Try fallback location just in case
        hdf5_path = (root / "experiments/data" / (dataset + ".hdf5")).string();
        if (!std::filesystem::exists(hdf5_path)) {
            std::cerr << "Dataset missing: " << hdf5_path << std::endl;
            return 1;
        }
    }

    std::cout << "Loading dataset: " << hdf5_path << std::endl;
    hnswdis::MatrixXf full_data;
    hnswdis::MatrixXf query_vectors;
    hnswdis::MatrixXi ground_truth;

    load_hdf5(hdf5_path, query_vectors, full_data, ground_truth);

    // Normalize data (for cosine/angular distance linear combinations)
    std::cout << "Normalizing data vectors..." << std::endl;
    normalize_matrix(full_data);
    normalize_matrix(query_vectors);

    std::cout << "Data points: " << full_data.rows() << ", Dims: " << full_data.cols() << std::endl;
    std::cout << "Queries: " << query_vectors.rows() << std::endl;

    // 2. Initialize the estimator
    std::cout << "Initializing CosineSimilarityEstimator..." << std::endl;
    hnswdis::CosineSimilarityEstimator estimator(full_data);

    // 3. Pick a query
    int q_idx = 0;
    Eigen::RowVectorXf q = query_vectors.row(q_idx);

    // 4. Theoretical distribution
    auto [th_mean, th_var] = estimator.get_practical_distribution(q);
    float th_std = std::sqrt(th_var);
    std::cout << "\n=============================================" << std::endl;
    std::cout << "[Theoretical Distribution for Query " << q_idx << "]" << std::endl;
    std::cout << "Mean:   " << th_mean << std::endl;
    std::cout << "Var:    " << th_var << std::endl;
    std::cout << "StdDev: " << th_std << std::endl;

    // 5. Empirical distribution
    std::cout << "\nComputing exact similarities (Linear Combination of distance vector)..." << std::endl;
    int n = full_data.rows();
    std::vector<float> sim(n);

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        sim[i] = q.dot(full_data.row(i));
    }

    double emp_sum = std::accumulate(sim.begin(), sim.end(), 0.0);
    double emp_mean = emp_sum / n;

    double emp_sq_sum = 0;
    double emp_cube_sum = 0;
    double emp_quad_sum = 0;

    #pragma omp parallel for reduction(+:emp_sq_sum, emp_cube_sum, emp_quad_sum)
    for (int i = 0; i < n; ++i) {
        double diff = sim[i] - emp_mean;
        emp_sq_sum += diff * diff;
        emp_cube_sum += diff * diff * diff;
        emp_quad_sum += diff * diff * diff * diff;
    }
    double emp_var = emp_sq_sum / (n - 1);
    double emp_std = std::sqrt(emp_var);
    double emp_skewness = (emp_cube_sum / n) / std::pow(emp_std, 3);
    double emp_kurtosis = (emp_quad_sum / n) / std::pow(emp_std, 4) - 3.0; // Excess kurtosis

    std::cout << "\n[Empirical Distribution for Query " << q_idx << "]" << std::endl;
    std::cout << "Mean:             " << emp_mean << std::endl;
    std::cout << "Var:              " << emp_var << std::endl;
    std::cout << "StdDev:           " << emp_std << std::endl;
    std::cout << "Skewness:         " << emp_skewness << " (Normal dist = 0)" << std::endl;
    std::cout << "Excess Kurtosis:  " << emp_kurtosis << " (Normal dist = 0)" << std::endl;

    // 6. Histogram comparison to verify normality
    std::cout << "\n[Normality Verification - Histogram Comparison]" << std::endl;
    boost::math::normal_distribution<> normal_dist(th_mean, th_std);

    int num_bins = 20;
    float min_val = th_mean - 4 * th_std;
    float max_val = th_mean + 4 * th_std;
    float bin_width = (max_val - min_val) / num_bins;

    std::vector<int> counts(num_bins, 0);
    for (int i = 0; i < n; ++i) {
        int bin = std::floor((sim[i] - min_val) / bin_width);
        if (bin >= 0 && bin < num_bins) {
            counts[bin]++;
        }
    }

    std::cout << "Range [Z-score]\t\tEmpirical%\tTheoretical%\tDifference\n";
    std::cout << "------------------------------------------------------------------\n";
    for (int i = 0; i < num_bins; ++i) {
        float bin_start = min_val + i * bin_width;
        float bin_end = bin_start + bin_width;

        float emp_prob = (float)counts[i] / n;
        float th_prob = boost::math::cdf(normal_dist, bin_end) - boost::math::cdf(normal_dist, bin_start);

        float z_start = (bin_start - th_mean) / th_std;
        float z_end = (bin_end - th_mean) / th_std;

        printf("[%5.2f, %5.2f)\t\t%6.2f%%\t\t%6.2f%%\t\t%+6.2f%%\n",
               z_start, z_end, emp_prob * 100, th_prob * 100, (emp_prob - th_prob) * 100);
    }
    std::cout << "=============================================" << std::endl;

    return 0;
}
