import sys

with open('../hnswlib/adaptive_ef.h', 'r') as f:
    content = f.read()

# We need to replace the body of compute_ground_truth_batch_parallel4 and compute_ground_truth_batch_parallel4_with_dist

new_compute_gt = """
    MatrixXi compute_ground_truth_batch_parallel4(
        const MatrixXf &query_vectors,
        const MatrixXf &data_vectors,
        const std::string &metric,
        const int k)
    {
        Eigen::setNbThreads(std::max(1u, std::thread::hardware_concurrency() / 4)); // Limit to 1/4 available threads

        size_t totalQueries = query_vectors.rows();
        size_t num_elements = data_vectors.rows();
        MatrixXi ground_truth(totalQueries, k);
        int *ground_truth_ptr = ground_truth.data();

        int numThreads = std::max(1u, std::thread::hardware_concurrency() / 4);
        omp_set_num_threads(numThreads);

        size_t batch_size = 50; 

        auto start_all = std::chrono::high_resolution_clock::now();

        for (size_t batch_start = 0; batch_start < totalQueries; batch_start += batch_size) {
            size_t current_batch_size = std::min(batch_size, totalQueries - batch_start);
            MatrixXf current_query_batch = query_vectors.middleRows(batch_start, current_batch_size);
            
            MatrixXf distances = current_query_batch * data_vectors.transpose();

#pragma omp parallel for schedule(static)
            for (int i = 0; i < static_cast<int>(current_batch_size); ++i)
            {
                int query_i = batch_start + i;
                std::priority_queue<std::pair<float, size_t>> topResults;

                const Eigen::Map<const RowVectorXf> local_distance(distances.row(i).data(), num_elements);

                for (size_t data_i = 0; data_i < k; ++data_i)
                {
                    float dist = 1.0f - local_distance(data_i); // distance-based comparion
                    topResults.emplace(dist, data_i);
                }
                float lastdist = topResults.top().first;
                for (size_t data_i = k; data_i < num_elements; ++data_i)
                {
                    float dist = 1.0f - local_distance(data_i); // distance-based comparion
                    if (dist <= lastdist)
                    {
                        topResults.emplace(dist, data_i);
                        if (topResults.size() > k)
                            topResults.pop();
                        lastdist = topResults.top().first;
                    }
                }

                int *local_ground_truth_ptr = ground_truth_ptr + query_i * k;

                size_t count = k;
                while (!topResults.empty())
                {
                    local_ground_truth_ptr[--count] = static_cast<int>(topResults.top().second);
                    topResults.pop();
                }
            }
        }
        
        auto end_all = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all);
        std::cout << "Ground truth computed in " << duration.count() << " ms using " << numThreads << " threads." << std::endl;

        return ground_truth;
    }
"""

old_compute_gt_start = "    MatrixXi compute_ground_truth_batch_parallel4("
old_compute_gt_end = "    }\n\n    std::pair<Eigen::MatrixXi, Eigen::MatrixXf> compute_ground_truth_batch_parallel4_with_dist("

idx1 = content.find(old_compute_gt_start)
idx2 = content.find(old_compute_gt_end)

if idx1 != -1 and idx2 != -1:
    content = content[:idx1] + new_compute_gt[1:] + "\n" + content[idx2 + 6:]
else:
    print("Could not find compute_ground_truth_batch_parallel4!")


new_compute_gt_dist = """
    std::pair<Eigen::MatrixXi, Eigen::MatrixXf> compute_ground_truth_batch_parallel4_with_dist(
        const MatrixXf &query_vectors,
        const MatrixXf &data_vectors,
        const std::string &metric,
        const int k)
    {
        size_t totalQueries = query_vectors.rows();
        size_t num_elements = data_vectors.rows();

        MatrixXi ground_truth(totalQueries, k);
        MatrixXf ground_truth_distances(totalQueries, k);

        int *ground_truth_ptr = ground_truth.data();
        float *distances_ptr = ground_truth_distances.data();

        int numThreads = std::max(1u, std::thread::hardware_concurrency() / 4);
        omp_set_num_threads(numThreads);
        
        size_t batch_size = 50; 

        for (size_t batch_start = 0; batch_start < totalQueries; batch_start += batch_size) {
            size_t current_batch_size = std::min(batch_size, totalQueries - batch_start);
            MatrixXf current_query_batch = query_vectors.middleRows(batch_start, current_batch_size);
            
            MatrixXf distances = current_query_batch * data_vectors.transpose();

#pragma omp parallel for schedule(static)
            for (int i = 0; i < static_cast<int>(current_batch_size); ++i)
            {
                int query_i = batch_start + i;
                std::priority_queue<std::pair<float, size_t>> topResults;

                const Eigen::Map<const RowVectorXf> local_distance(distances.row(i).data(), num_elements);

                for (size_t data_i = 0; data_i < k; ++data_i)
                {
                    float dist = 1.0f - local_distance(data_i); // distance-based comparion
                    topResults.emplace(dist, data_i);
                }
                float lastdist = topResults.top().first;
                for (size_t data_i = k; data_i < num_elements; ++data_i)
                {
                    float dist = 1.0f - local_distance(data_i); // distance-based comparion
                    if (dist <= lastdist)
                    {
                        topResults.emplace(dist, data_i);
                        if (topResults.size() > k)
                            topResults.pop();
                        lastdist = topResults.top().first;
                    }
                }

                int *local_ground_truth_ptr = ground_truth_ptr + query_i * k;
                float *local_dist_ptr = distances_ptr + query_i * k;

                size_t count = k;
                while (!topResults.empty())
                {
                    local_ground_truth_ptr[--count] = static_cast<int>(topResults.top().second);
                    local_dist_ptr[count] = topResults.top().first;
                    topResults.pop();
                }
            }
        }

        return {ground_truth, ground_truth_distances};
    }
"""

old_dist_start = "    std::pair<Eigen::MatrixXi, Eigen::MatrixXf> compute_ground_truth_batch_parallel4_with_dist("
old_dist_end = "    }\n\n    void update_ground_truth_with_new_data("

idx3 = content.find(old_dist_start)
idx4 = content.find(old_dist_end)

if idx3 != -1 and idx4 != -1:
    content = content[:idx3] + new_compute_gt_dist[1:] + "\n" + content[idx4 + 6:]
else:
    print("Could not find compute_ground_truth_batch_parallel4_with_dist!")


with open('../hnswlib/adaptive_ef.h', 'w') as f:
    f.write(content)
