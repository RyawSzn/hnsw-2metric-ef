#include <map>
// ada-ef
#pragma once

#include "estimator.h"
#include "hnswalg.h"
#include <omp.h>
#include <thread>

namespace hnswdis
{
    std::vector<std::pair<std::vector<size_t>, float>> hnsw_search_and_score(
        const hnswlib::HierarchicalNSW<float> &alg_hnsw,
        const hnswdis::MatrixXf &query_vectors,
        const hnswdis::MatrixXf &data_vectors,
        const hnswdis::ApproximatedScoreCalculator &score_cal,
        const size_t k,
        const size_t statics_length)
    {
        // size_t statics_length = 1 + 32 + 31 * 32; // 2-hop neighbors on the base layer

        std::vector<std::pair<std::vector<size_t>, float>> result;
        result.reserve(query_vectors.rows());

        for (int i = 0; i < query_vectors.rows(); ++i)
        {
            auto ret = alg_hnsw.adaptiveSearchKnn(
                query_vectors.row(i).data(), k, statics_length, score_cal);

            auto &pq = ret.first;
            size_t count = pq.size();
            std::vector<size_t> labels(count);
            while (!pq.empty())
            {
                labels[--count] = pq.top().second;
                pq.pop();
            }
            result.emplace_back(std::move(labels), std::move(ret.second));
        }

        return result;
    }

    std::vector<std::pair<std::vector<size_t>, float>> hnsw_search_and_score(
        const hnswlib::HierarchicalNSW<float> &alg_hnsw,
        const hnswdis::MatrixXf &query_vectors,
        const hnswdis::MatrixXf &data_vectors,
        const std::string &metric,
        const size_t k,
        const float truncation_ratio,
        const size_t statics_length)
    {
        hnswdis::ApproximatedScoreCalculator score_cal(truncation_ratio);

        return hnsw_search_and_score(alg_hnsw, query_vectors, data_vectors, score_cal, k, statics_length);
    }

    std::vector<std::vector<size_t>> hnsw_search(
        const hnswlib::HierarchicalNSW<float> &alg_hnsw,
        const hnswdis::MatrixXf &query_vectors,
        const size_t k)
    {
        std::vector<std::vector<size_t>> result;
        result.reserve(query_vectors.rows());
        for (int i = 0; i < query_vectors.rows(); i++)
        {
            auto ret = alg_hnsw.searchKnn(
                query_vectors.row(i).data(), k); // transforming to clser first

            size_t count = ret.size();
            std::vector<size_t> labels(count);
            while (!ret.empty())
            {
                labels[--count] = ret.top().second;
                ret.pop();
            }

            result.push_back(std::move(labels));
        }
        return result;
    }

    std::shared_ptr<hnswlib::SpaceInterface<float>> init_space(const std::string &metric, const int dim)
    {
        // std::cout << "Metric: " << metric << ", Dimension: " << dim << std::endl;

        std::shared_ptr<hnswlib::SpaceInterface<float>> space;
        if (metric == "l2")
        {
            space = std::make_shared<hnswlib::L2Space>(dim);
        }
        else if (metric == "cd" || metric == "ipd")
        {
            // cosine distance
            space = std::make_shared<hnswlib::InnerProductSpace>(dim);
        }
        else
        {
            throw std::runtime_error("Unsupported metric: " + metric);
        }
        return std::move(space);
    }

    // Multithreaded executor
    // The helper function copied from python_bindings/bindings.cpp (and that itself is copied from nmslib)
    // An alternative is using #pragme omp parallel for or any other C++ threading
    template <class Function>
    inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn)
    {
        if (numThreads <= 0)
        {
            numThreads = std::thread::hardware_concurrency();
        }

        if (numThreads == 1)
        {
            for (size_t id = start; id < end; id++)
            {
                fn(id, 0);
            }
        }
        else
        {
            std::vector<std::thread> threads;
            std::atomic<size_t> current(start);

            // keep track of exceptions in threads
            // https://stackoverflow.com/a/32428427/1713196
            std::exception_ptr lastException = nullptr;
            std::mutex lastExceptMutex;

            for (size_t threadId = 0; threadId < numThreads; ++threadId)
            {
                threads.push_back(std::thread([&, threadId]
                                              {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                } }));
            }
            for (auto &thread : threads)
            {
                thread.join();
            }
            if (lastException)
            {
                std::rethrow_exception(lastException);
            }
        }
    }

    MatrixXi compute_ground_truth(const hnswdis::MatrixXf &query_vectors,
                                  const hnswdis::MatrixXf &data_vectors,
                                  const std::string &metric,
                                  const int k)
    {
        const size_t num_t = std::max(1u, std::thread::hardware_concurrency() / 4);

        std::cout << "Using " << num_t << " threads" << std::endl;

        std::shared_ptr<hnswlib::SpaceInterface<float>> space = init_space(metric, data_vectors.cols());

        std::cout << "Building brute force index" << std::endl;
        hnswlib::BruteforceSearch<float> bf(space.get(), data_vectors.rows());

        // multi-threaded
        ParallelFor(0, data_vectors.rows(), num_t, [&](size_t i, size_t threadId)
                    { bf.addPoint(data_vectors.row(i).data(), i); });

        std::cout << "Brute force index built" << std::endl;

        std::cout << "Computing ground truth" << std::endl;

        std::vector<std::vector<int>> global_results;
        global_results.resize(query_vectors.rows());

        // multi-threaded
        ParallelFor(0, query_vectors.rows(), num_t, [&](size_t i, size_t threadId)
                    {
                        std::vector<int> local_results(k);
                        std::priority_queue<std::pair<float, size_t>> ret = bf.searchKnn(query_vectors.row(i).data(), k);
                        size_t count = k;
                        while (!ret.empty())
                        {
                            local_results[--count] = ret.top().second;
                            ret.pop();
                        }
                        global_results[i] = std::move(local_results); });

        MatrixXi ground_truth(query_vectors.rows(), k);
        for (size_t i = 0; i < query_vectors.rows(); i++)
        {
            ground_truth.row(i) = Eigen::VectorXi::Map(global_results[i].data(), k);
        }

        return ground_truth;
    }

    // This function is a parallelized version of the compute_ground_truth function
    // But it does not fully leverage the power of multi-threading. Sometimes it does sometimes it does not
    MatrixXi compute_ground_truth_batch_parallel2(
        const MatrixXf &query_vectors,
        const MatrixXf &data_vectors,
        const std::string &metric,
        const int k)
    {

        std::size_t numThreads = std::max(1u, std::thread::hardware_concurrency() / 4);

        MatrixXi ground_truth(query_vectors.rows(), k);

        // This ensures each thread gets roughly (rows / numThreads) queries
        size_t totalQueries = query_vectors.rows();
        size_t num_elements = data_vectors.rows();

        const float *data_ptr = data_vectors.data();
        const float *query_ptr = query_vectors.data();
        int *ground_truth_ptr = ground_truth.data();

        size_t dim = data_vectors.cols();
        auto space = init_space(metric, dim);
        auto fstdistfunc_ = space->get_dist_func();
        auto dist_func_param_ = space->get_dist_func_param();

        std::vector<std::thread> threads;
        threads.reserve(numThreads);
        std::atomic<size_t> current(0);

        std::exception_ptr lastException = nullptr;
        std::mutex exceptMutex;

        for (std::size_t t = 0; t < numThreads; t++)
        {
            threads.emplace_back([&, t]
                                 {
            std::priority_queue<std::pair<float, size_t>> topResults;
            while (true)
            {
                size_t query_i = current.fetch_add(1);
                if (query_i >= totalQueries)
                {
                    break;
                }
                try
                {
                const float *local_query_ptr = query_ptr + query_i * dim;
                int *local_ground_truth_ptr = ground_truth_ptr + query_i * k;

                for (size_t data_i = 0; data_i < k; ++data_i)
                {
                    float dist = fstdistfunc_(local_query_ptr, data_ptr + data_i * dim, dist_func_param_);
                    topResults.emplace(dist, data_i);
                }
                float lastdist = topResults.top().first;
                for (size_t data_i = k; data_i < num_elements; ++data_i)
                {
                    float dist = fstdistfunc_(local_query_ptr, data_ptr + data_i * dim, dist_func_param_);
                    if (dist <= lastdist)
                    {
                        topResults.emplace(dist, data_i);
                        if (topResults.size() > k)
                            topResults.pop();
                        lastdist = topResults.top().first;
                    }
                }

                size_t count = k;
                while (!topResults.empty())
                {
                    local_ground_truth_ptr[--count] = static_cast<int>(topResults.top().second);
                    topResults.pop();
                }

                }
                catch (...)
                {
                    // If something goes wrong, store the exception
                    std::unique_lock<std::mutex> lock(exceptMutex);
                    lastException = std::current_exception();
                }
            } });
        }

        // 5) Join threads
        for (auto &th : threads)
        {
            th.join();
        }

        // 6) If any thread threw an exception, rethrow here
        if (lastException)
        {
            std::rethrow_exception(lastException);
        }

        return ground_truth;
    }

    // This function is a parallelized version of the compute_ground_truth function
    // But it does not fully leverage the power of multi-threading. Sometimes it does sometimes it does not
    MatrixXi compute_ground_truth_batch_parallel3(
        const MatrixXf &query_vectors,
        const MatrixXf &data_vectors,
        const std::string &metric,
        const int k)
    {
        size_t totalQueries = query_vectors.rows();
        size_t num_elements = data_vectors.rows();
        size_t dim = data_vectors.cols();

        MatrixXi ground_truth(totalQueries, k);

        const float *data_ptr = data_vectors.data();
        const float *query_ptr = query_vectors.data();
        int *ground_truth_ptr = ground_truth.data();

        int numThreads = std::max(1u, std::thread::hardware_concurrency() / 4);
        omp_set_num_threads(numThreads);

        auto space = init_space(metric, dim);
        auto fstdistfunc_ = space->get_dist_func();
        auto dist_func_param_ = space->get_dist_func_param();

#pragma omp parallel for schedule(static)
        for (int query_i = 0; query_i < static_cast<int>(totalQueries); ++query_i)
        {
            std::priority_queue<std::pair<float, size_t>> topResults;

            const float *local_query_ptr = query_ptr + query_i * dim;
            int *local_ground_truth_ptr = ground_truth_ptr + query_i * k;

            for (size_t data_i = 0; data_i < k; ++data_i)
            {
                float dist = fstdistfunc_(local_query_ptr, data_ptr + data_i * dim, dist_func_param_);
                topResults.emplace(dist, data_i);
            }

            float lastdist = topResults.top().first;

            for (size_t data_i = k; data_i < num_elements; ++data_i)
            {
                float dist = fstdistfunc_(local_query_ptr, data_ptr + data_i * dim, dist_func_param_);
                if (dist <= lastdist)
                {
                    topResults.emplace(dist, data_i);
                    if (topResults.size() > k)
                        topResults.pop();
                    lastdist = topResults.top().first;
                }
            }

            size_t count = k;
            while (!topResults.empty())
            {
                local_ground_truth_ptr[--count] = static_cast<int>(topResults.top().second);
                topResults.pop();
            }
        }

        return ground_truth;
    }

    // This function is a parallelized version of the compute_ground_truth function
    // It is able to fully leverage the power of multi-threading by using Eigen's matrix multiplication.
    // But if the number of queries is large, it will consume a lot of memory. In this case, cutting the data into batches is recommended.
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

        return ground_truth;
    }


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


    void update_ground_truth_with_new_data(
        const MatrixXf &query_vectors,
        MatrixXi &ground_truth,
        MatrixXf &ground_truth_distances,
        const MatrixXf &updates_data,
        const int start_index)
    {

        const int n_queries = query_vectors.rows();
        const int k = ground_truth.cols();
        const int num_updates = updates_data.rows();

        int numThreads = std::max(1u, std::thread::hardware_concurrency() / 4);
        omp_set_num_threads(numThreads);

        int *ground_truth_ptr = ground_truth.data();
        float *distances_ptr = ground_truth_distances.data();

#pragma omp parallel for schedule(static)
        for (int query_i = 0; query_i < n_queries; ++query_i)
        {
            // priority queue: max-heap for top-k (distance, id)
            std::priority_queue<std::pair<float, int>> topResults;

            const RowVectorXf query = query_vectors.row(query_i);

            // Load existing top-k
            int *local_gt = ground_truth_ptr + query_i * k;
            float *local_dist = distances_ptr + query_i * k;
            for (int i = 0; i < k; ++i)
            {
                topResults.emplace(local_dist[i], local_gt[i]);
            }

            // Compare with new data
            for (int j = 0; j < num_updates; ++j)
            {
                float sim = query.dot(updates_data.row(j));
                float dist = 1.0f - sim;

                topResults.emplace(dist, j + start_index);
                if (topResults.size() > k)
                    topResults.pop();
            }

            // Write back new top-k
            for (int i = k - 1; i >= 0; --i)
            {
                local_gt[i] = topResults.top().second;
                local_dist[i] = topResults.top().first;
                topResults.pop();
            }
        }
    }

    // functions used for deletion experiments
    struct QueryHeap
    {
        std::vector<std::pair<float, int>> distances; // sorted ascending
    };
    std::vector<QueryHeap> build_full_gt_structure(
        const MatrixXf &queries,
        const MatrixXf &data)
    {
        const int n_queries = queries.rows();
        const int n_data = data.rows();
        std::vector<QueryHeap> gt(n_queries);

#pragma omp parallel for schedule(static)
        for (int i = 0; i < n_queries; ++i)
        {
            const RowVectorXf query = queries.row(i);
            auto &qh = gt[i];
            qh.distances.reserve(n_data);

            for (int j = 0; j < n_data; ++j)
            {
                float sim = query.dot(data.row(j));
                float dist = 1.0f - sim;
                qh.distances.emplace_back(dist, j);
            }
            std::sort(qh.distances.begin(), qh.distances.end());
        }
        return gt;
    }
    void export_topk(
        const std::vector<QueryHeap> &gt,
        int k,
        MatrixXi &ground_truth,
        MatrixXf &ground_truth_distances)
    {
        const int n_queries = gt.size();
        ground_truth.resize(n_queries, k);
        ground_truth_distances.resize(n_queries, k);

#pragma omp parallel for schedule(static)
        for (int i = 0; i < n_queries; ++i)
        {
            for (int j = 0; j < k; ++j)
            {
                ground_truth(i, j) = gt[i].distances[j].second;
                ground_truth_distances(i, j) = gt[i].distances[j].first;
            }
        }
    }
    void gt_delete_points(std::vector<QueryHeap> &gt,
                          const std::unordered_set<int> &deleted,
                          bool use_copy = false)
    {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < gt.size(); ++i)
        {
            auto &distances = gt[i].distances;

            if (use_copy)
            {
                // Single-pass copy version
                std::vector<std::pair<float, int>> new_vec;
                new_vec.reserve(distances.size());
                for (auto &p : distances)
                    if (!deleted.count(p.second))
                        new_vec.push_back(p);
                distances.swap(new_vec);
            }
            else
            {
                // In-place compaction version
                int write = 0;
                for (int read = 0; read < (int)distances.size(); ++read)
                    if (!deleted.count(distances[read].second))
                    {
                        if (write != read)
                            distances[write] = std::move(distances[read]);
                        ++write;
                    }
                distances.resize(write);
            }
        }
    }
    // ---- Serialization ----
    void save_query_heaps(const std::vector<QueryHeap> &heaps, const std::string &filename)
    {
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs)
            throw std::runtime_error("Cannot open file for writing: " + filename);

        size_t n_queries = heaps.size();
        ofs.write(reinterpret_cast<const char *>(&n_queries), sizeof(size_t));

        for (const auto &qh : heaps)
        {
            size_t n = qh.distances.size();
            ofs.write(reinterpret_cast<const char *>(&n), sizeof(size_t));
            if (n > 0)
                ofs.write(reinterpret_cast<const char *>(qh.distances.data()), n * sizeof(std::pair<float, int>));
        }
        ofs.close();
    }
    // ---- Deserialization ----
    std::vector<QueryHeap> load_query_heaps(const std::string &filename)
    {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs)
            throw std::runtime_error("Cannot open file for reading: " + filename);

        size_t n_queries = 0;
        ifs.read(reinterpret_cast<char *>(&n_queries), sizeof(size_t));

        std::vector<QueryHeap> heaps(n_queries);

        for (size_t i = 0; i < n_queries; ++i)
        {
            size_t n = 0;
            ifs.read(reinterpret_cast<char *>(&n), sizeof(size_t));
            heaps[i].distances.resize(n);
            if (n > 0)
                ifs.read(reinterpret_cast<char *>(heaps[i].distances.data()), n * sizeof(std::pair<float, int>));
        }
        ifs.close();
        return heaps;
    }

    std::vector<float> compute_recall(const hnswdis::MatrixXi &ground_truth, const std::vector<std::vector<size_t>> &search_result, const size_t k, bool verbose = true)
    {
        std::vector<float> recalls;
        recalls.resize(search_result.size());

        // Compute recall for each query
        // size_t k = search_result[0].size(); // Assuming all queries have the same k

        for (int i = 0; i < search_result.size(); ++i)
        {
            const hnswdis::RowVectorXi gt = ground_truth.row(i);

            std::unordered_set<int> get_set;
            // Insert ground truth into the set, only the first k elements
            for (int j = 0; j < k; ++j)
            {
                get_set.insert(gt(j));
            }

            int correct = 0;

            const auto &sr = search_result[i];

            for (const auto &item : sr)
            {
                if (get_set.count(item))
                {
                    correct++;
                }
            }
            recalls[i] = static_cast<float>(correct) / static_cast<float>(k);
        }

        if (verbose)
        {
            std::unordered_map<int, int> recall_distribution;
            for (const auto &recall : recalls)
            {
                int recall_key = static_cast<int>(recall * 100);
                recall_distribution[recall_key]++;
            }

            std::vector<std::pair<int, int>> sorted_recall_distribution(recall_distribution.begin(), recall_distribution.end());
            std::sort(sorted_recall_distribution.begin(), sorted_recall_distribution.end());

            std::cout << "Sorted Recall Distribution:" << std::endl;
            for (const auto &[key, value] : sorted_recall_distribution)
            {
                std::cout << "Recall: " << key << "%, Count: " << value << std::endl;
            }
        }

        return recalls;
    }

    std::shared_ptr<hnswdis::MatrixXf> sample_data(const hnswdis::MatrixXf &data_vectors, size_t sample_size)
    {
        // Random sampling query vectors
        std::mt19937 gen;
        gen.seed(123456789); // Set seed for reproducibility

        std::vector<int> indices(sample_size);

        std::uniform_int_distribution<> dis(0, data_vectors.rows());
        std::unordered_set<int> unique_indices;
        while (unique_indices.size() < sample_size)
        {
            unique_indices.insert(dis(gen));
        }
        std::copy(unique_indices.begin(), unique_indices.end(), indices.begin());

        std::shared_ptr<hnswdis::MatrixXf> query_vectors = std::make_shared<hnswdis::MatrixXf>(sample_size, data_vectors.cols());
        for (size_t i = 0; i < sample_size; ++i)
        {
            query_vectors->row(i) = data_vectors.row(indices[i]);
        }

        return query_vectors;
    }

    class RecallEstimator
    {
    private:
        std::vector<std::tuple<int, float, float, float, float, size_t>> recall_statistics;
        std::unordered_map<int, std::vector<int>> score_to_query_map;

        void init(const std::shared_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw,
                  const std::shared_ptr<hnswdis::MatrixXf> data_vectors,
                  const std::shared_ptr<hnswdis::MatrixXf> query_vectors,
                  const std::shared_ptr<hnswdis::MatrixXi> ground_truth,
                  const size_t k,
                  const hnswdis::ApproximatedScoreCalculator &score_cal,
                  const size_t ef,
                  const size_t statics_length,
                  const bool verbose = false)
        {
            alg_hnsw->setEf(ef);

            std::vector<std::pair<std::vector<size_t>, float>> search_score_result = hnsw_search_and_score(*alg_hnsw, *query_vectors, *data_vectors, score_cal, k, statics_length);

            std::vector<float> score_list;
            score_list.reserve(query_vectors->rows());

            std::vector<std::vector<size_t>> labels;
            labels.reserve(query_vectors->rows());

            for (const auto &r : search_score_result)
            {
                labels.push_back(std::move(r.first));
                score_list.push_back(r.second);
            }

            std::vector<float> recalls_ori = compute_recall(*ground_truth, labels, k, false);

            // Remove zero recalls, which are not useful for statistics
            std::vector<float> recalls;
            for (size_t i = 0; i < recalls_ori.size(); ++i)
            {
                if (recalls_ori[i] < 1e-3f)
                {
                    continue;
                }
                recalls.push_back(recalls_ori[i]);
            }

            float average_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0f) / recalls.size();
            std::cout << "Average Recall: " << average_recall << std::endl;

            std::unordered_map<int, std::vector<float>> grouped_recalls;
            for (size_t i = 0; i < score_list.size(); ++i)
            {
                int score_key = static_cast<int>(score_list[i]);
                grouped_recalls[score_key].push_back(recalls[i]);

                score_to_query_map[score_key].push_back(i); // for tracking the score group of queries
            }

            // Compute statistics for each score group
            for (auto &[key, value] : grouped_recalls)
            {
                float sum = std::accumulate(value.begin(), value.end(), 0.0f);
                float average = sum / value.size();

                size_t size = value.size();
                std::sort(value.begin(), value.end());

                float median = 0.0f;
                if (size % 2 == 0)
                {
                    median = (value[size / 2 - 1] + value[size / 2]) / 2.0f;
                }
                else
                {
                    median = value[size / 2];
                }
                float percentile_25 = value[static_cast<size_t>(0.25 * size)];
                float percentile_5 = value[static_cast<size_t>(0.05 * size)];

                recall_statistics.push_back({key, average, median, percentile_25, percentile_5, value.size()});
            }

            // Sort recall_statistics by score
            std::sort(recall_statistics.begin(), recall_statistics.end(), [](const auto &a, const auto &b)
                      { return std::get<0>(a) < std::get<0>(b); });

            // Print recall statistics
            if (verbose)
            {
                for (const auto &stat : recall_statistics)
                {
                    std::cout << "Score: " << std::get<0>(stat)
                              << ", Average Recall: " << std::get<1>(stat)
                              << ", Median Recall: " << std::get<2>(stat)
                              << ", 25th Percentile Recall: " << std::get<3>(stat)
                              << ", 5th Percentile Recall: " << std::get<4>(stat)
                              << ", Count: " << std::get<5>(stat)
                              << std::endl;
                }
            }
        }

    public:
        RecallEstimator(
            std::shared_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw,
            std::shared_ptr<hnswdis::MatrixXf> data_vectors,
            size_t sample_size,
            size_t k,
            const std::string &metric,
            const size_t ef,
            const float truncation_ratio,
            const size_t statics_length)
        {
            std::shared_ptr<hnswdis::MatrixXf> query_vectors = hnswdis::sample_data(*data_vectors, sample_size);
            MatrixXi ground_truth = compute_ground_truth_batch_parallel4(*query_vectors, *data_vectors, metric, k);

            hnswdis::ApproximatedScoreCalculator score_cal(truncation_ratio);

            init(alg_hnsw,
                 data_vectors,
                 query_vectors,
                 std::make_shared<hnswdis::MatrixXi>(ground_truth),
                 k,
                 score_cal,
                 ef,
                 statics_length);
        }

        RecallEstimator(
            std::shared_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw,
            std::shared_ptr<hnswdis::MatrixXf> data_vectors,
            std::shared_ptr<hnswdis::MatrixXf> query_vectors,
            std::shared_ptr<hnswdis::MatrixXi> ground_truth,
            size_t k,
            hnswdis::ApproximatedScoreCalculator &score_cal,
            const size_t ef,
            const size_t statics_length)
        {
            init(alg_hnsw,
                 data_vectors,
                 query_vectors,
                 ground_truth,
                 k,
                 score_cal,
                 ef,
                 statics_length);
        }

        RecallEstimator(
            const std::string &filename)
        {
            load_statistics(filename);
        }

        const std::unordered_map<int, std::vector<int>> &get_score_to_query_map() const
        {
            return score_to_query_map;
        }

        float estimate_recall(float score)
        {
            auto last = recall_statistics.back();

            if (score > std::get<0>(last))
            {
                return std::get<1>(last);
            }

            auto first = recall_statistics.front();

            if (score < std::get<0>(first))
            {
                return std::get<1>(first);
            }

            auto it = std::lower_bound(recall_statistics.begin(), recall_statistics.end(), score, [](const auto &a, const float &b)
                                       { return std::get<0>(a) < b; });

            if (it == recall_statistics.end())
            {
                return 0.0f;
            }

            return std::get<1>(*it); // 25th percentile
        }

        void save_satistics(const std::string &filename)
        {
            std::ofstream file(filename);
            file << "Score,Average Recall,Median Recall,25th Percentile Recall,5th Percentile Recall,Count\n";
            for (const auto &stat : recall_statistics)
            {
                file << std::get<0>(stat) << ","
                     << std::get<1>(stat) << ","
                     << std::get<2>(stat) << ","
                     << std::get<3>(stat) << ","
                     << std::get<4>(stat) << ","
                     << std::get<5>(stat) << "\n";
            }
            file.close();
        }

        void load_statistics(const std::string &filename)
        {
            std::cout << "Loading statistics from " << filename << std::endl;
            std::ifstream file(filename);
            std::string line;
            std::getline(file, line); // skip header
            while (std::getline(file, line))
            {
                std::istringstream iss(line);
                std::string token;
                std::vector<std::string> tokens;
                while (std::getline(iss, token, ','))
                {
                    tokens.push_back(token);
                }
                recall_statistics.push_back({std::stoi(tokens[0]), std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]), std::stof(tokens[4]), std::stoi(tokens[5])});
            }
            file.close();
        }

        const std::vector<std::tuple<int, float, float, float, float, size_t>> &get_recall_statistics() const
        {
            return recall_statistics;
        }
    };

    std::pair<MatrixXf, MatrixXi> compute_samplings(
        std::shared_ptr<hnswdis::MatrixXf> data_vectors,
        const std::string &metric,
        const int k,
        const size_t sample_size)
    {
        std::shared_ptr<hnswdis::MatrixXf> sample_query_vectors = hnswdis::sample_data(*data_vectors, sample_size);
        MatrixXi sample_ground_truth = compute_ground_truth_batch_parallel4(*sample_query_vectors, *data_vectors, metric, k);
        return {*sample_query_vectors, sample_ground_truth};
    }

    void serialize_samplings(const std::string &filename,
                             const MatrixXf &queries,
                             const MatrixXi &groundtruth)
    {
        std::ofstream out(filename, std::ios::binary);
        if (!out)
            throw std::runtime_error("Failed to open file for writing");

        // Save queries
        int rows = queries.rows();
        int cols = queries.cols();
        out.write(reinterpret_cast<const char *>(&rows), sizeof(int));
        out.write(reinterpret_cast<const char *>(&cols), sizeof(int));
        out.write(reinterpret_cast<const char *>(queries.data()), sizeof(float) * rows * cols);

        // Save ground truth
        rows = groundtruth.rows();
        cols = groundtruth.cols();
        out.write(reinterpret_cast<const char *>(&rows), sizeof(int));
        out.write(reinterpret_cast<const char *>(&cols), sizeof(int));
        out.write(reinterpret_cast<const char *>(groundtruth.data()), sizeof(int) * rows * cols);

        out.close();
    }

    void serialize_samplings(const std::string &filename,
                             const MatrixXf &queries,
                             const MatrixXi &groundtruth,
                             const MatrixXf &distances)
    {
        std::ofstream out(filename, std::ios::binary);
        if (!out)
            throw std::runtime_error("Failed to open file for writing");

        // Save queries
        int rows = queries.rows();
        int cols = queries.cols();
        out.write(reinterpret_cast<const char *>(&rows), sizeof(int));
        out.write(reinterpret_cast<const char *>(&cols), sizeof(int));
        out.write(reinterpret_cast<const char *>(queries.data()), sizeof(float) * rows * cols);

        // Save ground truth
        rows = groundtruth.rows();
        cols = groundtruth.cols();
        out.write(reinterpret_cast<const char *>(&rows), sizeof(int));
        out.write(reinterpret_cast<const char *>(&cols), sizeof(int));
        out.write(reinterpret_cast<const char *>(groundtruth.data()), sizeof(int) * rows * cols);

        // Save distances
        rows = distances.rows();
        cols = distances.cols();
        out.write(reinterpret_cast<const char *>(&rows), sizeof(int));
        out.write(reinterpret_cast<const char *>(&cols), sizeof(int));
        out.write(reinterpret_cast<const char *>(distances.data()), sizeof(float) * rows * cols);

        out.close();
    }

    void deserialize_samplings(const std::string &filename,
                               MatrixXf &queries,
                               MatrixXi &groundtruth)
    {
        std::ifstream in(filename, std::ios::binary);
        if (!in)
            throw std::runtime_error("Failed to open file for reading");

        // Load queries
        int rows, cols;
        in.read(reinterpret_cast<char *>(&rows), sizeof(int));
        in.read(reinterpret_cast<char *>(&cols), sizeof(int));
        queries.resize(rows, cols);
        in.read(reinterpret_cast<char *>(queries.data()), sizeof(float) * rows * cols);

        // Load ground truth
        in.read(reinterpret_cast<char *>(&rows), sizeof(int));
        in.read(reinterpret_cast<char *>(&cols), sizeof(int));
        groundtruth.resize(rows, cols);
        in.read(reinterpret_cast<char *>(groundtruth.data()), sizeof(int) * rows * cols);

        in.close();
    }

    void deserialize_samplings(const std::string &filename,
                               MatrixXf &queries,
                               MatrixXi &groundtruth,
                               MatrixXf &distances)
    {
        std::ifstream in(filename, std::ios::binary);
        if (!in)
            throw std::runtime_error("Failed to open file for reading");

        // Load queries
        int rows, cols;
        in.read(reinterpret_cast<char *>(&rows), sizeof(int));
        in.read(reinterpret_cast<char *>(&cols), sizeof(int));
        queries.resize(rows, cols);
        in.read(reinterpret_cast<char *>(queries.data()), sizeof(float) * rows * cols);

        // Load ground truth
        in.read(reinterpret_cast<char *>(&rows), sizeof(int));
        in.read(reinterpret_cast<char *>(&cols), sizeof(int));
        groundtruth.resize(rows, cols);
        in.read(reinterpret_cast<char *>(groundtruth.data()), sizeof(int) * rows * cols);

        // Load distances
        in.read(reinterpret_cast<char *>(&rows), sizeof(int));
        in.read(reinterpret_cast<char *>(&cols), sizeof(int));
        distances.resize(rows, cols);
        in.read(reinterpret_cast<char *>(distances.data()), sizeof(float) * rows * cols);

        in.close();
    }

    class EfAdapter
    {
    private:
        EfRecallTable ef_recall_estimators;

        std::vector<EfRecallTable> cv_tables;
        std::vector<float>         cv_centers;

        float expected_recall;
        float wae;
        int ef_upper_bound;

        // Collect d_ep for every query by running the greedy upper-layer search only.
        static std::vector<float> collect_cv(
            const hnswlib::HierarchicalNSW<float> &alg_hnsw,
            const MatrixXf &query_vectors,
            const hnswdis::ApproximatedScoreCalculator &score_cal,
            const size_t k,
            const size_t statics_length)
        {
            std::vector<float> cvs;
            cvs.reserve(query_vectors.rows());
            for (int i = 0; i < query_vectors.rows(); ++i) {
                float cv = 0.0f;
                alg_hnsw.adaptiveSearchKnn(query_vectors.row(i).data(), k, statics_length, score_cal, nullptr, &cv);
                cvs.push_back(cv);
            }
            return cvs;
        }

        void init(const std::shared_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw,
                  const std::shared_ptr<hnswdis::MatrixXf> data_vectors,
                  const size_t k,
                  const std::string metric,
                  const float truncation_ratio,
                  const size_t statics_length,
                  const std::shared_ptr<hnswdis::MatrixXf> query_vectors,
                  const std::shared_ptr<hnswdis::MatrixXi> ground_truth_ptr,
                  EfRecallTable &out_table)
        {
            hnswdis::ApproximatedScoreCalculator score_cal(truncation_ratio);

            size_t first_ef = k;
            size_t second_ef = static_cast<size_t>(1.5 * first_ef);

            RecallEstimator first_recall_estimator(alg_hnsw, data_vectors, query_vectors, ground_truth_ptr, k, score_cal, first_ef, statics_length);
            add_ef_recall(first_ef, first_recall_estimator, out_table);
            float first_average_recall = compute_average_recall(first_recall_estimator);
            std::cout << "Initial average recall with ef=" << first_ef << ": " << first_average_recall << std::endl;

            RecallEstimator second_recall_estimator(alg_hnsw, data_vectors, query_vectors, ground_truth_ptr, k, score_cal, second_ef, statics_length);
            add_ef_recall(second_ef, second_recall_estimator, out_table);
            float second_average_recall = compute_average_recall(second_recall_estimator);
            std::cout << "Initial average recall with ef=" << second_ef << ": " << second_average_recall << std::endl;

            auto score_to_query_map = first_recall_estimator.get_score_to_query_map();

            for (size_t i = 0; i < out_table.size(); ++i)
            {
                int score = out_table[i].first;
                auto &ef_recall_list = out_table[i].second;

                int latest_ef = ef_recall_list[1].first;
                float latest_agg_recall = ef_recall_list[1].second;

                int ef_diff = ef_recall_list[1].first - ef_recall_list[0].first;
                float recall_diff = ef_recall_list[1].second - ef_recall_list[0].second;

                auto it = score_to_query_map.find(score);
                if (it != score_to_query_map.end())
                {
                    while (expected_recall - latest_agg_recall > 1e-4f)
                    {
                        ef_diff = std::max((int)(ef_diff * (expected_recall - latest_agg_recall) / recall_diff), (int)(k * 0.5));
                        int ef = latest_ef + ef_diff;

                        if (ef > ef_upper_bound)
                            ef = ef_upper_bound;

                        alg_hnsw->setEf(ef);
                        std::vector<float> recalls_ori;
                        recalls_ori.reserve(it->second.size());

                        for (const auto &query_index : it->second)
                        {
                            auto ret = alg_hnsw->searchKnn(
                                query_vectors->row(query_index).data(), k);
                            size_t count = ret.size();
                            std::vector<size_t> labels(count);
                            while (!ret.empty())
                            {
                                labels[--count] = ret.top().second;
                                ret.pop();
                            }

                            const hnswdis::RowVectorXi gt = ground_truth_ptr->row(query_index);
                            std::unordered_set<int> get_set;
                            for (int j = 0; j < (int)k; ++j)
                                get_set.insert(gt(j));

                            int correct = 0;
                            for (const auto &item : labels)
                                if (get_set.count(item))
                                    correct++;

                            recalls_ori.push_back(static_cast<float>(correct) / k);
                        }

                        std::vector<float> recalls;
                        for (float r : recalls_ori)
                            if (r >= 1e-3f)
                                recalls.push_back(r);

                        float sum = 0.0f;
                        for (float r : recalls) sum += r;
                        float agg_recall = sum / recalls.size();

                        ef_recall_list.push_back({ef, agg_recall});

                        recall_diff = agg_recall - latest_agg_recall;
                        latest_ef = ef;
                        latest_agg_recall = agg_recall;

                        if (recall_diff < 1e-5f)
                        {
                            std::cout << "Recall diff is too small, break." << std::endl;
                            break;
                        }
                        if (latest_ef == ef_upper_bound)
                            break;
                    }
                }
            }

            auto &stat = first_recall_estimator.get_recall_statistics();

            // 1. Extract score to EF and count mapping
            std::map<int, size_t> score_to_ef;
            std::map<int, size_t> score_to_cnt;

            for (int i = 0; i < (int)stat.size(); ++i)
            {
                int score = std::get<0>(stat[i]);
                size_t cnt = std::get<5>(stat[i]);
                size_t ef = out_table[i].second.back().first;
                for (size_t j = 0; j < out_table[i].second.size() - 1; ++j)
                {
                    if (out_table[i].second[j].second >= expected_recall)
                    {
                        ef = out_table[i].second[j].first;
                        break;
                    }
                }
                score_to_ef[score] = ef;
                score_to_cnt[score] = cnt;
            }

            // 2. Identify continuous and discrete parts
            std::vector<int> scores;
            for (auto const& [score, _] : score_to_ef) {
                scores.push_back(score);
            }

            int best_start = 0, best_len = 0;
            int curr_start = 0, curr_len = 1;

            if (scores.size() > 0) {
                for (size_t i = 1; i < scores.size(); ++i) {
                    if (scores[i] == scores[i-1] + 1) {
                        curr_len++;
                    } else {
                        if (curr_len > best_len) {
                            best_len = curr_len;
                            best_start = curr_start;
                        }
                        curr_start = i;
                        curr_len = 1;
                    }
                }
                if (curr_len > best_len) {
                    best_len = curr_len;
                    best_start = curr_start;
                }
            }

            int cont_start_score = 0;
            int cont_end_score = 100;
            if (best_len > 0) {
                cont_start_score = scores[best_start];
                cont_end_score = scores[best_start + best_len - 1];
                if (best_len >= 3) {
                    cont_start_score += 1;
                    cont_end_score -= 1;
                }
            }

            // 3. Compute weighted averages for discrete regions
            float left_sum_ef = 0, right_sum_ef = 0;
            size_t left_sum_cnt = 0, right_sum_cnt = 0;

            for (auto const& [score, ef] : score_to_ef) {
                size_t cnt = score_to_cnt[score];
                if (score < cont_start_score) {
                    left_sum_ef += ef * cnt;
                    left_sum_cnt += cnt;
                } else if (score > cont_end_score) {
                    right_sum_ef += ef * cnt;
                    right_sum_cnt += cnt;
                }
            }

            size_t left_avg_ef = left_sum_cnt > 0 ? std::round(left_sum_ef / left_sum_cnt) : (score_to_ef.empty() ? 0 : score_to_ef.begin()->second);
            size_t right_avg_ef = right_sum_cnt > 0 ? std::round(right_sum_ef / right_sum_cnt) : (score_to_ef.empty() ? 0 : score_to_ef.rbegin()->second);

            // 4. Build smoothed table and calculate new weighted average EF
            EfRecallTable smoothed_table;
            float weighted_average_ef = 0.0f;
            float total_queries = query_vectors->rows();

            for (int s = 0; s <= 100; ++s) {
                size_t final_ef = 0;
                size_t cnt = score_to_cnt.count(s) ? score_to_cnt[s] : 0;

                if (s < cont_start_score) {
                    final_ef = left_avg_ef;
                    smoothed_table.push_back({s, {{(int)final_ef, expected_recall}}});
                } else if (s > cont_end_score) {
                    final_ef = right_avg_ef;
                    smoothed_table.push_back({s, {{(int)final_ef, expected_recall}}});
                } else {
                    if (score_to_ef.count(s)) {
                        final_ef = score_to_ef[s];
                        smoothed_table.push_back({s, {{(int)final_ef, expected_recall}}});
                    }
                }

                if (cnt > 0 && final_ef > 0) {
                    weighted_average_ef += cnt * final_ef / total_queries;
                }
            }

            out_table = smoothed_table;
            std::cout << "Weighted average ef: " << weighted_average_ef << std::endl;
            wae = weighted_average_ef;
        }

        void add_ef_recall(const int ef, const RecallEstimator &recall_estimator, EfRecallTable &out_table)
        {
            const std::vector<std::pair<int, float>> &score_recall = get_score_recall(recall_estimator);
            if (out_table.empty())
            {
                out_table.reserve(score_recall.size());
                for (size_t i = 0; i < score_recall.size(); ++i)
                {
                    std::vector<std::pair<int, float>> ef_recall;
                    ef_recall.push_back({ef, score_recall[i].second});
                    out_table.emplace_back(score_recall[i].first, std::move(ef_recall));
                }
            }
            else
            {
                if (score_recall.size() != out_table.size())
                {
                    throw std::runtime_error("Score recall size mismatch in add_ef_recall.");
                }
                for (size_t i = 0; i < score_recall.size(); ++i)
                {
                    auto &ef_recall = out_table[i].second;
                    ef_recall.emplace_back(ef, score_recall[i].second);
                }
            }
        }

        std::vector<std::pair<int, float>> get_score_recall(const RecallEstimator &recall_estimator)
        {
            auto recall_stat = recall_estimator.get_recall_statistics();

            std::vector<std::pair<int, float>> score_recall;
            score_recall.reserve(recall_stat.size());

            for (size_t i = 0; i < recall_stat.size(); ++i)
            {
                score_recall.push_back({std::get<0>(recall_stat[i]), std::get<1>(recall_stat[i])}); // score, average recall
            }
            return score_recall;
        }

        float compute_average_recall(RecallEstimator &recall_estimator)
        {
            auto recall_stat = recall_estimator.get_recall_statistics();
            float sum = 0.0f;
            size_t count = 0;
            for (const auto &stat : recall_stat)
            {
                sum += std::get<1>(stat) * std::get<5>(stat);
                count += std::get<5>(stat);
            }
            return sum / count;
        }

    public:
        EfAdapter(
            std::shared_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw,
            std::shared_ptr<hnswdis::MatrixXf> data_vectors,
            const size_t k,
            const std::string metric,
            const float expected_recall,
            const float truncation_ratio,
            const size_t statics_length,
            const std::string &samplings_filename,
            int ef_upper_bound = 5000
        ) : expected_recall(expected_recall), ef_upper_bound(ef_upper_bound)
        {

            std::ifstream file(samplings_filename);

            std::shared_ptr<hnswdis::MatrixXf> sample_query_vectors_ptr;
            std::shared_ptr<hnswdis::MatrixXi> sample_ground_truth_ptr;
            if (file.good())
            {
                // Load samplings from file
                std::cout << "Loading samplings from " << samplings_filename << std::endl;
                MatrixXf sample_query_vectors;
                MatrixXi sample_ground_truth;
                deserialize_samplings(samplings_filename, sample_query_vectors, sample_ground_truth);
                std::cout << "Samplings loaded from " << samplings_filename << std::endl;
                sample_query_vectors_ptr = std::make_shared<hnswdis::MatrixXf>(sample_query_vectors);
                sample_ground_truth_ptr = std::make_shared<hnswdis::MatrixXi>(sample_ground_truth);
            }
            else
            {
                // Sample data and compute ground truth
                std::cout << "Sampling data and computing ground truth..." << std::endl;
                auto pair = compute_samplings(data_vectors, metric, k, 2000);
                MatrixXf sample_query_vectors = pair.first;
                MatrixXi sample_ground_truth = pair.second;
                serialize_samplings(samplings_filename, sample_query_vectors, sample_ground_truth);
                std::cout << "Samplings saved to " << samplings_filename << std::endl;
                sample_query_vectors_ptr = std::make_shared<hnswdis::MatrixXf>(sample_query_vectors);
                sample_ground_truth_ptr = std::make_shared<hnswdis::MatrixXi>(sample_ground_truth);
            }

            init(alg_hnsw, data_vectors, k, metric, truncation_ratio, statics_length, sample_query_vectors_ptr, sample_ground_truth_ptr, ef_recall_estimators);
        }

        EfAdapter(
            std::shared_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw,
            std::shared_ptr<hnswdis::MatrixXf> data_vectors,
            const size_t k,
            const std::string metric,
            const float expected_recall,
            const float truncation_ratio,
            const size_t statics_length,
            const std::shared_ptr<hnswdis::MatrixXf> query_vectors,
            const std::shared_ptr<hnswdis::MatrixXi> ground_truth_ptr,
            int ef_upper_bound = 5000) : expected_recall(expected_recall), ef_upper_bound(ef_upper_bound)
        {
            init(alg_hnsw, data_vectors, k, metric, truncation_ratio, statics_length, query_vectors, ground_truth_ptr, ef_recall_estimators);
        }

        EfAdapter(
            const std::string &filename)
        {
            deserialize(filename);
        }

        void init_with_cv_buckets(
            const std::shared_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw,
            const std::shared_ptr<hnswdis::MatrixXf> data_vectors,
            const size_t k,
            const std::string metric,
            const float truncation_ratio,
            const size_t statics_length,
            const std::shared_ptr<hnswdis::MatrixXf> query_vectors,
            const std::shared_ptr<hnswdis::MatrixXi> ground_truth_ptr,
            const int n_cv_tables)
        {
            const int n = query_vectors->rows();

            hnswdis::ApproximatedScoreCalculator score_cal(truncation_ratio);
            std::vector<float> cvs = collect_cv(*alg_hnsw, *query_vectors, score_cal, k, statics_length);

            std::vector<int> order(n);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](int a, int b) { return cvs[a] < cvs[b]; });

            int actual_n_cv_tables = std::max(1, n_cv_tables);
            int chunk = n / actual_n_cv_tables;

            cv_centers.clear();
            for (int t = 0; t < actual_n_cv_tables; ++t) {
                int lo = t * chunk;
                int hi = (t == actual_n_cv_tables - 1) ? n : (t + 1) * chunk;
                cv_centers.push_back(cvs[order[(lo + hi) / 2]]);
            }

            cv_tables.resize(actual_n_cv_tables);

            float accumulated_wae = 0.0f;
            for (int t = 0; t < actual_n_cv_tables; ++t)
            {
                int lo = t * chunk;
                int hi = (t == actual_n_cv_tables - 1) ? n : (t + 1) * chunk;

                int bucket_size = hi - lo;
                MatrixXf bucket_queries(bucket_size, query_vectors->cols());
                MatrixXi bucket_gt(bucket_size, ground_truth_ptr->cols());
                for (int r = 0; r < bucket_size; ++r)
                {
                    bucket_queries.row(r) = query_vectors->row(order[lo + r]);
                    bucket_gt.row(r)      = ground_truth_ptr->row(order[lo + r]);
                }

                std::cout << "Training cv-bucket " << t
                          << " [" << cvs[order[lo]] << ", "
                          << (t < actual_n_cv_tables - 1 ? cvs[order[hi]] : std::numeric_limits<float>::infinity())
                          << ") with " << bucket_size << " queries." << std::endl;

                init(alg_hnsw,
                     data_vectors,
                     k, metric, truncation_ratio, statics_length,
                     std::make_shared<MatrixXf>(bucket_queries),
                     std::make_shared<MatrixXi>(bucket_gt),
                     cv_tables[t]);
                
                accumulated_wae += wae * ((float)bucket_size / n);
            }
            wae = accumulated_wae;
        }

        size_t estimate_ef(float score)
        {
            auto entry = std::lower_bound(ef_recall_estimators.begin(), ef_recall_estimators.end(), score, [](const auto &a, const float &b)
                                          { return a.first < b; });

            if (entry == ef_recall_estimators.begin())
            {
                entry = ef_recall_estimators.begin();
            }
            else if (entry == ef_recall_estimators.end())
            {
                entry = std::prev(ef_recall_estimators.end());
            }

            for (const auto &ef_recall : entry->second)
            {
                if (ef_recall.second >= expected_recall)
                {
                    return ef_recall.first;
                }
            }
            return entry->second.back().first;
        }

        static void write_table(std::ofstream &out, const EfRecallTable &table)
        {
            size_t sz = table.size();
            hnswlib::writeBinaryPOD(out, sz);
            for (const auto &entry : table)
            {
                hnswlib::writeBinaryPOD(out, entry.first);
                size_t recall_size = entry.second.size();
                hnswlib::writeBinaryPOD(out, recall_size);
                for (const auto &p : entry.second)
                {
                    hnswlib::writeBinaryPOD(out, p.first);
                    hnswlib::writeBinaryPOD(out, p.second);
                }
            }
        }

        static void read_table(std::ifstream &in, EfRecallTable &table)
        {
            size_t sz;
            hnswlib::readBinaryPOD(in, sz);
            table.resize(sz);
            for (auto &entry : table)
            {
                hnswlib::readBinaryPOD(in, entry.first);
                size_t recall_size;
                hnswlib::readBinaryPOD(in, recall_size);
                entry.second.resize(recall_size);
                for (auto &p : entry.second)
                {
                    hnswlib::readBinaryPOD(in, p.first);
                    hnswlib::readBinaryPOD(in, p.second);
                }
            }
        }

        void serialize(const std::string &filename) const
        {
            std::ofstream out(filename, std::ios::binary);
            if (!out)
                throw std::runtime_error("Failed to open file for writing: " + filename);

            write_table(out, ef_recall_estimators);

            hnswlib::writeBinaryPOD(out, expected_recall);
            hnswlib::writeBinaryPOD(out, wae);

            size_t n_cv = cv_tables.size();
            hnswlib::writeBinaryPOD(out, n_cv);
            for (const auto &t : cv_tables)
                write_table(out, t);

            size_t n_thresh = cv_centers.size();
            hnswlib::writeBinaryPOD(out, n_thresh);
            for (float v : cv_centers)
                hnswlib::writeBinaryPOD(out, v);

            out.close();
        }

        void deserialize(const std::string &filename)
        {
            std::ifstream in(filename, std::ios::binary);
            if (!in)
                throw std::runtime_error("Failed to open file for reading: " + filename);

            read_table(in, ef_recall_estimators);

            hnswlib::readBinaryPOD(in, expected_recall);
            hnswlib::readBinaryPOD(in, wae);

            size_t n_cv;
            hnswlib::readBinaryPOD(in, n_cv);
            cv_tables.resize(n_cv);
            for (auto &t : cv_tables)
                read_table(in, t);

            size_t n_thresh;
            hnswlib::readBinaryPOD(in, n_thresh);
            cv_centers.resize(n_thresh);
            for (float &v : cv_centers)
                hnswlib::readBinaryPOD(in, v);

            in.close();
        }

        void print() const
        {
            for (const auto &entry : ef_recall_estimators)
            {
                std::cout << "Score: " << entry.first << std::endl;
                for (const auto &ef_recall : entry.second)
                    std::cout << "EF: " << ef_recall.first << ", Recall: " << ef_recall.second << std::endl;
            }
        }

        const EfRecallTable &get_ef_recall_estimators() const { return ef_recall_estimators; }

        const std::vector<EfRecallTable> &get_all_tables() const { return cv_tables; }

        const std::vector<float> &get_cv_centers() const { return cv_centers; }

        bool has_cv_tables() const { return !cv_tables.empty(); }

        float get_expected_recall() const { return expected_recall; }

        float get_wae() const { return wae; }
    };
}
