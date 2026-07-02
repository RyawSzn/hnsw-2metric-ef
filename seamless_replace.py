import re

# ==========================================
# 1. hnswlib/hnswalg.h
# ==========================================
with open('hnswlib/hnswalg.h', 'r') as f:
    hnswalg = f.read()

# Fix adaptiveSearchBaseLayerST signature
hnswalg = hnswalg.replace(
    "hnswdis::Sketch* sketch,\n        dist_t d_ep = 0,\n        BaseFilterFunctor* isIdAllowed = nullptr",
    "hnswdis::Sketch* sketch,\n        float* out_cv = nullptr,\n        BaseFilterFunctor* isIdAllowed = nullptr"
)

# Fix adaptiveSearchBaseLayerST2 signature
hnswalg = hnswalg.replace(
    "hnswdis::Sketch* sketch,\n        dist_t d_ep = 0,\n        BaseFilterFunctor* isIdAllowed = nullptr",
    "hnswdis::Sketch* sketch,\n        float* out_cv = nullptr,\n        BaseFilterFunctor* isIdAllowed = nullptr"
)

# Fix the internal calculation (twice, for ST and ST2)
old_calc1 = """                        score = score_calculator.compute_score(data_point, *((size_t *) dist_func_param_), edge_evals.data(), edge_evals.size());
                        if (sketch) {
                            ef = sketch->estimate_ef2(score, d_ep); // used for estimating ef"""

new_calc1 = """                        score = score_calculator.compute_score(data_point, *((size_t *) dist_func_param_), edge_evals.data(), edge_evals.size());
                        float cv = 0.0f;
                        if (top_candidates.size() > 1) {
                            auto temp_q = top_candidates;
                            float sum = 0.0f, sum_sq = 0.0f;
                            int q_size = temp_q.size();
                            while (!temp_q.empty()) {
                                float d = temp_q.top().first;
                                sum += d;
                                sum_sq += d * d;
                                temp_q.pop();
                            }
                            float mean = sum / q_size;
                            if (mean > 1e-6f) {
                                float variance = std::max(0.0f, (sum_sq / q_size) - (mean * mean));
                                cv = std::sqrt(variance) / mean;
                            }
                        }
                        if (out_cv) *out_cv = cv;
                        if (sketch) {
                            ef = sketch->estimate_ef2(score, cv); // used for estimating ef"""
hnswalg = hnswalg.replace(old_calc1, new_calc1)

old_calc2 = """                        score = score_calculator.compute_score(data_point, *((size_t *) dist_func_param_), edge_evals.data(), edge_evals.size());
                        if (sketch) {
                            ef = sketch->estimate_ef2(score, d_ep);  // get estimated ef"""

new_calc2 = """                        score = score_calculator.compute_score(data_point, *((size_t *) dist_func_param_), edge_evals.data(), edge_evals.size());
                        float cv = 0.0f;
                        if (top_candidates.size() > 1) {
                            auto temp_q = top_candidates;
                            float sum = 0.0f, sum_sq = 0.0f;
                            int q_size = temp_q.size();
                            while (!temp_q.empty()) {
                                float d = temp_q.top().first;
                                sum += d;
                                sum_sq += d * d;
                                temp_q.pop();
                            }
                            float mean = sum / q_size;
                            if (mean > 1e-6f) {
                                float variance = std::max(0.0f, (sum_sq / q_size) - (mean * mean));
                                cv = std::sqrt(variance) / mean;
                            }
                        }
                        if (out_cv) *out_cv = cv;
                        if (sketch) {
                            ef = sketch->estimate_ef2(score, cv);  // get estimated ef"""
hnswalg = hnswalg.replace(old_calc2, new_calc2)

# Fix adaptiveSearchKnn signature
hnswalg = hnswalg.replace(
    "hnswdis::Sketch* sketch = nullptr,\n        BaseFilterFunctor* isIdAllowed = nullptr,\n        bool print_ef = false",
    "hnswdis::Sketch* sketch = nullptr,\n        float* out_cv = nullptr,\n        BaseFilterFunctor* isIdAllowed = nullptr,\n        bool print_ef = false"
)
hnswalg = hnswalg.replace(
    "score_calculator, sketch, curdist, isIdAllowed);",
    "score_calculator, sketch, out_cv, isIdAllowed);"
)

# Fix adaptiveSearchKnnTest signature
hnswalg = hnswalg.replace(
    "hnswdis::Sketch* sketch = nullptr,\n        BaseFilterFunctor* isIdAllowed = nullptr) const {",
    "hnswdis::Sketch* sketch = nullptr,\n        float* out_cv = nullptr,\n        BaseFilterFunctor* isIdAllowed = nullptr) const {"
)

with open('hnswlib/hnswalg.h', 'w') as f:
    f.write(hnswalg)

# ==========================================
# 2. hnswlib/sketch.h
# ==========================================
with open('hnswlib/sketch.h', 'r') as f:
    sketch = f.read()

sketch = sketch.replace("d_ep", "cv")
sketch = sketch.replace("dep_centers", "cv_centers")
sketch = sketch.replace("dep_tables", "cv_tables")

with open('hnswlib/sketch.h', 'w') as f:
    f.write(sketch)

# ==========================================
# 3. hnswlib/adaptive_ef.h
# ==========================================
with open('hnswlib/adaptive_ef.h', 'r') as f:
    adaptive = f.read()

old_collect = """        static std::vector<float> collect_dep(
            const hnswlib::HierarchicalNSW<float> &alg_hnsw,
            const MatrixXf &query_vectors)
        {
            std::vector<float> deps;
            deps.reserve(query_vectors.rows());
            for (int i = 0; i < query_vectors.rows(); ++i)
                deps.push_back(alg_hnsw.computeEntryPointDistance(query_vectors.row(i).data()));
            return deps;
        }"""
new_collect = """        static std::vector<float> collect_cv(
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
        }"""
adaptive = adaptive.replace(old_collect, new_collect)

# Rename n_dep_tables, dep_centers, dep_tables
adaptive = adaptive.replace("n_dep", "n_cv")
adaptive = adaptive.replace("dep_centers", "cv_centers")
adaptive = adaptive.replace("dep_tables", "cv_tables")
adaptive = adaptive.replace("deps[", "cvs[")
adaptive = adaptive.replace("dep-bucket", "cv-bucket")

# Update EfAdapter constructor signature default value
adaptive = adaptive.replace("int n_cv_tables = 5", "int n_cv_tables = 5") 

# Update init_with_dep_buckets
adaptive = adaptive.replace("void init_with_dep_buckets(", "void init_with_cv_buckets(")
adaptive = adaptive.replace("init_with_dep_buckets(", "init_with_cv_buckets(")

adaptive = adaptive.replace("std::vector<float> deps = collect_dep(*alg_hnsw, *query_vectors);", 
                            "std::vector<float> cvs = collect_cv(*alg_hnsw, *query_vectors, score_cal, k, statics_length);")

# Serialization
adaptive = adaptive.replace("writeBinaryPOD(out, n_dep);", "writeBinaryPOD(out, n_cv);")
adaptive = adaptive.replace("readBinaryPOD(in, n_dep);", "readBinaryPOD(in, n_cv);")

# Replace get_dep_...
adaptive = adaptive.replace("get_dep_centers", "get_cv_centers")
adaptive = adaptive.replace("get_dep_tables", "get_cv_tables")

with open('hnswlib/adaptive_ef.h', 'w') as f:
    f.write(adaptive)

# ==========================================
# 4. experiments_driver/run.cpp & util.h
# ==========================================
with open('experiments_driver/run.cpp', 'r') as f:
    run = f.read()

run = run.replace("get_dep_centers", "get_cv_centers")
run = run.replace("get_dep_tables", "get_cv_tables")

with open('experiments_driver/run.cpp', 'w') as f:
    f.write(run)

with open('experiments_driver/util.h', 'r') as f:
    util = f.read()

util = util.replace("get_dep_centers", "get_cv_centers")
util = util.replace("get_dep_tables", "get_cv_tables")

with open('experiments_driver/util.h', 'w') as f:
    f.write(util)

print("Seamless replacement complete.")
