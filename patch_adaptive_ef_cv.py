import re

with open('hnswlib/adaptive_ef.h', 'r') as f:
    content = f.read()

# 1. Update collect_dep to collect_cv
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
            const size_t k,
            const float truncation_ratio,
            const size_t statics_length)
        {
            hnswdis::ApproximatedScoreCalculator score_cal(truncation_ratio);
            std::vector<float> cvs;
            cvs.reserve(query_vectors.rows());
            for (int i = 0; i < query_vectors.rows(); ++i) {
                auto ret = alg_hnsw.adaptiveSearchKnn(query_vectors.row(i).data(), k, statics_length, score_cal);
                cvs.push_back(std::get<2>(ret));
            }
            return cvs;
        }"""
content = content.replace(old_collect, new_collect)

# 2. Update EfAdapter constructor parameter
content = content.replace("const int n_dep_tables = 5)", "const int n_cv_tables = 5)")

# 3. Update variables inside init_with_dep_buckets (now init_with_cv_buckets)
content = content.replace("void init_with_dep_buckets(", "void init_with_cv_buckets(")
content = content.replace("std::vector<float> deps = collect_dep(*alg_hnsw, *query_vectors);", "std::vector<float> cvs = collect_cv(*alg_hnsw, *query_vectors, k, truncation_ratio, statics_length);")
content = content.replace("deps[a] < deps[b]", "cvs[a] < cvs[b]")
content = content.replace("n_dep_tables", "n_cv_tables")
content = content.replace("dep_centers", "cv_centers")
content = content.replace("dep_tables", "cv_tables")
content = content.replace("deps[order", "cvs[order")
content = content.replace("Training dep-bucket", "Training cv-bucket")

# Also update the call to init_with_cv_buckets inside the constructor
content = content.replace("init_with_dep_buckets(", "init_with_cv_buckets(")

# 4. Serialization
content = content.replace("n_dep;", "n_cv;")
content = content.replace("n_dep = ", "n_cv = ")
content = content.replace("&n_dep", "&n_cv")

# Update Sketch.h references (sketch is passed dep_centers and dep_tables)
# In adaptive_ef.h, EfAdapter provides getters: get_dep_centers(), get_dep_tables()
content = content.replace("get_dep_centers()", "get_cv_centers()")
content = content.replace("get_dep_tables()", "get_cv_tables()")

with open('hnswlib/adaptive_ef.h', 'w') as f:
    f.write(content)

with open('hnswlib/sketch.h', 'r') as f:
    sketch_content = f.read()

sketch_content = sketch_content.replace("d_ep", "cv")
sketch_content = sketch_content.replace("dep_centers", "cv_centers")
sketch_content = sketch_content.replace("dep_tables", "cv_tables")

with open('hnswlib/sketch.h', 'w') as f:
    f.write(sketch_content)

