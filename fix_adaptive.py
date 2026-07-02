import re

with open('hnswlib/adaptive_ef.h', 'r') as f:
    content = f.read()

# I am replicating what patch_adaptive_ef_cv.py did, but ensuring I add ApproximatedScoreCalculator correctly
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
content = content.replace(old_collect, new_collect)

# Rename n_dep_tables, dep_centers, dep_tables
content = content.replace("n_dep", "n_cv")
content = content.replace("dep_centers", "cv_centers")
content = content.replace("dep_tables", "cv_tables")
content = content.replace("deps[", "cvs[")
content = content.replace("dep-bucket", "cv-bucket")
content = content.replace("int n_cv_tables = 5", "int n_cv_tables = 5") 
content = content.replace("void init_with_dep_buckets(", "void init_with_cv_buckets(")
content = content.replace("init_with_dep_buckets(", "init_with_cv_buckets(")

old_line = "std::vector<float> cvs = collect_cv(*alg_hnsw, *query_vectors);"
new_line = "hnswdis::ApproximatedScoreCalculator score_cal(truncation_ratio);\n            std::vector<float> cvs = collect_cv(*alg_hnsw, *query_vectors, score_cal, k, statics_length);"
content = content.replace(old_line, new_line)

content = content.replace("writeBinaryPOD(out, n_cv);", "writeBinaryPOD(out, n_cv);")
content = content.replace("readBinaryPOD(in, n_cv);", "readBinaryPOD(in, n_cv);")
content = content.replace("get_dep_centers", "get_cv_centers")
content = content.replace("get_dep_tables", "get_cv_tables")
content = content.replace("has_dep_tables", "has_cv_tables")

with open('hnswlib/adaptive_ef.h', 'w') as f:
    f.write(content)

