import re

with open('hnswlib/adaptive_ef.h', 'r') as f:
    adaptive = f.read()

adaptive = adaptive.replace("std::vector<float> cvs = collect_cv(*alg_hnsw, *query_vectors, score_cal, k, statics_length);", "hnswdis::ApproximatedScoreCalculator score_cal(truncation_ratio);\\n            std::vector<float> cvs = collect_cv(*alg_hnsw, *query_vectors, score_cal, k, statics_length);")
adaptive = adaptive.replace("has_dep_tables", "has_cv_tables")

with open('hnswlib/adaptive_ef.h', 'w') as f:
    f.write(adaptive)

with open('experiments_driver/run.cpp', 'r') as f:
    run = f.read()

run = run.replace("has_dep_tables", "has_cv_tables")
run = run.replace("train_dep_buckets", "train_cv_buckets")
run = run.replace("init_with_dep_buckets", "init_with_cv_buckets")

with open('experiments_driver/run.cpp', 'w') as f:
    f.write(run)

