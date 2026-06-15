import os

files_to_patch = [
    '/home/ryawszn/dev/cpp/hnsw-2metric-ef/research/compare_M.cpp',
    '/home/ryawszn/dev/cpp/hnsw-2metric-ef/research/lookup_RC.cpp',
    '/home/ryawszn/dev/cpp/hnsw-2metric-ef/research/lookup_RC_10x10.cpp'
]

for filepath in files_to_patch:
    if not os.path.exists(filepath):
        continue
        
    with open(filepath, 'r') as f:
        code = f.read()

    # 1. Remove distribution.h include
    code = code.replace('#include "../hnswlib/distribution.h"\n', '')

    # 2. Replace estimator initialization with global_mean computation
    old_init = "hnswdis::CosineSimilarityEstimator estimator(full_data);"
    new_init = """std::cout << "Computing global centroid independently..." << std::endl;
    Eigen::RowVectorXf global_mean = full_data.colwise().mean();"""
    
    # In compare_M.cpp it has a comment
    old_init_2 = """// Initialize Ada-EF Estimator to get theoretical mean distance
    hnswdis::CosineSimilarityEstimator estimator(full_data);"""
    
    if old_init_2 in code:
        code = code.replace(old_init_2, new_init)
    elif old_init in code:
        code = code.replace(old_init, new_init)

    # 3. Replace the inner loop computation
    old_loop_calc = """auto [th_mean, th_var] = estimator.get_practical_distribution(q);
        float d_mean = std::max(0.01f, 1.0f - (float)th_mean);"""
        
    new_loop_calc = """float th_mean = q.dot(global_mean);
        float d_mean = std::max(0.01f, 1.0f - th_mean);"""
        
    if old_loop_calc in code:
        code = code.replace(old_loop_calc, new_loop_calc)

    with open(filepath, 'w') as f:
        f.write(code)
        
    print(f"Patched {filepath}")

