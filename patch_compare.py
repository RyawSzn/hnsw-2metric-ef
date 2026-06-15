with open('/home/ryawszn/dev/cpp/hnsw-2metric-ef/research/compare_adaef.cpp', 'r') as f:
    code = f.read()

# Fix the lookup table path
code = code.replace('"research/lookup_table_RC_10x10.csv"', '"research/lookup_table_10x10.csv"')

# Add sample size parsing
old_parse = """    float target_recall = 0.95f;
    std::string dataset = "deep-image-96-angular";
    if (argc > 1) dataset = argv[1];"""

new_parse = """    float target_recall = 0.95f;
    std::string dataset = "glove-100-angular";
    int sample_size = 2000;

    if (argc > 1) dataset = argv[1];
    if (argc > 2) sample_size = std::stoi(argv[2]);"""
code = code.replace(old_parse, new_parse)

# Apply sample size logic before Ada-EF initialization
old_nq = """    int nq = query_vectors.rows();
    int k = 10;"""

new_nq = """    int total_queries = query_vectors.rows();
    int nq = total_queries;
    std::vector<int> query_indices(total_queries);
    std::iota(query_indices.begin(), query_indices.end(), 0);
    if (sample_size > 0 && sample_size < total_queries) {
        std::srand(42);
        std::random_shuffle(query_indices.begin(), query_indices.end());
        query_indices.resize(sample_size);
        nq = sample_size;
    }
    int k = 10;"""
code = code.replace(old_nq, new_nq)

# Patch the omp loops to use query_indices[i] instead of i directly
# Ada-EF loop:
old_ada = """    #pragma omp parallel for reduction(+:adaef_hits)
    for (int i = 0; i < nq; i++) {
        auto pq = alg_hnsw_ptr->adaptiveSearchKnnTest(
            query_vectors.row(i).data(), k, statics_length, score_cal, &sketch
        );
        while (!pq.empty()) {
            tableint id = pq.top().second;
            pq.pop();
            for (int c = 0; c < k; c++) {
                if (ground_truth(i, c) == id) {"""

new_ada = """    #pragma omp parallel for reduction(+:adaef_hits)
    for (int i = 0; i < nq; i++) {
        int q_idx = query_indices[i];
        auto pq = alg_hnsw_ptr->adaptiveSearchKnnTest(
            query_vectors.row(q_idx).data(), k, statics_length, score_cal, &sketch
        );
        while (!pq.empty()) {
            tableint id = pq.top().second;
            pq.pop();
            for (int c = 0; c < k; c++) {
                if (ground_truth(q_idx, c) == id) {"""
code = code.replace(old_ada, new_ada)

# Probe loop:
old_probe = """    #pragma omp parallel for
    for (int i = 0; i < nq; i++) {
        Eigen::RowVectorXf q = query_vectors.row(i);"""

new_probe = """    #pragma omp parallel for
    for (int i = 0; i < nq; i++) {
        int q_idx = query_indices[i];
        Eigen::RowVectorXf q = query_vectors.row(q_idx);"""
code = code.replace(old_probe, new_probe)

# Our 2D Search loop:
old_mat = """    #pragma omp parallel for reduction(+:matrix_hits)
    for (int i = 0; i < nq; i++) {
        alg_hnsw_ptr->setEf(target_efs[i]);
        auto pq = alg_hnsw_ptr->searchKnn(query_vectors.row(i).data(), k);
        while (!pq.empty()) {
            tableint id = pq.top().second;
            pq.pop();
            for (int c = 0; c < k; c++) {
                if (ground_truth(i, c) == id) {"""

new_mat = """    #pragma omp parallel for reduction(+:matrix_hits)
    for (int i = 0; i < nq; i++) {
        int q_idx = query_indices[i];
        int ef_used = target_efs[i];
        if (ef_used == 0) ef_used = 500; // fallback if empty bin
        alg_hnsw_ptr->setEf(ef_used);
        auto pq = alg_hnsw_ptr->searchKnn(query_vectors.row(q_idx).data(), k);
        while (!pq.empty()) {
            tableint id = pq.top().second;
            pq.pop();
            for (int c = 0; c < k; c++) {
                if (ground_truth(q_idx, c) == id) {"""
code = code.replace(old_mat, new_mat)

# Evaluation print loop:
old_print = """    for (int i = 0; i < nq; i++) {
        
        // 1. Run Ada-EF
        alg_hnsw_ptr->setEf(wae);
        auto res_ada = alg_hnsw_ptr->adaptiveSearchKnn(query_vectors.row(i).data(), k, statics_length, score_cal, &sketch);
        float score = res_ada.second;
        int ef_ada = std::max((int)wae, (int)sketch.estimate_ef2(score));
        
        int hits_ada = 0;
        auto pq_ada = res_ada.first;
        while (!pq_ada.empty()) {
            tableint id = pq_ada.top().second;
            pq_ada.pop();
            for (int c = 0; c < k; c++) {
                if (ground_truth(i, c) == id) { hits_ada++; break; }
            }
        }
        float rec_ada = hits_ada / 10.0f;
        adaef_total_ef += ef_ada;

        // 2. Run Our 2D Method
        int ef_mat = target_efs[i];
        alg_hnsw_ptr->setEf(ef_mat);
        auto pq_mat = alg_hnsw_ptr->searchKnn(query_vectors.row(i).data(), k);
        int hits_mat = 0;
        while (!pq_mat.empty()) {
            tableint id = pq_mat.top().second;
            pq_mat.pop();
            for (int c = 0; c < k; c++) {
                if (ground_truth(i, c) == id) { hits_mat++; break; }
            }
        }"""

new_print = """    for (int i = 0; i < nq; i++) {
        int q_idx = query_indices[i];
        
        // 1. Run Ada-EF
        alg_hnsw_ptr->setEf(wae);
        auto res_ada = alg_hnsw_ptr->adaptiveSearchKnn(query_vectors.row(q_idx).data(), k, statics_length, score_cal, &sketch);
        float score = res_ada.second;
        int ef_ada = std::max((int)wae, (int)sketch.estimate_ef2(score));
        
        int hits_ada = 0;
        auto pq_ada = res_ada.first;
        while (!pq_ada.empty()) {
            tableint id = pq_ada.top().second;
            pq_ada.pop();
            for (int c = 0; c < k; c++) {
                if (ground_truth(q_idx, c) == id) { hits_ada++; break; }
            }
        }
        float rec_ada = hits_ada / 10.0f;
        adaef_total_ef += ef_ada;

        // 2. Run Our 2D Method
        int ef_mat = target_efs[i];
        if (ef_mat == 0) ef_mat = 500;
        alg_hnsw_ptr->setEf(ef_mat);
        auto pq_mat = alg_hnsw_ptr->searchKnn(query_vectors.row(q_idx).data(), k);
        int hits_mat = 0;
        while (!pq_mat.empty()) {
            tableint id = pq_mat.top().second;
            pq_mat.pop();
            for (int c = 0; c < k; c++) {
                if (ground_truth(q_idx, c) == id) { hits_mat++; break; }
            }
        }"""
code = code.replace(old_print, new_print)

with open('/home/ryawszn/dev/cpp/hnsw-2metric-ef/research/compare_adaef.cpp', 'w') as f:
    f.write(code)
