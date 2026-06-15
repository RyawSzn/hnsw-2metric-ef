with open('/home/ryawszn/dev/cpp/hnsw-2metric-ef/research/compare_predictors.cpp', 'r') as f:
    code = f.read()

# Add baseline_recall to CSV header
code = code.replace(
    'out << "query_id,M_Graph,M_RC,m_Spear,m_Sphere,true_ef\\n";',
    'out << "query_id,M_Graph,M_RC,m_Spear,m_Sphere,true_ef,baseline_recall\\n";'
)

# Add baseline recall calculation right after spear_W sort
old_spear_sort = 'std::sort(spear_W.begin(), spear_W.end());'
new_spear_sort = """std::sort(spear_W.begin(), spear_W.end());
        int base_hits = 0;
        auto pq_copy = top_candidates;
        while (!pq_copy.empty()) {
            tableint id = pq_copy.top().second;
            pq_copy.pop();
            for (int c = 0; c < 10; c++) {
                if (ground_truth(q_idx, c) == id) { base_hits++; break; }
            }
        }
        float baseline_recall = base_hits / 10.0f;"""
code = code.replace(old_spear_sort, new_spear_sort)

# Update the CSV write line
code = code.replace(
    'out << q_idx << "," << M_Graph << "," << M_RC << "," << m_Spear << "," << m_Sphere << "," << true_ef << "\\n";',
    'out << q_idx << "," << M_Graph << "," << M_RC << "," << m_Spear << "," << m_Sphere << "," << true_ef << "," << baseline_recall << "\\n";'
)

with open('/home/ryawszn/dev/cpp/hnsw-2metric-ef/research/compare_predictors.cpp', 'w') as f:
    f.write(code)
