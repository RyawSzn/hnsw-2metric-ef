with open('/home/ryawszn/dev/cpp/hnsw-2metric-ef/2metric/compare/compare_ef.cpp', 'r') as f:
    text = f.read()

text = text.replace('#pragma omp parallel for reduction(+:total_err_ada, total_err_2metric, ada_under, metric2_under)\\\n    for (int i = 0; i < sample_size; i++) {', '#pragma omp parallel for reduction(+:total_err_ada, total_err_2metric, ada_under, metric2_under)\n    for (int i = 0; i < sample_size; i++) {')
text = text.replace('    for (int i = 0; i < sample_size; i++) {\n        int idx = q_idx[i];', '#pragma omp parallel for reduction(+:total_err_ada, total_err_2metric, ada_under, metric2_under)\n    for (int i = 0; i < sample_size; i++) {\n        int idx = q_idx[i];')

# Also fix the weird braces if any were corrupted. The error:
# "i was not declared in this scope", "out does not name a type"
# means the `for` loop declaration was completely destroyed.

