import os
import re

file_path = "research/test_categorizers.cpp"
with open(file_path, "r") as f:
    content = f.read()

content = content.replace("Estimator2Metric::probe_query_research", "Estimator2Metric::probe_query")
content = content.replace("est.RC", "0") # RC removed
content = content.replace("est.RV_rank", "est.revisit_rank")
content = content.replace("est.d_mean", "0")
content = content.replace("est.d_ep", "est.entry_point_dist")
content = content.replace("est.m_LID", "0")

with open(file_path, "w") as f:
    f.write(content)
