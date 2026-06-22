import re

file_path = "2metric/estimator.h"
with open(file_path, "r") as f:
    content = f.read()

# Remove ResearchEstimatorResult
content = re.sub(r'/\*\*\s*\n\s*\* @struct EdgeEval.*?(?=struct EdgeEval)', '', content, flags=re.DOTALL)
content = re.sub(r'struct ResearchEstimatorResult \{.*?\};\n\n', '', content, flags=re.DOTALL)

# Remove probe_query_research
content = re.sub(r'    static ResearchEstimatorResult probe_query_research\(.*?return res;\n    \}\n\n', '', content, flags=re.DOTALL)

with open(file_path, "w") as f:
    f.write(content)

