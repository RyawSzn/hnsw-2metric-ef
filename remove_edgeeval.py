import re

file_path = "2metric/estimator.h"
with open(file_path, "r") as f:
    content = f.read()

content = re.sub(r'struct EdgeEval \{.*?\};\n\n', '', content, flags=re.DOTALL)

with open(file_path, "w") as f:
    f.write(content)

