import re

file_path = "2metric/statistics.h"
with open(file_path, "r") as f:
    content = f.read()

content = content.replace("int lower = (ef / 500) * 500;", "int lower = ((ef - 1) / 500) * 500 + 1;")

with open(file_path, "w") as f:
    f.write(content)

