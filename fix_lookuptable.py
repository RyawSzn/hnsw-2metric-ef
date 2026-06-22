import re

file_path = "2metric/lookuptable.h"
with open(file_path, "r") as f:
    content = f.read()

content = content.replace("EP_lower", "ep_lower_bound")
content = content.replace("EP_upper", "ep_upper_bound")
content = content.replace("RV_lower", "rv_lower_bound")
content = content.replace("RV_upper", "rv_upper_bound")

# Add back default constructor if missing
if "LookupTable2D() :" not in content and "LookupTable2D() {}" not in content:
    content = content.replace("public:\n", "public:\n    LookupTable2D() : default_ef(32), target_recall(0.95f) {}\n")

with open(file_path, "w") as f:
    f.write(content)

