import re

with open('hnswlib/hnswalg.h', 'r') as f:
    content = f.read()

# The string to be replaced
old_str = """                            if (ef < ef_copy) {
                                ef = ef_copy;
                            }
                            if (ef > ef_copy * 3) {
                                ef = ef_copy * 3;
                            }"""

# The replacement string
new_str = """                            if (ef < ef_copy) {
                                ef = ef_copy;
                            }"""

if old_str in content:
    content = content.replace(old_str, new_str)
    print("Upper bound removed successfully.")
else:
    print("Could not find the upper bound logic to remove.")

with open('hnswlib/hnswalg.h', 'w') as f:
    f.write(content)

