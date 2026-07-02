import re

with open('hnswlib/hnswalg.h', 'r') as f:
    content = f.read()

# Pattern to find and remove the combination logic
old_block = """                                float variance = std::max(0.0f, (sum_sq / q_size) - (mean * mean));
                                cv = std::sqrt(variance) / mean;
                                // Combine RV and CV. High RV = Easy. High CV = Easy.
                                // We multiply them so both factors compound.
                                score = score * (1.0f + 50.0f * cv); 
                            }"""

new_block = """                                float variance = std::max(0.0f, (sum_sq / q_size) - (mean * mean));
                                cv = std::sqrt(variance) / mean;
                            }"""

if old_block in content:
    content = content.replace(old_block, new_block)
    print("Combination logic successfully removed.")
else:
    print("Could not find the combination block.")

with open('hnswlib/hnswalg.h', 'w') as f:
    f.write(content)
