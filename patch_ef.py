with open("hnswlib/adaptive_ef.h", "r") as f:
    text = f.read()

# We want to add the check right at the beginning of the while loop to prevent division by zero
target_str = "while (expected_recall - latest_agg_recall > 1e-4f) // improve the recall for quries with lower recall\n                    {"

replacement = target_str + "\n                        if (recall_diff < 1e-5f) {\n                            std::cout << \"Recall diff too small initially, break.\" << std::endl;\n                            break;\n                        }"

if target_str in text:
    text = text.replace(target_str, replacement)
    with open("hnswlib/adaptive_ef.h", "w") as f:
        f.write(text)
    print("Patched successfully")
else:
    print("Could not find target string")

