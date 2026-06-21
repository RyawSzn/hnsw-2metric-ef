with open('/home/ryawszn/dev/cpp/hnsw-2metric-ef/2metric/lookuptable.h', 'r') as f:
    text = f.read()

text = text.replace('std::string qc_str = line.substr(p2 + 4, p3 - (p2 + 4));', 
                    'std::string qc_str = line.substr(p2 + 3, p3 - (p2 + 3));')

with open('/home/ryawszn/dev/cpp/hnsw-2metric-ef/2metric/lookuptable.h', 'w') as f:
    f.write(text)

print("Fixed qc_str substring indices")
