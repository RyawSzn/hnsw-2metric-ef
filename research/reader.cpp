#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <filesystem>

template <typename T>
void readBinaryPOD(std::istream &in, T &podRef) {
    in.read(reinterpret_cast<char*>(&podRef), sizeof(T));
}

void read_ef_adapter(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open " << filename << "\n";
        return;
    }

    size_t estimators_size;
    readBinaryPOD(in, estimators_size);
    std::cout << "EfAdapter Estimators Size: " << estimators_size << "\n\n";

    for (size_t i = 0; i < estimators_size; ++i) {
        int score;
        readBinaryPOD(in, score);
        size_t recall_size;
        readBinaryPOD(in, recall_size);
        
        std::cout << "Score Bucket " << score << " (" << recall_size << " ef-recall pairs):\n  ";
        for (size_t j = 0; j < recall_size; ++j) {
            int ef;
            float recall;
            readBinaryPOD(in, ef);
            readBinaryPOD(in, recall);
            std::cout << "[ef=" << ef << ", rec=" << recall << "] ";
        }
        std::cout << "\n\n";
    }
}

void read_lid_table(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open " << filename << "\n";
        return;
    }

    int magic = 0;
    readBinaryPOD(in, magic);
    if (magic != 0x4C494433) {
        std::cerr << "Not a valid LID2DTable file (bad magic: " << std::hex << magic << ")\n";
        return;
    }

    int ef_min, ef_max, dim_macro, dim_micro;
    readBinaryPOD(in, ef_min);
    readBinaryPOD(in, ef_max);
    readBinaryPOD(in, dim_macro);
    readBinaryPOD(in, dim_micro);

    std::cout << "Lid2DTable Configuration:\n"
              << "  ef_min: " << ef_min << "\n"
              << "  ef_max: " << ef_max << "\n"
              << "  dim_macro: " << dim_macro << "\n"
              << "  dim_micro: " << dim_micro << "\n\n";

    std::vector<int> grid(dim_macro * dim_micro);
    in.read(reinterpret_cast<char*>(grid.data()), grid.size() * sizeof(int));

    std::cout << "Table Grid (macro \\ micro):\n";
    for (int i = 0; i < dim_macro; ++i) {
        for (int j = 0; j < dim_micro; ++j) {
            std::cout << grid[i * dim_micro + j] << "\t";
        }
        std::cout << "\n";
    }
}

void read_estimator(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open " << filename << "\n";
        return;
    }

    size_t type_len;
    readBinaryPOD(in, type_len);
    std::string type(type_len, ' ');
    in.read(&type[0], type_len);

    int rows, cols;
    readBinaryPOD(in, rows);
    readBinaryPOD(in, cols);

    std::cout << "Estimator Type: " << type << "\n"
              << "Covariance Matrix: " << rows << "x" << cols << "\n";

    in.seekg(rows * cols * sizeof(float), std::ios::cur);

    int mean_size;
    readBinaryPOD(in, mean_size);
    std::cout << "Mean Vector Size: " << mean_size << "\n";

    std::vector<float> means(mean_size);
    in.read(reinterpret_cast<char*>(means.data()), mean_size * sizeof(float));

    int var_size;
    readBinaryPOD(in, var_size);
    std::cout << "Variance Vector Size: " << var_size << "\n";

    std::vector<float> variances(var_size);
    in.read(reinterpret_cast<char*>(variances.data()), var_size * sizeof(float));

    std::cout << "Means (first 10): ";
    for(int i = 0; i < std::min(10, mean_size); ++i) std::cout << means[i] << " ";
    std::cout << "...\n";

    std::cout << "Variances (first 10): ";
    for(int i = 0; i < std::min(10, var_size); ++i) std::cout << variances[i] << " ";
    std::cout << "...\n";
}

void read_samplings(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open " << filename << "\n";
        return;
    }

    int q_rows, q_cols;
    readBinaryPOD(in, q_rows);
    readBinaryPOD(in, q_cols);

    std::cout << "Samplings Queries Matrix:\n"
              << "  Rows (queries): " << q_rows << "\n"
              << "  Cols (dimensions): " << q_cols << "\n";

    in.seekg(q_rows * q_cols * sizeof(float), std::ios::cur);

    int gt_rows, gt_cols;
    readBinaryPOD(in, gt_rows);
    readBinaryPOD(in, gt_cols);

    std::cout << "Samplings Ground Truth Matrix:\n"
              << "  Rows (queries): " << gt_rows << "\n"
              << "  Cols (neighbors k): " << gt_cols << "\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <type> <file.bin>\n"
                  << "Supported Types:\n"
                  << "  ada_ef     (for ef_adaptor--*.bin files)\n"
                  << "  lid_table  (for lid-2d-table-*.bin files)\n"
                  << "  estimator  (for statistics/*-estimator-*.bin files)\n"
                  << "  samplings  (for sampling/*-samplings-*.bin files)\n";
        return 1;
    }

    std::string type = argv[1];
    std::string arg_file = argv[2];
    std::filesystem::path full_path = arg_file;

    if (!full_path.is_absolute() && full_path.parent_path().empty()) {
        const char* exp_root_env = std::getenv("EXPERIMENTS_ROOT");
        std::filesystem::path root_dir = exp_root_env ? exp_root_env : ".";
        
        if (type == "ada_ef" || type == "lid_table") {
            full_path = root_dir / "estimation_table" / arg_file;
        } else if (type == "estimator") {
            full_path = root_dir / "statistics" / arg_file;
        } else if (type == "samplings") {
            full_path = root_dir / "sampling" / arg_file;
        }
    }

    std::string filename = full_path.string();
    std::cout << "Reading from: " << filename << "\n\n";

    if (type == "ada_ef") {
        read_ef_adapter(filename);
    } else if (type == "lid_table") {
        read_lid_table(filename);
    } else if (type == "estimator") {
        read_estimator(filename);
    } else if (type == "samplings") {
        read_samplings(filename);
    } else {
        std::cerr << "Unknown type: " << type << "\n";
        return 1;
    }

    return 0;
}
