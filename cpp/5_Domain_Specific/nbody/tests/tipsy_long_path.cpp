/* Copyright (c) 2026 NVIDIA CORPORATION. All rights reserved.
 *
 * Standalone host regression test for reading a Tipsy file whose path is
 * longer than the old 256-byte stack buffer.
 *
 * Build and run from the repository root:
 *   clang++ -std=c++17 -Icpp/5_Domain_Specific/nbody \
 *     cpp/5_Domain_Specific/nbody/tests/tipsy_long_path.cpp \
 *     -o /tmp/tipsy_long_path && /tmp/tipsy_long_path
 */

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct TestReal4
{
    float x, y, z, w;
};

// Turn the production function's success exit on an open failure into a test
// failure so an unreadable path cannot make this regression pass accidentally.
#define exit(status) throw std::runtime_error("Tipsy file could not be opened")
#include "../tipsy.h"
#undef exit

int main()
{
    namespace fs                   = std::filesystem;
    const std::string     runId    = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const fs::path        root     = fs::temp_directory_path() / ("nbody_tipsy_long_path_test_" + runId);
    const std::string     filename = "input.tipsy";
    constexpr std::size_t targetPathLength = 264;

    fs::remove_all(root);
    const std::size_t fixedLength = root.string().size() + 3 + filename.size();
    if (fixedLength >= targetPathLength) {
        std::cerr << "Temporary directory path is too long for this test\n";
        return 1;
    }

    const std::size_t componentLength = targetPathLength - fixedLength;
    const std::string first(componentLength / 2, 'a');
    const std::string second(componentLength - first.size(), 'b');
    const fs::path    path = root / first / second / filename;
    if (path.string().size() != targetPathLength || first.size() > 255 || second.size() > 255) {
        std::cerr << "Could not construct a valid 264-byte file path\n";
        return 1;
    }

    fs::create_directories(path.parent_path());
    {
        std::ofstream output(path, std::ios::binary);
        if (!output) {
            std::cerr << "Could not create temporary Tipsy file\n";
            return 1;
        }
        const dump header{};
        output.write(reinterpret_cast<const char *>(&header), sizeof(header));
    }

    std::vector<TestReal4> positions;
    std::vector<TestReal4> velocities;
    std::vector<int>       ids;
    int                    total       = -1;
    int                    firstCount  = -1;
    int                    secondCount = -1;
    int                    thirdCount  = -1;

    try {
        read_tipsy_file(positions, velocities, ids, path.string(), total, firstCount, secondCount, thirdCount);
    }
    catch (const std::runtime_error &error) {
        std::cerr << error.what() << '\n';
        fs::remove_all(root);
        return 1;
    }

    fs::remove_all(root);
    if (total != 0 || firstCount != 0 || secondCount != 0 || thirdCount != 0) {
        std::cerr << "Unexpected particle counts for empty Tipsy file\n";
        return 1;
    }
    return 0;
}
