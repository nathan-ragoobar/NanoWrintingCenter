#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <filesystem>
#include <cstdlib>
#include <vector>
#include <array>  // Added for std::array

namespace fs = std::filesystem;

// Helper functions
bool writeStringToFile(const std::string& filename, const std::string& content) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to create file: " << filename << std::endl;
        return false;
    }
    file << content;
    file.close();
    return true;
}

std::string readStringFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return "";
    }
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    return content;
}

// Execute command and get output (Windows implementation)
std::pair<int, std::string> exec(const std::string& cmd) {
    std::string result;
    char buffer[128];
    
    #ifdef _WIN32
    FILE* pipe = _popen(cmd.c_str(), "r");
    if (!pipe) {
        return {-1, "Error executing command"};
    }
    while (fgets(buffer, sizeof buffer, pipe) != NULL) {
        result += buffer;
    }
    int exitCode = _pclose(pipe);
    return {exitCode, result};
    #else
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        return {-1, "Error executing command"};
    }
    while (fgets(buffer, sizeof buffer, pipe) != NULL) {
        result += buffer;
    }
    int exitCode = WEXITSTATUS(pclose(pipe));
    return {exitCode, result};
    #endif
}

bool runInferenceCLI(const std::string& inputPrompt, int maxTokens, std::string& output) {
    // Create temporary file for the prompt
    std::string tempFileName = "temp_input.txt";
    if (!writeStringToFile(tempFileName, inputPrompt)) {
        return false;
    }
    
    // Build command
    std::string command;
    #ifdef _WIN32
    command = "inference_cli.exe --generate \"" + inputPrompt + "\" " + std::to_string(maxTokens);
    #else
    command = "./inference_cli --generate \"" + inputPrompt + "\" " + std::to_string(maxTokens);
    #endif
    
    std::cout << "Executing: " << command << std::endl;
    auto [exitCode, result] = exec(command);
    
    if (exitCode != 0) {
        std::cerr << "Command failed with exit code: " << exitCode << std::endl;
        std::cerr << "Output: " << result << std::endl;
        return false;
    }
    
    output = result;
    return true;
}

bool testModelLoad() {
    std::cout << "\n=== Testing Model Load ===" << std::endl;
    
    #ifdef _WIN32
    auto [exitCode, output] = exec("inference_cli.exe --load");
    #else
    auto [exitCode, output] = exec("./inference_cli --load");
    #endif
    
    if (exitCode == 0 && output.find("SUCCESS") != std::string::npos) {
        std::cout << "✓ Model load test passed" << std::endl;
        return true;
    } else {
        std::cerr << "✗ Model load test failed" << std::endl;
        std::cerr << "Output: " << output << std::endl;
        return false;
    }
}

bool testSimpleInference() {
    std::cout << "\n=== Testing Simple Inference ===" << std::endl;
    
    // Create test prompt
    std::string prompt = "The quick brown fox jumps over the lazy";
    std::string output;
    
    // Run inference
    auto start = std::chrono::high_resolution_clock::now();
    bool success = runInferenceCLI(prompt, 10, output);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Inference time: " << elapsed.count() << " seconds" << std::endl;
    
    if (!success) {
        std::cerr << "✗ Inference failed" << std::endl;
        return false;
    }
    
    // Verify output
    std::cout << "Generated text: " << output << std::endl;
    
    if (output.find(prompt) == std::string::npos) {
        std::cerr << "✗ Output doesn't contain the prompt" << std::endl;
        return false;
    }
    
    std::cout << "✓ Simple inference test passed" << std::endl;
    return true;
}

bool testEmptyPrompt() {
    std::cout << "\n=== Testing Empty Prompt ===" << std::endl;
    
    std::string output;
    
    bool success = runInferenceCLI("", 20, output);
    if (!success) {
        std::cerr << "✗ Inference with empty prompt failed" << std::endl;
        return false;
    }
    
    std::cout << "Generated from empty prompt: " << output << std::endl;
    
    if (output.empty()) {
        std::cout << "Note: Empty prompt produced no output (this might be expected)" << std::endl;
    }
    
    std::cout << "✓ Empty prompt test passed" << std::endl;
    return true;
}

bool testLongerGeneration() {
    std::cout << "\n=== Testing Longer Generation ===" << std::endl;
    
    std::string prompt = "In a world where AI has become commonplace,";
    std::string output;
    
    // Generate 40 tokens
    auto start = std::chrono::high_resolution_clock::now();
    bool success = runInferenceCLI(prompt, 40, output);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Long inference time: " << elapsed.count() << " seconds" << std::endl;
    
    if (!success) {
        std::cerr << "✗ Longer generation test failed" << std::endl;
        return false;
    }
    
    std::cout << "Long generation output: " << output << std::endl;
    
    if (output.size() <= prompt.size() + 10) {
        std::cerr << "✗ Long generation produced minimal text" << std::endl;
        return false;
    }
    
    std::cout << "✓ Longer generation test passed" << std::endl;
    return true;
}

bool testInvalidCommand() {
    std::cout << "\n=== Testing Invalid Command ===" << std::endl;
    
    // Try to run with an invalid command
    #ifdef _WIN32
    auto [exitCode, output] = exec("inference_cli.exe --invalid-command");
    #else
    auto [exitCode, output] = exec("./inference_cli --invalid-command");
    #endif
    
    if (exitCode == 0) {
        std::cerr << "✗ CLI unexpectedly succeeded with invalid command" << std::endl;
        return false;
    }
    
    std::cout << "✓ Invalid command test passed" << std::endl;
    return true;
}

int main() {
    std::cout << "===================================" << std::endl;
    std::cout << "   INFERENCE CLI TEST SUITE" << std::endl;
    std::cout << "===================================" << std::endl;
    
    // Track test results
    std::vector<bool> results;
    
    // Run tests
    results.push_back(testModelLoad());
    results.push_back(testSimpleInference());
    results.push_back(testEmptyPrompt());
    results.push_back(testLongerGeneration());
    results.push_back(testInvalidCommand());
    
    // Summarize results
    std::cout << "\n===================================" << std::endl;
    std::cout << "   TEST SUMMARY" << std::endl;
    std::cout << "===================================" << std::endl;
    
    int passCount = 0;
    for (bool result : results) {
        if (result) passCount++;
    }
    
    std::cout << "Passed " << passCount << " of " << results.size() << " tests" << std::endl;
    
    return passCount == results.size() ? 0 : 1;
}