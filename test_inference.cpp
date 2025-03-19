#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>
#include "inference.hpp"

// Helper function to check if a file exists
bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

// Helper function to display missing files
void checkRequiredFiles() {
    std::vector<std::string> requiredFiles = {
        "gpt2_124M100Steps.bin",
        "vocab.bpe",
        "encoder.json"
    };
    
    std::cout << "Checking required files:" << std::endl;
    bool allFilesExist = true;
    
    for (const auto& file : requiredFiles) {
        bool exists = fileExists(file);
        std::cout << " - " << file << ": " 
                  << (exists ? "Found" : "MISSING") << std::endl;
        if (!exists) {
            allFilesExist = false;
        }
    }
    
    if (!allFilesExist) {
        std::cout << "\nWARNING: Some required files are missing. "
                  << "Make sure they are in the current directory." << std::endl;
    } else {
        std::cout << "\nAll required files are present." << std::endl;
    }
}

// Helper to measure execution time
void runWithTiming(const std::string& prompt, int tokens) {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::string response = Inference::generateResponse(prompt, tokens);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    
    std::cout << "\n--- Generated Text ---\n";
    std::cout << response << std::endl;
    std::cout << "\n--- Performance ---\n";
    std::cout << "Time taken: " << std::fixed << std::setprecision(2) 
              << duration.count() << " seconds" << std::endl;
    std::cout << "Tokens generated: " << tokens << std::endl;
    std::cout << "Tokens per second: " << std::fixed << std::setprecision(2) 
              << tokens / duration.count() << std::endl;
}

void testBasicGeneration() {
    std::cout << "\n=== Basic Generation Test ===\n";
    std::string prompt = "Once upon a time in a distant galaxy";
    int tokens = 30;
    
    std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
    std::cout << "Generating " << tokens << " tokens...\n";
    
    runWithTiming(prompt, tokens);
}

void testEmptyPrompt() {
    std::cout << "\n=== Empty Prompt Test ===\n";
    std::string prompt = "";
    int tokens = 20;
    
    std::cout << "Prompt: <empty>" << std::endl;
    std::cout << "Generating " << tokens << " tokens...\n";
    
    runWithTiming(prompt, tokens);
}

void testLongGeneration() {
    std::cout << "\n=== Long Generation Test ===\n";
    std::string prompt = "The following is a detailed technical explanation of quantum computing:";
    int tokens = 100;
    
    std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
    std::cout << "Generating " << tokens << " tokens...\n";
    
    runWithTiming(prompt, tokens);
}

void testCodeGeneration() {
    std::cout << "\n=== Code Generation Test ===\n";
    std::string prompt = "// Define a function to calculate the fibonacci sequence in C++\n#include <iostream>\n\n";
    int tokens = 50;
    
    std::cout << "Prompt: \n" << prompt << std::endl;
    std::cout << "Generating " << tokens << " tokens...\n";
    
    runWithTiming(prompt, tokens);
}

int main() {
    std::cout << "=== Inference Class Test ===\n\n";
    
    // First check if all required files exist
    checkRequiredFiles();
    
    try {
        // Run the tests
        testBasicGeneration();
        testEmptyPrompt();
        testLongGeneration();
        testCodeGeneration();
        
        // Clean up at the end
        std::cout << "\nCleaning up resources...\n";
        Inference::cleanup();
        std::cout << "Test completed successfully.\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error during testing: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}