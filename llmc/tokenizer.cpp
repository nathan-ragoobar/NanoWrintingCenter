#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <regex>
#include <climits>
#include "json.hpp"
#include "tokenizer.hpp"
/*
using json = nlohmann::json;
 
class GPT2Tokenizer {
private:
    std::unordered_map<std::string, int> encoder;  // Token → ID
    std::unordered_map<int, std::string> decoder;  // ID → Token
    std::unordered_map<std::string, int> bpeRanks;

    // Load BPE merges from file
    void loadBPE(const std::string& filename) {
        std::ifstream file(filename);
        std::string line;
        int rank = 0;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            std::string token1, token2;
            iss >> token1 >> token2;
            bpeRanks[token1 + " " + token2] = rank++;
        }
    }

    // Load JSON encoder mapping
    void loadEncoder(const std::string& filename) {
        std::ifstream file(filename);
        json jsonData;
        file >> jsonData;

        for (auto it = jsonData.begin(); it != jsonData.end(); ++it) {
            int id = it.value();
            encoder[it.key()] = id;
            decoder[id] = it.key();
        }
    }

    // BPE Merge function (returns tokenized word in BPE format)
    std::vector<std::string> bpe(const std::string& word) {
        std::vector<std::string> tokens;
        for (char c : word) {
            tokens.push_back(std::string(1, c));
        }

        while (tokens.size() > 1) {
            std::pair<int, std::pair<std::string, std::string>> minPair = {INT_MAX, {"", ""}};
            for (size_t i = 0; i < tokens.size() - 1; ++i) {
                std::string pair = tokens[i] + " " + tokens[i + 1];
                if (bpeRanks.find(pair) != bpeRanks.end() && bpeRanks[pair] < minPair.first) {
                    minPair = {bpeRanks[pair], {tokens[i], tokens[i + 1]}};
                }
            }

            if (minPair.first == INT_MAX) break;

            // Merge the pair
            std::vector<std::string> newTokens;
            bool merged = false;
            for (size_t i = 0; i < tokens.size(); ++i) {
                if (i < tokens.size() - 1 && tokens[i] == minPair.second.first && tokens[i + 1] == minPair.second.second && !merged) {
                    newTokens.push_back(tokens[i] + tokens[i + 1]);
                    ++i; // Skip the next token as it is merged
                    merged = true;
                } else {
                    newTokens.push_back(tokens[i]);
                }
            }
            tokens = std::move(newTokens);
        }

        return tokens;
    }

public:
    // Constructor
    GPT2Tokenizer(const std::string& bpeFile, const std::string& encoderFile) {
        loadBPE(bpeFile);
        loadEncoder(encoderFile);
    }

    // Tokenize input text into token IDs
    std::vector<int> encode(const std::string& text) {
        std::regex wordRegex(R"(\S+)");
        std::sregex_iterator iter(text.begin(), text.end(), wordRegex);
        std::sregex_iterator end;

        std::vector<int> tokenIDs;
        while (iter != end) {
            std::string word = iter->str();
            std::vector<std::string> bpeTokens = bpe(word);
            for (const auto& token : bpeTokens) {
                if (encoder.find(token) != encoder.end()) {
                    tokenIDs.push_back(encoder[token]);
                } else {
                    std::cerr << "Warning: Token not found in encoder: " << token << std::endl;
                }
            }
            ++iter;
        }
        return tokenIDs;
    }

    // Decode token IDs back into text
    std::string decode(const std::vector<int>& tokenIDs) {
        std::ostringstream result;
        for (int id : tokenIDs) {
            if (decoder.find(id) != decoder.end()) {
                result << decoder[id] << " ";
            } else {
                std::cerr << "Warning: ID not found in decoder: " << id << std::endl;
            }
        }
        return result.str();
    }
};
*/
int main() {
    nano::GPT2Tokenizer tokenizer("vocab.bpe", "encoder.json");

    std::string text = "Hello, world! My name is Nathan";
    std::vector<int> tokens = tokenizer.encode(text);

    std::cout << "Token IDs: ";
    for (int id : tokens) {
        std::cout << id << " ";
    }
    std::cout << std::endl;

    std::string decodedText = tokenizer.decode(tokens);
    std::cout << "Decoded Text: " << decodedText << std::endl;

    return 0;
}



/*
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <regex>
#include <climits>
#include <algorithm>

class GPT2Tokenizer {
private:
    std::unordered_map<std::string, int> bpeRanks;

    // Load BPE merges from file
    void loadBPE(const std::string& filename) {
        std::ifstream file(filename);
        std::string line;
        int rank = 0;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            std::string token1, token2;
            iss >> token1 >> token2;
            bpeRanks[token1 + " " + token2] = rank++;
        }
    }

    // BPE Merge function
    std::string bpe(const std::string& word) {
        std::vector<std::string> tokens;
        for (char c : word) {
            tokens.push_back(std::string(1, c));
        }

        while (tokens.size() > 1) {
            std::pair<int, std::pair<std::string, std::string>> minPair = {INT_MAX, {"", ""}};
            for (size_t i = 0; i < tokens.size() - 1; ++i) {
                std::string pair = tokens[i] + " " + tokens[i + 1];
                if (bpeRanks.find(pair) != bpeRanks.end() && bpeRanks[pair] < minPair.first) {
                    minPair = {bpeRanks[pair], {tokens[i], tokens[i + 1]}};
                }
            }

            if (minPair.first == INT_MAX) break;

            // Merge the pair
            std::vector<std::string> newTokens;
            bool merged = false;
            for (size_t i = 0; i < tokens.size(); ++i) {
                if (i < tokens.size() - 1 && tokens[i] == minPair.second.first && tokens[i + 1] == minPair.second.second && !merged) {
                    newTokens.push_back(tokens[i] + tokens[i + 1]);
                    ++i; // Skip the next token as it is merged
                    merged = true;
                } else {
                    newTokens.push_back(tokens[i]);
                }
            }
            tokens = std::move(newTokens);
        }

        std::ostringstream result;
        for (const auto& token : tokens) {
            result << token << " ";
        }
        return result.str();
    }

public:
    // Constructor
    GPT2Tokenizer(const std::string& bpeFile) {
        loadBPE(bpeFile);
    }

    // Tokenize input text
    std::vector<std::string> tokenize(const std::string& text) {
        std::regex wordRegex(R"(\S+)");
        std::sregex_iterator iter(text.begin(), text.end(), wordRegex);
        std::sregex_iterator end;

        std::vector<std::string> tokens;
        while (iter != end) {
            std::string word = iter->str();
            tokens.push_back(bpe(word));
            ++iter;
        }
        return tokens;
    }
};

int main() {
    GPT2Tokenizer tokenizer("vocab.bpe");

    std::string text = "Hello, world!";
    std::vector<std::string> tokens = tokenizer.tokenize(text);

    std::cout << "Tokens:" << std::endl;
    for (const auto& token : tokens) {
        std::cout << token << std::endl;
    }

    return 0;
}
*/