#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include "utils.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <regex>
#include <climits>
#include "json.hpp"

using json = nlohmann::json;

namespace nano {

    class GPT2Tokenizer {
        private:
            std::unordered_map<std::string, int> encoder;  // Token → ID
            std::unordered_map<int, std::string> decoder;  // ID → Token
            std::unordered_map<std::string, int> bpeRanks;
            int eot_token_id;
        
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
                std::string processed_word = word;
                if (!word.empty() && word[0] == ' ') {
                    // Replace leading space with Ġ (U+0120) using correct UTF-8 bytes (0xC4 0xA0)
                    processed_word = "\xC4\xA0" + word.substr(1);
                }
                
                // Instead of character-by-character, we need to properly handle UTF-8
                std::vector<std::string> tokens;
                size_t i = 0;
                while (i < processed_word.length()) {
                    // Check for UTF-8 multi-byte sequences
                    if ((processed_word[i] & 0xE0) == 0xC0) { // 2-byte UTF-8
                        if (i + 1 < processed_word.length()) {
                            tokens.push_back(processed_word.substr(i, 2));
                            i += 2;
                        } else {
                            tokens.push_back(std::string(1, processed_word[i]));
                            i++;
                        }
                    } else {
                        tokens.push_back(std::string(1, processed_word[i]));
                        i++;
                    }
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
            
            // Initialize the EOT token ID
            if (encoder.find("<|endoftext|>") != encoder.end()) {
                eot_token_id = encoder["<|endoftext|>"];
            } else {
                // Fallback value if not found
                eot_token_id = 50256; // Standard GPT-2 EOT token ID
                std::cerr << "Warning: <|endoftext|> token not found in encoder. Using default value: " 
                          << eot_token_id << std::endl;
            }
        }

            // Getter for EOT token
            int eot_token() const {
                return eot_token_id;
            }

            // Check if the tokenizer is properly initialized
            bool is_initialized() const {
                return !encoder.empty() && !decoder.empty() && !bpeRanks.empty();
            }
        
            // Tokenize input text into token IDs
            std::vector<int> encode(const std::string& text) {
                // Modified regex to treat consecutive special characters as a single token
                std::regex wordRegex(R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+|[#]+|[^\s[:alpha:][:digit:]]+|\s+)");
                std::sregex_iterator iter(text.begin(), text.end(), wordRegex);
                std::sregex_iterator end;
            
                std::vector<int> tokenIDs;
                while (iter != end) {
                    std::string word = iter->str();
                    
                    // Check if this is a special sequence like "###" that should be treated as a single token
                    auto directIt = encoder.find(word);
                    if (directIt != encoder.end()) {
                        // If the entire sequence is a token, use it directly
                        tokenIDs.push_back(directIt->second);
                    } else {
                        // Otherwise use BPE
                        std::vector<std::string> bpeTokens = bpe(word);
                        for (const auto& token : bpeTokens) {
                            if (encoder.find(token) != encoder.end()) {
                                tokenIDs.push_back(encoder[token]);
                            } else {
                                std::cerr << "Warning: Token not found in encoder: " << token << std::endl;
                            }
                        }
                    }
                    ++iter;
                }
                return tokenIDs;
            }

            std::string decode(const std::vector<int>& tokenIDs) {
                std::ostringstream result;
                for (int id : tokenIDs) {
                    if (decoder.find(id) != decoder.end()) {
                        std::string token = decoder[id];
                        // Handle special characters
                        if (!token.empty()) {
                            // Check for UTF-8 encoding of 'Ġ' (unicode character U+0120)
                            // Usually appears as a special space token in GPT-2
                            if (token.length() >= 2 && 
                                (unsigned char)token[0] == 0xC4 && 
                                (unsigned char)token[1] == 0xA0) {
                                result << ' ' << token.substr(2);  // Space followed by the rest
                            }
                            // Check for UTF-8 encoding of 'Ċ' (unicode character U+010A)
                            // Usually represents newline
                            else if (token.length() >= 2 && 
                                     (unsigned char)token[0] == 0xC4 && 
                                     (unsigned char)token[1] == 0x8A) {
                                result << '\n' << token.substr(2);  // Newline followed by rest
                            }
                            else if (token == "<|endoftext|>") {
                                result << "\n[END]\n";
                            }
                            // Try to handle BPE token formats
                            else if (token[0] == 'Ġ') {  // Special GPT-2 space token
                                result << ' ' << token.substr(1);
                            }
                            else if (token[0] == 'Ċ') {  // Special GPT-2 newline token
                                result << '\n' << token.substr(1);
                            }
                            else {
                                result << token;  // Regular token
                            }
                        }
                    } else {
                        // Handle unknown tokens
                        result << "[UNK:" << id << "]";
                    }
                }
                return result.str();
            }
               
        };


class Tokenizer {
private:
    std::vector<std::string> tokens_;
    std::unordered_map<std::string, uint32_t> token_to_id_;
    uint32_t vocab_size_;
    uint32_t eot_token_;
    bool init_ok_;

     // Helper for longest token match
    std::pair<uint32_t, size_t> find_longest_token(const std::string& text, size_t start) const {
        size_t longest_len = 0;
        uint32_t token_id = UINT32_MAX;
        
        // Try each possible substring starting at 'start'
        for (size_t end = start + 1; end <= text.length(); end++) {
            std::string substr = text.substr(start, end - start);
            auto it = token_to_id_.find(substr);
            if (it != token_to_id_.end() && substr.length() > longest_len) {
                longest_len = substr.length();
                token_id = it->second;
            }
        }
        
        return {token_id, longest_len};
    }

public:
    Tokenizer() : init_ok_(false) {}

    void init(const std::string& dict_file="./llmc/gpt2.txt") {
        FILE* file = fopen(dict_file.c_str(), "r");
        if (!file) {
            fprintf(stderr, "Failed to open dictionary file: %s\n", dict_file.c_str());
            return;
        }

        char line[1024];
        uint32_t token_id = 0;
        tokens_.clear();
        token_to_id_.clear();

        while (fgets(line, sizeof(line), file)) {
            size_t len = strlen(line);
            if (len > 0 && line[len-1] == '\n') {
                line[len-1] = '\0';
            }

            char* token = strtok(line, "\t");
            char* id_str = strtok(NULL, "\t");
            
            if (id_str) {
                token_id = std::stoul(id_str);
            }

            // Ensure vector has space
            if (token_id >= tokens_.size()) {
                tokens_.resize(token_id + 1);
            }
            
            tokens_[token_id] = token;
            token_to_id_[token] = token_id;
            
            if (!id_str) {
                token_id++;
            }
        }

        vocab_size_ = tokens_.size();
        eot_token_ = token_to_id_["<|endoftext|>"];
        init_ok_ = true;

        fcloseCheck(file);
    }

    std::vector<uint32_t> encode_string(const std::string& text) const {
        std::vector<uint32_t> result;
        size_t pos = 0;
        
        while (pos < text.length()) {
            auto [token_id, length] = find_longest_token(text, pos);
            if (token_id == UINT32_MAX || length == 0) {
                // No token found, move forward by one byte
                pos++;
            } else {
                result.push_back(token_id);
                pos += length;
            }
        }
        
        return result;
    }

    std::string decode_string(const std::vector<uint32_t>& token_ids) const {
        std::string result;
        for (uint32_t id : token_ids) {
            if (id < vocab_size_) {
                result += tokens_[id];
            }
        }
        return result;
    }

    std::string decode_string(const int* gen_tokens, size_t length) const {
        std::string result;
        for (size_t i = 0; i < length; ++i) {
            uint32_t id = gen_tokens[i];
            if (id < vocab_size_) {
                result += tokens_[id];
            }
        }
        return result;
    }

    uint32_t encode(const std::string& token) const {
        auto it = token_to_id_.find(token);
        if (it != token_to_id_.end()) {
            return it->second;
        }
        fprintf(stderr, "Token not found: %s\n", token.c_str());
        return UINT32_MAX;
    }

    std::string decode(uint32_t token_id) const {
        if (!init_ok_) {
            fprintf(stderr, "Tokenizer not initialized\n");
            return "";
        }
        if (token_id >= vocab_size_) {
            fprintf(stderr, "Invalid token ID: %u\n", token_id);
            return "";
        }
        return tokens_[token_id];
    }

    uint32_t get_vocab_size() const { return vocab_size_; }
    uint32_t get_eot_token() const { return eot_token_; }
    bool is_initialized() const { return init_ok_; }
};

} // namespace nano

#endif