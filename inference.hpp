#pragma once
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <queue>
#include <sstream>
#include "./gpt2.hpp"
#include "llmc/tokenizer.hpp"
#include "./nano.hpp"


class Inference {
private:
    // Static members for singleton pattern
    static std::unique_ptr<gpt2::GPT2> model;
    static std::unique_ptr<nano::GPT2Tokenizer> tokenizer;
    static std::mutex modelMutex;
    static bool isModelLoaded;
    
    // Tokenizer cache implementation
    class TokenizerCache {
    public:
        TokenizerCache(nano::GPT2Tokenizer& base_tokenizer) : tokenizer(base_tokenizer) {}
        
        std::string decode(int token) {
            auto it = cache.find(token);
            if (it != cache.end())
                return it->second;
                
            std::vector<int> token_vec{token};
            std::string result = tokenizer.decode(token_vec);
            cache[token] = result;
            return result;
        }
        
    private:
        nano::GPT2Tokenizer& tokenizer;
        std::unordered_map<int, std::string> cache;
    };
    
    // Helper function for random number generation
    static unsigned int random_u32(unsigned long long* state) {
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        return (*state * 0x2545F4914F6CDD1Dull) >> 32;
    }
    
    static float random_f32(unsigned long long* state) {
        return (random_u32(state) >> 8) / 16777216.0f;
    }
    
    static int sample_mult(float* probabilities, int n, float coin) {
        float cdf = 0.0f;
        for (int i = 0; i < n; i++) {
            cdf += probabilities[i];
            if (coin < cdf) {
                return i;
            }
        }
        return n - 1;  // in case of rounding errors
    }
    
    // Load model if not already loaded
    static bool ensureModelLoaded() {
        std::lock_guard<std::mutex> lock(modelMutex);
        if (!isModelLoaded) {
            try {
                // Initialize model
                model = std::make_unique<gpt2::GPT2>();
                const char* model_path = "./gpt2_124M100Steps.bin";
                if (!model->BuildFromCheckpoint(model_path)) {
                    return false;
                }
                
                // Initialize tokenizer
                tokenizer = std::make_unique<nano::GPT2Tokenizer>("vocab.bpe", "encoder.json");
                isModelLoaded = true;
            } catch (const std::exception& e) {
                return false;
            }
        }
        return isModelLoaded;
    }

public:
    static std::string generateResponse(const std::string& input, int maxTokens) {
        // Check if the model is loaded
        if (!ensureModelLoaded()) {
            return "Error: Failed to load the model or tokenizer.";
        }
        
        try {
            // Tokenize the input
            std::vector<int> input_tokens = tokenizer->encode(input);
            
            // Setup parameters
            int B = 1;  // batch size
            int T = 512;  // max sequence length
            unsigned long long rng_state = 1337;
            
            // Allocate memory
            std::vector<int> gen_tokens(B * T, 0);
            int V = model->config.vocab_size;
            std::unique_ptr<float[]> logit = std::make_unique<float[]>(B * T * V);
            std::unique_ptr<float[]> prob = std::make_unique<float[]>(B * T * V);
            nn::Softmax softmax;
            
            // Initialize tokenizer cache for decoding
            TokenizerCache tokenizer_cache(*tokenizer);
            
            // Fill the gen_tokens with input tokens (and pad if needed)
            for (int i = 0; i < B * T; ++i) {
                if (i < input_tokens.size()) {
                    gen_tokens[i] = input_tokens[i];
                } else {
                    gen_tokens[i] = 50256;  // EOT token
                }
            }
            
            // Generate tokens
            std::stringstream output;
            int genT = std::min(maxTokens + static_cast<int>(input_tokens.size()), T);
            
            // First output the input
            std::string input_text = tokenizer->decode(input_tokens);
            output << input_text;
            
            // Then generate new tokens
            for (int t = input_tokens.size(); t < genT; t++) {
                auto gen_tokens_2d = TTypes<int>::ConstMatrix(gen_tokens.data(), B, T);
                auto logit_3d = Make3DTensor(logit.get(), B, T, V);
                model->gpt2_->Forward(gen_tokens_2d, logit_3d);
                auto logit_2d = MakeConstMatrix(logit.get(), B * T, V);
                auto prob_2d = MakeMatrix(prob.get(), B * T, V);
                softmax.Forward(logit_2d, prob_2d);
                
                float* probs = prob.get() + (t - 1) * V;
                float coin = random_f32(&rng_state);
                int next_token = sample_mult(probs, model->config.vocab_size, coin);
                gen_tokens[t] = next_token;
                
                // Decode and append to output
                std::vector<int> token_vec{next_token};
                output << tokenizer_cache.decode(next_token);
            }
            
            return output.str();
        } catch (const std::exception& e) {
            return "Error during inference: " + std::string(e.what());
        }
    }
    
    // Clean up resources
    static void cleanup() {
        std::lock_guard<std::mutex> lock(modelMutex);
        model.reset();
        tokenizer.reset();
        isModelLoaded = false;
    }
};

// Define static members
std::unique_ptr<gpt2::GPT2> Inference::model = nullptr;
std::unique_ptr<nano::GPT2Tokenizer> Inference::tokenizer = nullptr;
std::mutex Inference::modelMutex;
bool Inference::isModelLoaded = false;