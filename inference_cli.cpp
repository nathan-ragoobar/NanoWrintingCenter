#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include "./gpt2.hpp"
#include "llmc/tokenizer.hpp"
#include "./nano.hpp"

unsigned int random_u32(unsigned long long* state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
  }
float random_f32(unsigned long long* state) {  // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
  }
  
int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
      cdf += probabilities[i];
      if (coin < cdf) {
        return i;
      }
    }
    return n - 1;  // in case of rounding errors
  }

// Function to apply temperature to probability distribution
void apply_temperature(float* probabilities, int n, float temperature) {
    if (temperature == 1.0f) return; // No change needed for temperature 1.0
    
    // Apply temperature by dividing logits by temperature
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        // We need to convert back to logits, apply temperature, then convert back to probabilities
        if (probabilities[i] > 0) { // Avoid log(0)
            probabilities[i] = powf(probabilities[i], 1.0f/temperature);
            sum += probabilities[i];
        }
    }
    
    // Renormalize to get a valid probability distribution
    if (sum > 0) {
        for (int i = 0; i < n; i++) {
            probabilities[i] /= sum;
        }
    }
}

int main(int argc, char** argv) {
    // Parse command line arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " [--load] [--generate \"input text\" max_tokens [--temperature temp] [model_path]]" << std::endl;
        return 1;
    }
    
    std::string command = argv[1];
    
    if (command == "--load") {
        // Try to load the model and report success/failure
        try {
            gpt2::GPT2 model;
            const char* model_path = "./gpt2_124M100Steps.bin";  // Default path
            // Check if a path was provided
            if (argc > 2) {
                model_path = argv[2];  // Use the provided path
            }
            if (model.BuildFromCheckpoint(model_path)) {
                nano::GPT2Tokenizer tokenizer("vocab.bpe", "encoder.json");
                std::cout << "SUCCESS: Model loaded successfully" << std::endl;
                return 0;
            } else {
                std::cerr << "ERROR: Failed to load model from " << model_path << std::endl;
                return 1;
            }
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception while loading model: " << e.what() << std::endl;
            return 1;
        }
    }
    else if (command == "--generate") {
        if (argc < 4) {
            std::cerr << "ERROR: --generate requires input text and max_tokens" << std::endl;
            return 1;
        }
        
        std::string input = argv[2];
        int maxTokens;
        try {
            maxTokens = std::stoi(argv[3]);
        } catch (...) {
            std::cerr << "ERROR: max_tokens must be a valid integer" << std::endl;
            return 1;
        }
        
        // Default temperature value
        float temperature = 1.0f;
        
        // Check for temperature flag
        for (int i = 4; i < argc - 1; i++) {
            std::string arg = argv[i];
            if (arg == "--temperature") {
                try {
                    temperature = std::stof(argv[i+1]);
                    // Ensure temperature is within reasonable bounds
                    if (temperature <= 0.0f) {
                        std::cerr << "WARNING: Temperature must be positive, using 1.0" << std::endl;
                        temperature = 1.0f;
                    }
                } catch (...) {
                    std::cerr << "WARNING: Invalid temperature value, using 1.0" << std::endl;
                }
                // Skip the next argument since it's the temperature value
                i++;
            }
        }
        
        try {
            // Initialize model
            gpt2::GPT2 model;
            
            // Check if a model path was provided as the last argument
            const char* model_path = "./c4_gpt2_124M_step_4000.bin"; // Default path
            
            // Look for the model path in args that are not flags
            for (int i = 4; i < argc; i++) {
                std::string arg = argv[i];
                if (arg != "--temperature" && (i == argc - 1 || argv[i-1] != "--temperature")) {
                    model_path = argv[i];
                }
            }
            
            if (!model.BuildFromCheckpoint(model_path)) {
                std::cerr << "ERROR: Failed to load model from " << model_path << std::endl;
                return 1;
            }
            
            // Initialize tokenizer
            nano::GPT2Tokenizer tokenizer("vocab.bpe", "encoder.json");
            
            // Hardcoded tokens for "### Instruction: Correct the grammar in this text ### Input: "
            std::vector<int> instruction_tokens = {
                21017, 46486, 25, 22941, 262, 23491, 287, 428, 2420, 220, 21017, 23412, 25, 220
            };
            
            // Tokenize the input separately
            std::vector<int> text_tokens = tokenizer.encode(input);
            
            // Combine instruction tokens with input tokens
            std::vector<int> input_tokens;
            input_tokens.reserve(instruction_tokens.size() + text_tokens.size());
            input_tokens.insert(input_tokens.end(), instruction_tokens.begin(), instruction_tokens.end());
            input_tokens.insert(input_tokens.end(), text_tokens.begin(), text_tokens.end());

            
            // Setup parameters
            int B = 1;  // batch size
            int T = 1024;  // max sequence length
            unsigned long long rng_state = 1337;
            
            // Allocate memory
            std::vector<int> gen_tokens(B * T, 0);
            int V = model.config.vocab_size;
            std::vector<float> logit(B * T * V);
            std::vector<float> prob(B * T * V);
            nn::Softmax softmax;
            
            // Fill the gen_tokens with input tokens (and pad if needed)
            for (int i = 0; i < input_tokens.size() && i < B * T; ++i) {
                gen_tokens[i] = input_tokens[i];
            }
            
            // Generate tokens
            int genT = std::min(maxTokens + static_cast<int>(input_tokens.size()), T);
            
            // Generate new tokens
            for (int t = input_tokens.size(); t < genT; t++) {
                auto gen_tokens_2d = TTypes<int>::ConstMatrix(gen_tokens.data(), B, T);
                auto logit_3d = Make3DTensor(logit.data(), B, T, V);
                model.gpt2_->Forward(gen_tokens_2d, logit_3d);
                auto logit_2d = MakeConstMatrix(logit.data(), B * T, V);
                auto prob_2d = MakeMatrix(prob.data(), B * T, V);
                softmax.Forward(logit_2d, prob_2d);
                
                float* probs = prob.data() + (t - 1) * V;
                
                // Apply temperature to modify the probability distribution
                apply_temperature(probs, model.config.vocab_size, temperature);
                
                float coin = random_f32(&rng_state);
                int next_token = sample_mult(probs, model.config.vocab_size, coin);
                gen_tokens[t] = next_token;
            }

            //Consider implementing token filtering at the generation stage.
            //That way we could break after getting the end token "### End"
            
            // Decode the full token sequence
            std::vector<int> result_tokens;

            // Only include the new tokens generated by the model, not the input tokens
            result_tokens.assign(gen_tokens.begin() + input_tokens.size(), gen_tokens.begin() + genT);

            // Decode just the generated tokens
            std::string result = tokenizer.decode(result_tokens);
            
            // Extract just the corrected sentence between "### Output:" and "### End"
            size_t start_pos = result.find("### Output:");
            size_t end_pos = result.find("### End");

            if (start_pos != std::string::npos && end_pos != std::string::npos) {
                // Extract the content between the markers, accounting for the length of "### Output:"
                start_pos += std::string("### Output:").length();
                std::string corrected_text = result.substr(start_pos, end_pos - start_pos);
                
                // Trim leading/trailing whitespace
                corrected_text.erase(0, corrected_text.find_first_not_of(" \t\n\r"));
                corrected_text.erase(corrected_text.find_last_not_of(" \t\n\r") + 1);
                
                // Output just the corrected text
                std::cout << corrected_text << std::endl;
            } else {
                // If markers aren't found, output the full result
                std::cout << result << std::endl;
            }
            
            return 0;
            
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception during inference: " << e.what() << std::endl;
            return 1;
        }
    }
    else {
        std::cerr << "ERROR: Unknown command: " << command << std::endl;
        return 1;
    }
    
    return 0;
}