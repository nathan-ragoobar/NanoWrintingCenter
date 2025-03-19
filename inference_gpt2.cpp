#ifdef _WIN32
#include "./llmc/unistd.h"  // Include Windows implementation
#else
#include <unistd.h>     // Include POSIX implementation
#endif

#include <iostream>
#include <memory>
#include <condition_variable>
#include <queue>

#include "gpt2.hpp"
//#include "llmc/dataloader.h"
//#include "llmc/tokenizer.h"
#include "nano.hpp"
#include "llmc/tokenizer.hpp"

// Forward declaration of the TokenizerCache class we'll define later
//class TokenizerCache;

// Add this class before your main() function
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

TokenizerCache* tokenizer_cache_ptr = nullptr;

// Thread and synchronization variables
std::thread decode_thread;
std::mutex token_mutex;
std::condition_variable cv;
std::queue<int> token_queue;
bool done = false;

// Function to start the decoder thread
void start_decoder_thread() {
    decode_thread = std::thread([]() {
        while (!done) {
            std::unique_lock<std::mutex> lock(token_mutex);
            cv.wait(lock, [&]{ return !token_queue.empty() || done; });
            
            if (!token_queue.empty()) {
                int token = token_queue.front();
                token_queue.pop();
                lock.unlock();
                
                // Decode and print in background
                if (tokenizer_cache_ptr) {
                    std::string text = tokenizer_cache_ptr->decode(token);
                    std::cout << text << std::flush;
                } else {
                    std::cout << "[" << token << "]" << std::flush;
                }
            }
        }
    });
}

// sampler

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



void print_usage(const char* program_name) {
  printf("Usage: %s [options]\n", program_name);
  printf("Options:\n");
  printf("  --model PATH      Path to model weights file (default: gpt2_124M.bin)\n");
  printf("  --genlen N        Number of tokens to generate (default: 64)\n");
  printf("  --help            Display this help message\n");
}

int main(int argc, char** argv) {

    // Default model path
    const char* model_path = "./gpt2_124M100Steps.bin";
    int genT = 64;  // Default generation length
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--genlen") == 0 && i + 1 < argc) {
            genT = atoi(argv[++i]);
            if (genT <= 0 || genT > 1024) {
                fprintf(stderr, "Error: Generation length must be between 1 and 1024\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    struct timespec start, end;

    // Print configuration
    printf("Configuration:\n");
    printf("  Model path: %s\n", model_path);
    printf("  Generation length: %d tokens\n\n", genT);

    // Load the model using the path from command line
    gpt2::GPT2 model;
    printf("Loading model from %s...\n", model_path);
    if (model.BuildFromCheckpoint(model_path) == false) {
        fprintf(stderr, "Error: Failed to load model from %s\n", model_path);
        return 1;
    }
    printf("Model loaded successfully\n");

    // Rest of the code remains the same
    int B = 4;
    int T = 64;

    // build the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    
    nano::GPT2Tokenizer tokenizer_gpt2("vocab.bpe", "encoder.json");

    // Initialize the tokenizer cache and set the global pointer
    TokenizerCache tokenizer_cache(tokenizer_gpt2);
    tokenizer_cache_ptr = &tokenizer_cache;

    // Start the decoder thread
    start_decoder_thread();

    // some memory for generating samples from the model
    unsigned long long rng_state = 1337;
    int* gen_tokens = (int*)mallocCheck(B * T * sizeof(int));
    //const int genT = 64;  // number of steps of inference we will do

    int V = model.config.vocab_size;
    std::unique_ptr<float[]> logit = std::make_unique<float[]>(B * T * V);
    std::unique_ptr<float[]> prob = std::make_unique<float[]>(B * T * V);
    //nn::Parameter label(nn::DT_FLOAT, B * T * V);
    nn::Softmax softmax;

    // Ask the user for input
    std::string input;
    std::cout << "Enter a prompt: ";
    std::getline(std::cin, input);

    // Variable to keep track of the last printed length
    size_t last_printed = 0;

    //Start the timer for inference
    clock_gettime(CLOCK_MONOTONIC, &start);

    //Tokenize the input
    //std::vector<uint32_t> input_tokens = tokenizer_nano.encode_string(input);
    std::vector<int> input_tokens = tokenizer_gpt2.encode(input);

    //Print the tokens
    std::cout << "Input tokens: ";
    for (int i = 0; i < input_tokens.size(); i++) {
        std::cout << input_tokens[i] << " ";
    }
    std::cout << "\n";
    
    // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
    // Initialize gen_tokens with the input tokens and pad with EOT token
    for (int i = 0; i < B * T; ++i) {
        if (i < input_tokens.size()) {
            gen_tokens[i] = input_tokens[i];
        } else {
            gen_tokens[i] = tokenizer.eot_token;
        }
    }
      
     if(1){
      //New Decoder
      // now sample from the model autoregressively
      printf("generating:\n---\n");
      for (int t = input_tokens.size(); t < genT; t++) {
        // note that inference is very wasteful here because for each token
        // we re-calculate the forward pass for all of (B,T) positions from
        // scratch but the inference here is just for sanity checking anyway and
        // we can maybe optimize a bit more later, with careful tests
        auto gen_tokens_2d = TTypes<int>::ConstMatrix(gen_tokens, B, T);
        auto logit_3d = Make3DTensor(logit.get(), B, T, V);
        model.gpt2_->Forward(gen_tokens_2d, logit_3d);
        auto logit_2d = MakeConstMatrix(logit.get(), B * T, V);
        auto prob_2d = MakeMatrix(prob.get(), B * T, V);
        softmax.Forward(logit_2d, prob_2d);
        // furthermore, below we're only using b=0 (i.e. the first row) of all B
        // rows we're in principle running B "inference streams" in parallel
        // here but only using position 0 get the Vp-dimensional vector probs[0,
        // t-1, :]
        float* probs = prob.get() + (t - 1) * V;
        float coin = random_f32(&rng_state);
        // note we're only sampling from the first V elements, ignoring padding
        // (the probabilities in the padded region should be zero anyway)
        int next_token = sample_mult(probs, model.config.vocab_size, coin);
        gen_tokens[t] = next_token;
        
        // Push token to queue for background processing
        {
          std::lock_guard<std::mutex> lock(token_mutex);
          token_queue.push(next_token);
      }
      cv.notify_one();
  }
  
  // Wait for decoder thread to finish processing the queue
  {
      std::unique_lock<std::mutex> lock(token_mutex);
      while (!token_queue.empty()) {
          lock.unlock();
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
          lock.lock();
      }
  }
  
  // Clean up when done
  done = true;
  cv.notify_all();
  if (decode_thread.joinable()) {
      decode_thread.join();
  }
  
  printf("\n---\n");

clock_gettime(CLOCK_MONOTONIC, &end);

double time_elapsed_s =
    (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

// Calculate tokens generated (excluding input tokens)
int tokens_generated = genT - input_tokens.size();
double tokens_per_second = tokens_generated / time_elapsed_s;

printf("Inference performance:\n");
printf("- Total time: %.2f ms\n", time_elapsed_s * 1000);
printf("- Tokens generated: %d\n", tokens_generated);
printf("- Throughput: %.2f tokens/second\n", tokens_per_second);
printf("- Latency: %.2f ms/token\n", (time_elapsed_s * 1000) / tokens_generated);

}
else{

  //Original decoder
  // now sample from the model autoregressively
  printf("generating:\n---\n");

    // Print the input prompt first
    for (size_t i = 0; i < input_tokens.size(); i++) {
      if (tokenizer.init_ok) {
        const char* token_str = tokenizer_decode(&tokenizer, input_tokens[i]);
        safe_printf(token_str);
      } else {
        // fall back to printing the token id
        printf("%d ", input_tokens[i]);
      }
    }
    fflush(stdout);

    clock_gettime(CLOCK_MONOTONIC, &start);

    // Start generation from the end of the input
    for (int t = input_tokens.size(); t < genT; t++) {
      auto gen_tokens_2d = TTypes<int>::ConstMatrix(gen_tokens, B, T);
      auto logit_3d = Make3DTensor(logit.get(), B, T, V);
      model.gpt2_->Forward(gen_tokens_2d, logit_3d);
      auto logit_2d = MakeConstMatrix(logit.get(), B * T, V);
      auto prob_2d = MakeMatrix(prob.get(), B * T, V);
      softmax.Forward(logit_2d, prob_2d);
      
      // Get probabilities for the current position
      float* probs = prob.get() + (t - 1) * V;
      float coin = random_f32(&rng_state);
      
      int next_token = sample_mult(probs, model.config.vocab_size, coin);
      gen_tokens[t] = next_token;
      
      // print the generated token
      if (tokenizer.init_ok) {
        const char* token_str = tokenizer_decode(&tokenizer, next_token);
        safe_printf(token_str);
      } else {
        // fall back to printing the token id
        printf("%d ", next_token);
      }
      fflush(stdout);
    }
    printf("\n---\n");

    // Add after the generation loop:
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    int tokens_generated = genT - input_tokens.size();
    printf("Generated %d tokens in %.3f seconds (%.1f tokens/sec)\n", 
          tokens_generated, time_elapsed_s, tokens_generated/time_elapsed_s);
}

return 0;
}

/*
// START OF VALIDATION CODE
//---------------------------------------------------------------------------------
    const char* tiny_stories_val = "dev/data/tinystories/TinyStories_val.bin";
  
    const char* tiny_shakespeare_val =
           "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    const char* val_tokens = access(tiny_shakespeare_val, F_OK) != -1
           ? tiny_shakespeare_val
           : tiny_stories_val;
    
    nn::Parameter label(nn::DT_FLOAT, B * T * V);
    DataLoader val_loader;
    dataloader_init(&val_loader, val_tokens, B, T, 0, 1, 0);
    // Add initialization here
    //if (!dataloader_init(&val_loader, "val.bin", B, T)) {  // Make sure val.bin exists
      //fprintf(stderr, "Failed to initialize validation dataloader\n");
      //return 1;
  //}
    int val_num_batches = 5;
    bool USE_FAST_SOFTMAX = true;

    for (int step = 0; step <= 10; step++) {
    
      float val_loss = 0.0f;
      dataloader_reset(&val_loader);
      for (int i = 0; i < val_num_batches; i++) {
        dataloader_next_batch(&val_loader);
        float loss = 0.0f;
        auto idx = TTypes<int>::ConstMatrix(val_loader.inputs, B, T);
        if (USE_FAST_SOFTMAX) {
          auto target = TTypes<int>::ConstMatrix(val_loader.targets, B, T);
          auto logit_3d = Make3DTensor(logit.get(), B, T, V);
          model.gpt2_->ForwardCPU(idx, target, logit_3d, &loss);
        } else {
          label.ZeroData();
          nn::OntHot(MakeConstFlat(val_loader.targets, B * T),
                      label.matrix<float>(B * T, V));
          auto label_3d = label.const_tensor_3d<float>(B, T, V);
          auto logit_3d = Make3DTensor(logit.get(), B, T, V);
          model.gpt2_->ForwardGPU(idx, label_3d, logit_3d, &loss);
        }
        val_loss += loss;
      }
      val_loss /= val_num_batches;

      if (step == 0) {
        size_t num_activations = model.gpt2_->NumActivations();
        printf("num_activations: %zu(%zu MB)\n", num_activations,
                num_activations * sizeof(floatX) / 1024 / 1024);
      }
      printf("val loss %f\n", val_loss);
    }

    // Add cleanup
    tokenizer_cache_ptr = nullptr;
    dataloader_free(&val_loader);
    return 0;
      

}

*/