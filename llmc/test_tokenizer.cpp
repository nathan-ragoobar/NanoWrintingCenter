#include <gtest/gtest.h>
#include "tokenizer.hpp"
#include <fstream>

//UNIT TEST FOR TOKENIZER

class TokenizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::ofstream dict("test_dict.txt");
        dict << "Hello\t0\n";
        dict << "world\t1\n";
        dict << "<|endoftext|>\t2\n";
        dict << "!\t3\n";
        dict << " \t4\n";
        dict << "Test\t5\n";
        dict << "ing\t6\n";
        dict.close();
    }

    void TearDown() override {
        remove("test_dict.txt");
    }

    nano::Tokenizer tokenizer;
};

TEST_F(TokenizerTest, EncodeString) {
    tokenizer.init("test_dict.txt");
    
    std::vector<uint32_t> tokens = tokenizer.encode_string("Hello world!");
    std::vector<uint32_t> expected = {0, 4, 1, 3};
    EXPECT_EQ(tokens, expected);

    tokens = tokenizer.encode_string("Testing");
    expected = {5, 6};
    EXPECT_EQ(tokens, expected);
}

TEST_F(TokenizerTest, DecodeString) {
    tokenizer.init("test_dict.txt");
    
    std::vector<uint32_t> tokens = {0, 4, 1, 3};
    EXPECT_EQ(tokenizer.decode_string(tokens), "Hello world!");

    tokens = {5, 6};
    EXPECT_EQ(tokenizer.decode_string(tokens), "Testing");
}

TEST_F(TokenizerTest, Initialization) {
    tokenizer.init("test_dict.txt");
    EXPECT_TRUE(tokenizer.is_initialized());
    EXPECT_EQ(tokenizer.get_vocab_size(), 7);
    EXPECT_EQ(tokenizer.get_eot_token(), 2);
}

TEST_F(TokenizerTest, EncodeValidTokens) {
    tokenizer.init("test_dict.txt");
    EXPECT_EQ(tokenizer.encode("Hello"), 0);
    EXPECT_EQ(tokenizer.encode("world"), 1);
    EXPECT_EQ(tokenizer.encode("<|endoftext|>"), 2);
    EXPECT_EQ(tokenizer.encode("Test"), 5);
}

TEST_F(TokenizerTest, DecodeValidTokens) {
    tokenizer.init("test_dict.txt");
    EXPECT_EQ(tokenizer.decode(0), "Hello");
    EXPECT_EQ(tokenizer.decode(1), "world");
    EXPECT_EQ(tokenizer.decode(2), "<|endoftext|>");
    EXPECT_EQ(tokenizer.decode(5), "Test");
}

TEST_F(TokenizerTest, InvalidTokens) {
    tokenizer.init("test_dict.txt");
    EXPECT_EQ(tokenizer.encode("nonexistent"), UINT32_MAX);
    EXPECT_EQ(tokenizer.decode(999), "");
}

TEST_F(TokenizerTest, UninitializedTokenizer) {
    EXPECT_EQ(tokenizer.encode("hello"), UINT32_MAX);
    EXPECT_EQ(tokenizer.decode(0), "");
}

TEST_F(TokenizerTest, MissingDictionary) {
    tokenizer.init("nonexistent.txt");
    EXPECT_FALSE(tokenizer.is_initialized());
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}