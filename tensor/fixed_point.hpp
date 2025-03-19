#ifndef FIXED_POINT_HPP
#define FIXED_POINT_HPP

#include <iostream>
#include <cmath>
#include <cstdint>

class FixedPointQ5_10 {
private:
    int16_t value; // 16-bit integer to store the fixed-point value

    static constexpr int fractional_bits = 10;
    static constexpr int fractional_mask = (1 << fractional_bits) - 1;
    static constexpr int max_integer_part = 15; // 2^4 - 1 (5 bits for integer part, including sign bit)
    static constexpr int min_integer_part = -16; // -2^4 (5 bits for integer part, including sign bit)


public:
    // Constructors
    FixedPointQ5_10() : value(0) {}
     FixedPointQ5_10(int integer_part) {
        if (integer_part > max_integer_part) {
            integer_part = max_integer_part;
        } else if (integer_part < min_integer_part) {
            integer_part = min_integer_part;
        }
        value = integer_part << fractional_bits;
    }

    FixedPointQ5_10(float float_value) {
        float max_value = static_cast<float>(max_integer_part + 1) - 1.0f / (1 << fractional_bits);
        float min_value = static_cast<float>(min_integer_part);
        if (float_value > max_value) {
            float_value = max_value;
        } else if (float_value < min_value) {
            float_value = min_value;
        }
        value = static_cast<int16_t>(round(float_value * (1 << fractional_bits)));
    }
    // Conversion to float
    float toFloat() const {
        return static_cast<float>(value) / (1 << fractional_bits);
    }

    // Addition
    FixedPointQ5_10 operator+(const FixedPointQ5_10& other) const {
        FixedPointQ5_10 result;
        result.value = value + other.value;
        return result;
    }

    // Subtraction
    FixedPointQ5_10 operator-(const FixedPointQ5_10& other) const {
        FixedPointQ5_10 result;
        result.value = value - other.value;
        return result;
    }

    // Multiplication
    FixedPointQ5_10 operator*(const FixedPointQ5_10& other) const {
        FixedPointQ5_10 result;
        result.value = (value * other.value) >> fractional_bits;
        return result;
    }

    // Division
    FixedPointQ5_10 operator/(const FixedPointQ5_10& other) const {
        FixedPointQ5_10 result;
        result.value = (value << fractional_bits) / other.value;
        return result;
    }

    // Output stream operator
    friend std::ostream& operator<<(std::ostream& os, const FixedPointQ5_10& fp) {
        os << fp.toFloat();
        return os;
    }
};

#endif // FIXED_POINT_HPP