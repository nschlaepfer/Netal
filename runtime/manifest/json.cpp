/*
 * NeuralStudio runtime - minimal JSON reader
 * SPDX-License-Identifier: MIT
 */

#include "json.h"

#include <cmath>
#include <cstdio>
#include <sstream>

namespace neuralstudio::json {

const Value Value::kNull;
const std::vector<Value> Value::kEmptyArray;

Value Value::Boolean(bool value) {
    Value v;
    v.type_ = Type::Boolean;
    v.bool_ = value;
    return v;
}

Value Value::Number(double value) {
    Value v;
    v.type_ = Type::Number;
    v.number_ = value;
    return v;
}

Value Value::String(std::string value) {
    Value v;
    v.type_ = Type::String;
    v.string_ = std::move(value);
    return v;
}

Value Value::Array() {
    Value v;
    v.type_ = Type::Array;
    return v;
}

Value Value::Object() {
    Value v;
    v.type_ = Type::Object;
    return v;
}

bool Value::AsBool(bool fallback) const {
    return type_ == Type::Boolean ? bool_ : fallback;
}

double Value::AsNumber(double fallback) const {
    return type_ == Type::Number ? number_ : fallback;
}

std::string Value::AsString(const std::string& fallback) const {
    return type_ == Type::String ? string_ : fallback;
}

const Value& Value::operator[](const std::string& key) const {
    if (type_ != Type::Object) {
        return kNull;
    }
    const auto it = object_.find(key);
    return it == object_.end() ? kNull : it->second;
}

bool Value::Has(const std::string& key) const {
    return type_ == Type::Object && object_.find(key) != object_.end();
}

const std::vector<Value>& Value::items() const {
    return type_ == Type::Array ? array_ : kEmptyArray;
}

std::size_t Value::size() const {
    if (type_ == Type::Array) { return array_.size(); }
    if (type_ == Type::Object) { return object_.size(); }
    return 0;
}

void Value::Set(const std::string& key, Value value) {
    if (type_ != Type::Object) {
        type_ = Type::Object;
    }
    object_[key] = std::move(value);
}

void Value::Push(Value value) {
    if (type_ != Type::Array) {
        type_ = Type::Array;
    }
    array_.push_back(std::move(value));
}

namespace {

class Parser {
public:
    Parser(const std::string& text, ParseError& error) : text_(text), error_(error) {}

    bool Run(Value& out_value) {
        SkipWhitespace();
        if (!ParseValue(out_value)) {
            return false;
        }
        SkipWhitespace();
        if (pos_ != text_.size()) {
            return Fail("Unexpected trailing characters");
        }
        return true;
    }

private:
    bool AtEnd() const { return pos_ >= text_.size(); }
    char Peek() const { return text_[pos_]; }

    bool Fail(const std::string& message) {
        error_.ok = false;
        error_.position = pos_;
        error_.message = message;

        std::size_t line = 1;
        std::size_t column = 1;
        for (std::size_t i = 0; i < pos_ && i < text_.size(); ++i) {
            if (text_[i] == '\n') {
                ++line;
                column = 1;
            } else {
                ++column;
            }
        }
        error_.line = line;
        error_.column = column;
        return false;
    }

    void SkipWhitespace() {
        while (!AtEnd()) {
            const char c = Peek();
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    bool ParseValue(Value& out_value) {
        if (AtEnd()) { return Fail("Unexpected end of input"); }
        const char c = Peek();
        switch (c) {
        case '{': return ParseObject(out_value);
        case '[': return ParseArray(out_value);
        case '"': {
            std::string s;
            if (!ParseString(s)) { return false; }
            out_value = Value::String(std::move(s));
            return true;
        }
        case 't': case 'f': return ParseBool(out_value);
        case 'n': return ParseNull(out_value);
        default:
            if (c == '-' || (c >= '0' && c <= '9')) {
                return ParseNumber(out_value);
            }
            return Fail("Unexpected character");
        }
    }

    bool ParseObject(Value& out_value) {
        out_value = Value::Object();
        ++pos_;  /* consume '{' */
        SkipWhitespace();
        if (!AtEnd() && Peek() == '}') {
            ++pos_;
            return true;
        }
        while (true) {
            SkipWhitespace();
            if (AtEnd() || Peek() != '"') {
                return Fail("Expected a string key");
            }
            std::string key;
            if (!ParseString(key)) { return false; }
            SkipWhitespace();
            if (AtEnd() || Peek() != ':') {
                return Fail("Expected ':' after key");
            }
            ++pos_;
            SkipWhitespace();
            Value member;
            if (!ParseValue(member)) { return false; }
            out_value.Set(key, std::move(member));
            SkipWhitespace();
            if (AtEnd()) { return Fail("Unterminated object"); }
            const char c = Peek();
            if (c == ',') {
                ++pos_;
                continue;
            }
            if (c == '}') {
                ++pos_;
                return true;
            }
            return Fail("Expected ',' or '}' in object");
        }
    }

    bool ParseArray(Value& out_value) {
        out_value = Value::Array();
        ++pos_;  /* consume '[' */
        SkipWhitespace();
        if (!AtEnd() && Peek() == ']') {
            ++pos_;
            return true;
        }
        while (true) {
            SkipWhitespace();
            Value element;
            if (!ParseValue(element)) { return false; }
            out_value.Push(std::move(element));
            SkipWhitespace();
            if (AtEnd()) { return Fail("Unterminated array"); }
            const char c = Peek();
            if (c == ',') {
                ++pos_;
                continue;
            }
            if (c == ']') {
                ++pos_;
                return true;
            }
            return Fail("Expected ',' or ']' in array");
        }
    }

    bool ParseString(std::string& out_string) {
        out_string.clear();
        ++pos_;  /* consume opening quote */
        while (!AtEnd()) {
            const char c = text_[pos_++];
            if (c == '"') {
                return true;
            }
            if (c == '\\') {
                if (AtEnd()) { return Fail("Unterminated escape"); }
                const char escape = text_[pos_++];
                switch (escape) {
                case '"':  out_string.push_back('"'); break;
                case '\\': out_string.push_back('\\'); break;
                case '/':  out_string.push_back('/'); break;
                case 'b':  out_string.push_back('\b'); break;
                case 'f':  out_string.push_back('\f'); break;
                case 'n':  out_string.push_back('\n'); break;
                case 'r':  out_string.push_back('\r'); break;
                case 't':  out_string.push_back('\t'); break;
                case 'u': {
                    if (!ParseUnicodeEscape(out_string)) { return false; }
                    break;
                }
                default:
                    return Fail("Invalid escape sequence");
                }
            } else {
                out_string.push_back(c);
            }
        }
        return Fail("Unterminated string");
    }

    bool ParseUnicodeEscape(std::string& out_string) {
        std::uint32_t code = 0;
        if (!ReadHex4(code)) { return false; }

        /* Surrogate pair handling, so manifest text outside the BMP is read
           correctly rather than corrupted. */
        if (code >= 0xD800 && code <= 0xDBFF) {
            if (pos_ + 1 >= text_.size() || text_[pos_] != '\\' || text_[pos_ + 1] != 'u') {
                return Fail("Expected a low surrogate");
            }
            pos_ += 2;
            std::uint32_t low = 0;
            if (!ReadHex4(low)) { return false; }
            if (low < 0xDC00 || low > 0xDFFF) {
                return Fail("Invalid low surrogate");
            }
            code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
        }

        AppendUtf8(code, out_string);
        return true;
    }

    bool ReadHex4(std::uint32_t& out_code) {
        if (pos_ + 4 > text_.size()) { return Fail("Truncated \\u escape"); }
        std::uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = text_[pos_++];
            value <<= 4;
            if (c >= '0' && c <= '9') { value |= static_cast<std::uint32_t>(c - '0'); }
            else if (c >= 'a' && c <= 'f') { value |= static_cast<std::uint32_t>(c - 'a' + 10); }
            else if (c >= 'A' && c <= 'F') { value |= static_cast<std::uint32_t>(c - 'A' + 10); }
            else { return Fail("Invalid hex digit in \\u escape"); }
        }
        out_code = value;
        return true;
    }

    static void AppendUtf8(std::uint32_t code, std::string& out) {
        if (code <= 0x7F) {
            out.push_back(static_cast<char>(code));
        } else if (code <= 0x7FF) {
            out.push_back(static_cast<char>(0xC0 | (code >> 6)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        } else if (code <= 0xFFFF) {
            out.push_back(static_cast<char>(0xE0 | (code >> 12)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (code >> 18)));
            out.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        }
    }

    bool ParseNumber(Value& out_value) {
        const std::size_t start = pos_;
        if (!AtEnd() && Peek() == '-') { ++pos_; }
        while (!AtEnd()) {
            const char c = Peek();
            if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' ||
                c == '+' || c == '-') {
                ++pos_;
            } else {
                break;
            }
        }
        const std::string token = text_.substr(start, pos_ - start);
        try {
            out_value = Value::Number(std::stod(token));
        } catch (...) {
            return Fail("Invalid number");
        }
        return true;
    }

    bool ParseBool(Value& out_value) {
        if (text_.compare(pos_, 4, "true") == 0) {
            pos_ += 4;
            out_value = Value::Boolean(true);
            return true;
        }
        if (text_.compare(pos_, 5, "false") == 0) {
            pos_ += 5;
            out_value = Value::Boolean(false);
            return true;
        }
        return Fail("Invalid literal");
    }

    bool ParseNull(Value& out_value) {
        if (text_.compare(pos_, 4, "null") == 0) {
            pos_ += 4;
            out_value = Value::Null();
            return true;
        }
        return Fail("Invalid literal");
    }

    const std::string& text_;
    ParseError& error_;
    std::size_t pos_ = 0;
};

}  // namespace

bool Parse(const std::string& text, Value& out_value, ParseError& out_error) {
    out_error = ParseError{};
    Parser parser(text, out_error);
    return parser.Run(out_value);
}

}  // namespace neuralstudio::json
