/*
 * NeuralStudio runtime - minimal JSON reader
 * SPDX-License-Identifier: MIT
 *
 * A small, dependency-free JSON parser. The runtime must read package manifests
 * in native code without pulling a third-party library into the process that
 * owns the GPU, so this exists rather than a vendored parser.
 *
 * Scope: enough of RFC 8259 to read manifests correctly - objects, arrays,
 * strings with escapes, numbers, booleans, and null. It is a reader only; it
 * does not serialize. Duplicate keys keep the last value, matching common
 * parser behavior.
 */

#ifndef NEURALSTUDIO_RUNTIME_JSON_H
#define NEURALSTUDIO_RUNTIME_JSON_H

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace neuralstudio::json {

enum class Type { Null, Boolean, Number, String, Array, Object };

class Value {
public:
    Value() : type_(Type::Null) {}

    static Value Null() { return Value(); }
    static Value Boolean(bool value);
    static Value Number(double value);
    static Value String(std::string value);
    static Value Array();
    static Value Object();

    Type type() const { return type_; }

    bool is_null() const { return type_ == Type::Null; }
    bool is_bool() const { return type_ == Type::Boolean; }
    bool is_number() const { return type_ == Type::Number; }
    bool is_string() const { return type_ == Type::String; }
    bool is_array() const { return type_ == Type::Array; }
    bool is_object() const { return type_ == Type::Object; }

    /* Typed accessors with defaults. These never throw, because manifest
       reading should degrade to a named validation error, not an exception. */
    bool AsBool(bool fallback = false) const;
    double AsNumber(double fallback = 0.0) const;
    std::string AsString(const std::string& fallback = "") const;

    /* Object member access. Returns a null value when absent, so chained reads
       of optional manifest fields stay simple. */
    const Value& operator[](const std::string& key) const;
    bool Has(const std::string& key) const;

    /* Array access. */
    const std::vector<Value>& items() const;
    std::size_t size() const;

    /* Mutation, used by the parser and by tests. */
    void Set(const std::string& key, Value value);
    void Push(Value value);

private:
    Type type_;
    bool bool_ = false;
    double number_ = 0.0;
    std::string string_;
    std::vector<Value> array_;
    std::map<std::string, Value> object_;

    static const Value kNull;
    static const std::vector<Value> kEmptyArray;
};

struct ParseError {
    bool ok = true;
    std::size_t position = 0;
    std::size_t line = 0;
    std::size_t column = 0;
    std::string message;
};

/* Parses text into out_value. On failure returns false and fills out_error. */
bool Parse(const std::string& text, Value& out_value, ParseError& out_error);

}  // namespace neuralstudio::json

#endif  /* NEURALSTUDIO_RUNTIME_JSON_H */
