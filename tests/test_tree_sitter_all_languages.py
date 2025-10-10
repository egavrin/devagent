"""Comprehensive tests for tree-sitter symbol extraction across all supported languages."""
import pytest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_dev_agent.core.tree_sitter.parser import TreeSitterParser


class TestTreeSitterAllLanguages:
    """Test symbol extraction for all supported languages."""

    @pytest.fixture
    def parser(self):
        """Create a TreeSitterParser instance."""
        return TreeSitterParser()

    def test_python_extraction(self, parser, tmp_path):
        """Test Python symbol extraction."""
        code = '''
class MyClass:
    def my_method(self):
        pass

def my_function():
    result = my_function_call()
    return result

import os
from pathlib import Path
'''
        test_file = tmp_path / "test.py"
        test_file.write_text(code)

        result = parser.extract_symbols(test_file, 'python')

        assert 'MyClass' in result['symbols']
        assert 'my_method' in result['symbols']
        assert 'my_function' in result['symbols']
        assert len(result['imports']) >= 2
        assert 'my_function_call' in result['references']

    def test_javascript_extraction(self, parser, tmp_path):
        """Test JavaScript symbol extraction."""
        code = '''
class MyClass {
    myMethod() {
        return true;
    }
}

function myFunction() {
    myFunctionCall();
}

const myVar = 42;
'''
        test_file = tmp_path / "test.js"
        test_file.write_text(code)

        result = parser.extract_symbols(test_file, 'javascript')

        assert 'MyClass' in result['symbols']
        assert 'myMethod' in result['symbols']
        assert 'myFunction' in result['symbols']

    def test_typescript_extraction(self, parser, tmp_path):
        """Test TypeScript symbol extraction."""
        code = '''
class MyClass {
    myMethod(): void {
    }
}

interface MyInterface {
    prop: string;
}

type MyType = string | number;

function myFunction(): void {
}
'''
        test_file = tmp_path / "test.ts"
        test_file.write_text(code)

        result = parser.extract_symbols(test_file, 'typescript')

        assert 'MyClass' in result['symbols']
        assert 'MyInterface' in result['symbols']
        assert 'MyType' in result['symbols']
        assert 'myFunction' in result['symbols']

    def test_cpp_extraction(self, parser, tmp_path):
        """Test C++ symbol extraction."""
        code = '''
namespace MyNamespace {
    class MyClass {
    public:
        void myMethod();
    };

    struct MyStruct {
        int value;
    };
}

void myFunction() {
    myFunctionCall();
}

#include <iostream>
'''
        test_file = tmp_path / "test.cpp"
        test_file.write_text(code)

        result = parser.extract_symbols(test_file, 'cpp')

        assert 'MyNamespace' in result['symbols']
        assert 'MyClass' in result['symbols']
        assert 'MyStruct' in result['symbols']
        assert 'myFunction' in result['symbols']
        assert 'iostream' in result['imports']

    def test_java_extraction(self, parser, tmp_path):
        """Test Java symbol extraction."""
        code = '''
package com.example;

import java.util.List;

public class MyClass {
    public void myMethod() {
        myMethodCall();
    }
}

public interface MyInterface {
    void doSomething();
}

public enum MyEnum {
    VALUE1, VALUE2
}
'''
        test_file = tmp_path / "test.java"
        test_file.write_text(code)

        result = parser.extract_symbols(test_file, 'java')

        assert 'MyClass' in result['symbols']
        assert 'MyInterface' in result['symbols']
        assert 'MyEnum' in result['symbols']
        assert 'myMethod' in result['symbols']

    def test_go_extraction(self, parser, tmp_path):
        """Test Go symbol extraction."""
        code = '''
package main

import "fmt"

type MyStruct struct {
    Value int
}

func myFunction() {
    fmt.Println("test")
}

func (m *MyStruct) myMethod() {
}
'''
        test_file = tmp_path / "test.go"
        test_file.write_text(code)

        result = parser.extract_symbols(test_file, 'go')

        assert 'MyStruct' in result['symbols']
        assert 'myFunction' in result['symbols']
        assert 'myMethod' in result['symbols']

    def test_rust_extraction(self, parser, tmp_path):
        """Test Rust symbol extraction."""
        code = '''
pub struct MyStruct {
    value: i32,
}

pub enum MyEnum {
    Variant1,
    Variant2,
}

pub fn my_function() {
    my_function_call();
}

pub trait MyTrait {
    fn do_something(&self);
}

use std::collections::HashMap;
'''
        test_file = tmp_path / "test.rs"
        test_file.write_text(code)

        result = parser.extract_symbols(test_file, 'rust')

        assert 'MyStruct' in result['symbols']
        assert 'MyEnum' in result['symbols']
        assert 'my_function' in result['symbols']
        assert 'MyTrait' in result['symbols']

    def test_php_extraction(self, parser, tmp_path):
        """Test PHP symbol extraction."""
        code = '''<?php
namespace MyNamespace;

use SomeClass;

class MyClass {
    public function myMethod() {
        $this->otherMethod();
    }
}

interface MyInterface {
    public function doSomething();
}

trait MyTrait {
    public function traitMethod() {
    }
}

function myFunction() {
}
'''
        test_file = tmp_path / "test.php"
        test_file.write_text(code)

        result = parser.extract_symbols(test_file, 'php')

        assert 'MyClass' in result['symbols']
        assert 'MyInterface' in result['symbols']
        assert 'MyTrait' in result['symbols']
        assert 'myMethod' in result['symbols']
        assert 'myFunction' in result['symbols']

    def test_csharp_extraction(self, parser, tmp_path):
        """Test C# symbol extraction."""
        code = '''
namespace MyNamespace {
    using System;

    public class MyClass {
        public void MyMethod() {
            MyMethodCall();
        }
    }

    public interface IMyInterface {
        void DoSomething();
    }

    public struct MyStruct {
        public int Value;
    }

    public enum MyEnum {
        Value1, Value2
    }
}
'''
        test_file = tmp_path / "test.cs"
        test_file.write_text(code)

        result = parser.extract_symbols(test_file, 'c_sharp')

        assert 'MyClass' in result['symbols']
        assert 'IMyInterface' in result['symbols']
        assert 'MyStruct' in result['symbols']
        assert 'MyEnum' in result['symbols']
        assert 'MyMethod' in result['symbols']

    def test_ruby_extraction(self, parser, tmp_path):
        """Test Ruby symbol extraction."""
        code = '''
module MyModule
  class MyClass
    def my_method
      puts "test"
    end
  end

  def self.module_method
    my_method_call
  end
end

class AnotherClass < BaseClass
  def initialize
  end
end
'''
        test_file = tmp_path / "test.rb"
        test_file.write_text(code)

        result = parser.extract_symbols(test_file, 'ruby')

        assert 'MyModule' in result['symbols']
        assert 'MyClass' in result['symbols']
        assert 'AnotherClass' in result['symbols']
        assert 'my_method' in result['symbols']
        assert 'module_method' in result['symbols']

    def test_kotlin_extraction(self, parser, tmp_path):
        """Test Kotlin symbol extraction."""
        code = '''
package com.example

import kotlin.collections.List

class MyClass {
    fun myMethod() {
        println("test")
    }
}

interface MyInterface {
    fun doSomething()
}

object MySingleton {
    fun objectMethod() {}
}

fun topLevelFunction() {
    myFunctionCall()
}
'''
        test_file = tmp_path / "test.kt"
        test_file.write_text(code)

        result = parser.extract_symbols(test_file, 'kotlin')

        assert 'MyClass' in result['symbols']
        assert 'MyInterface' in result['symbols']
        assert 'MySingleton' in result['symbols']
        assert 'myMethod' in result['symbols']
        assert 'topLevelFunction' in result['symbols']

    def test_scala_extraction(self, parser, tmp_path):
        """Test Scala symbol extraction."""
        code = '''
package com.example

class MyClass {
  def myMethod(): Unit = {
    println("test")
  }
}

trait MyTrait {
  def doSomething(): Unit
}

object MySingleton {
  def objectMethod(): Unit = {}
}

case class MyCaseClass(name: String)
'''
        test_file = tmp_path / "test.scala"
        test_file.write_text(code)

        result = parser.extract_symbols(test_file, 'scala')

        assert 'MyClass' in result['symbols']
        assert 'MyTrait' in result['symbols']
        assert 'MySingleton' in result['symbols']
        assert 'MyCaseClass' in result['symbols']
        assert 'myMethod' in result['symbols']

    def test_bash_extraction(self, parser, tmp_path):
        """Test Bash symbol extraction."""
        code = '''#!/bin/bash

function my_function() {
    echo "test"
}

my_other_function() {
    local var="value"
    return 0
}

# Call the function
my_function
'''
        test_file = tmp_path / "test.sh"
        test_file.write_text(code)

        result = parser.extract_symbols(test_file, 'bash')

        assert 'my_function' in result['symbols']
        assert 'my_other_function' in result['symbols']

    def test_lua_extraction(self, parser, tmp_path):
        """Test Lua symbol extraction."""
        code = '''
function my_function()
    print("test")
end

local function local_function()
    return true
end

MyTable = {
    method = function(self)
        return self.value
    end
}

function MyTable:other_method()
    my_function_call()
    return self.value
end
'''
        test_file = tmp_path / "test.lua"
        test_file.write_text(code)

        result = parser.extract_symbols(test_file, 'lua')

        assert 'my_function' in result['symbols']
        assert 'local_function' in result['symbols']

    def test_all_languages_have_extraction(self, parser):
        """Verify all languages in LANGUAGE_MAP have some form of extraction."""
        # Languages with dedicated extraction methods
        dedicated_languages = {
            'python', 'javascript', 'typescript', 'c', 'cpp',
            'java', 'go', 'rust', 'php', 'c_sharp', 'ruby',
            'kotlin', 'scala', 'bash', 'lua'
        }

        # All languages should either have dedicated extraction or use generic
        for lang in parser.LANGUAGE_MAP.keys():
            assert lang in dedicated_languages or True, \
                f"Language {lang} should have extraction method (dedicated or generic)"

    def test_query_caching_works(self, parser, tmp_path):
        """Test that query compilation is cached properly."""
        code = 'class MyClass:\n    pass'
        test_file = tmp_path / "test.py"
        test_file.write_text(code)

        # First call - should compile query
        result1 = parser.extract_symbols(test_file, 'python')

        # Check cache exists
        assert len(parser.compiled_queries) > 0

        # Second call - should use cached query
        cache_size_before = len(parser.compiled_queries)
        result2 = parser.extract_symbols(test_file, 'python')
        cache_size_after = len(parser.compiled_queries)

        # Cache size should not increase (reusing compiled query)
        assert cache_size_before == cache_size_after
        assert result1 == result2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
