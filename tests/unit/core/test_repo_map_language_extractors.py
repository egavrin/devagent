"""Tests for the regex-based language extractors in RepoMap."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_dev_agent.core.repo_map import FileInfo, RepoMap


def make_repo(tmp_path: Path) -> RepoMap:
    """Helper to instantiate RepoMap without tree-sitter."""
    return RepoMap(root_path=tmp_path, cache_enabled=False, use_tree_sitter=False)


def make_file(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def new_file_info(path: str) -> FileInfo:
    return FileInfo(path=path, size=0, modified_time=0.0)


def test_extract_with_regex_dispatches_known_language(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path)
    called = {}

    def fake_go(path: Path, info: FileInfo) -> None:
        called["go"] = (path, info)

    monkeypatch.setattr(repo, "_extract_go_info", fake_go)

    file_path = make_file(tmp_path, "main.go", "package main")
    repo._extract_with_regex(file_path, new_file_info("main.go"), "go")

    assert "go" in called


def test_extract_with_regex_falls_back_to_generic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path)
    called = {}

    def fake_generic(path: Path, info: FileInfo) -> None:
        called["generic"] = True
        info.symbols.append("FallbackSymbol")

    monkeypatch.setattr(repo, "_extract_generic_info", fake_generic)

    file_path = make_file(tmp_path, "script.sh", "echo 'hello'")
    info = new_file_info("script.sh")
    repo._extract_with_regex(file_path, info, "bash")

    assert called.get("generic") is True
    assert "FallbackSymbol" in info.symbols


def test_extract_go_info_handles_interfaces_and_imports(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    file_path = make_file(
        tmp_path,
        "example.go",
        """
        package analytics

        import (
            "fmt"
            sql "database/sql"
        )

        import "log"

        type Reader interface {
            Read(p []byte) (n int, err error)
        }

        type Worker struct {
            io.Writer
            Buffer
        }

        func processData(ch chan int) {
            go worker(ch)
        }
        """,
    )
    info = new_file_info("example.go")

    repo._extract_go_info(file_path, info)

    assert {"Reader", "Worker", "processData"} <= set(info.symbols)
    assert {"fmt", "database/sql", "log", "package:analytics"} <= set(info.imports)


def test_extract_ruby_info_captures_modules_and_requires(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    file_path = make_file(
        tmp_path,
        "example.rb",
        """
        require "json"

        module Utilities
          def helper; end
        end

        class Report
          include Utilities

          def generate_report!
          end
        end
        """,
    )
    info = new_file_info("example.rb")

    repo._extract_ruby_info(file_path, info)

    assert {"Utilities", "Report", "helper", "generate_report!"} <= set(info.symbols)
    assert "json" in info.imports


def test_extract_kotlin_info_handles_data_classes_and_companion(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    file_path = make_file(
        tmp_path,
        "User.kt",
        """
        package com.example.app

        import kotlin.collections.Map

        data class User(val id: Int, val name: String)

        class Processor {
            companion object Factory {
                fun create(): Processor = Processor()
            }
        }

        fun String.toTitleCase(): String = this.uppercase()

        val logLevel = "INFO"
        """,
    )
    info = new_file_info("User.kt")

    repo._extract_kotlin_info(file_path, info)

    symbols = set(info.symbols)
    assert {"User", "Processor", "create", "logLevel"} <= symbols
    assert "Factory" in symbols  # companion objects should be captured as symbols
    imports = set(info.imports)
    assert "kotlin.collections.Map" in imports
    assert "package:com.example.app" in imports


def test_extract_scala_info_captures_traits_and_case_classes(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    file_path = make_file(
        tmp_path,
        "Example.scala",
        """
        package analytics.core

        import scala.concurrent.Future

        trait Loggable {
            def log(msg: String): Unit
        }

        case class Person(name: String, age: Int)

        object Converters

        def enrich(data: List[Int])(implicit ctx: Context): List[Int] = data
        """,
    )
    info = new_file_info("Example.scala")

    repo._extract_scala_info(file_path, info)

    assert {"Loggable", "Person", "Converters", "log", "enrich"} <= set(info.symbols)
    imports = set(info.imports)
    assert "scala.concurrent.Future" in imports
    assert "package:analytics.core" in imports


def test_extract_swift_info_understands_protocols_and_generics(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    file_path = make_file(
        tmp_path,
        "Graphics.swift",
        """
        import Foundation

        protocol Drawable {
            func draw()
        }

        struct Stack<Element> {
            var items: [Element]
            func push(_ item: Element) {}
        }

        class Canvas: Drawable {
            private func renderScene() {}
        }

        extension String {
            func reversed() -> String { return "" }
        }
        """,
    )
    info = new_file_info("Graphics.swift")

    repo._extract_swift_info(file_path, info)

    assert {
        "Drawable",
        "Stack",
        "Canvas",
        "draw",
        "push",
        "renderScene",
        "reversed",
        "items",
    } <= set(info.symbols)
    assert "Foundation" in info.imports


def test_extract_dart_info_captures_async_functions_and_exports(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    file_path = make_file(
        tmp_path,
        "service.dart",
        """
        import 'package:app/api.dart';
        export 'package:app/models.dart';

        mixin Logger {
          void log(String msg);
        }

        class Service with Logger {
          Future<void> fetchData() async {
            await api.get();
          }
        }
        """,
    )
    info = new_file_info("service.dart")

    repo._extract_dart_info(file_path, info)

    assert {"Logger", "Service", "log", "fetchData"} <= set(info.symbols)
    assert "package:app/api.dart" in info.imports
    assert "package:app/models.dart" in info.exports


def test_extract_lua_info_handles_local_functions_and_requires(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    file_path = make_file(
        tmp_path,
        "module.lua",
        """
        local json = require("json")

        local function helper(x)
            return x * 2
        end

        MyModule = {}

        function MyModule.process(item)
            return item
        end
        """,
    )
    info = new_file_info("module.lua")

    repo._extract_lua_info(file_path, info)

    assert {"helper", "MyModule", "MyModule.process"} <= set(info.symbols)
    assert "json" in info.imports


def test_extract_php_info_records_namespace_and_use_statements(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    file_path = make_file(
        tmp_path,
        "Controller.php",
        """
        <?php
        namespace App\\Http\\Controllers;

        use App\\Contracts\\Renderable;

        trait Singleton {
            public function instance() {}
        }

        interface Renderable {
            public function render(): string;
        }

        class Controller {
            use Singleton;

            public function handle() {}
        }
        """,
    )
    info = new_file_info("Controller.php")

    repo._extract_php_info(file_path, info)

    assert {"Singleton", "Renderable", "Controller", "App\\Http\\Controllers"} <= set(info.symbols)
    assert {"App\\Contracts\\Renderable"} <= set(info.imports)


def test_extract_generic_info_handles_unknown_languages(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    file_path = make_file(
        tmp_path,
        "schema.proto",
        """
        message UserProfile {
            required string name = 1;
        }

        service ProfileService {
            rpc GetProfile (ProfileRequest) returns (ProfileResponse);
        }

        import "google/protobuf/timestamp.proto";
        """,
    )
    info = new_file_info("schema.proto")

    repo._extract_generic_info(file_path, info)

    assert "GetProfile" in info.symbols
    assert "google/protobuf/timestamp.proto" in info.imports
