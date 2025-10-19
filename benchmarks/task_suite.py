"""Standardized task suite for benchmarking context engineering.

This module provides a collection of 50+ standardized tasks across different
categories and difficulty levels for comprehensive benchmarking.
"""

from .framework import BenchmarkTask, TaskCategory, TaskDifficulty


def get_standard_task_suite() -> list[BenchmarkTask]:
    """Get the complete standard task suite (50+ tasks).

    Returns:
        List of BenchmarkTask objects
    """
    tasks = []

    # ========== DEBUGGING TASKS (10 tasks) ==========

    tasks.append(BenchmarkTask(
        name="Debug Import Error",
        description="Fix a missing import causing ModuleNotFoundError",
        category=TaskCategory.DEBUGGING,
        difficulty=TaskDifficulty.EASY,
        prompt="The code raises 'ModuleNotFoundError: No module named requests'. Fix the import issue.",
        expected_outcome="Code runs without import errors",
        estimated_time_seconds=120
    ))

    tasks.append(BenchmarkTask(
        name="Debug Null Pointer",
        description="Fix AttributeError from accessing None attribute",
        category=TaskCategory.DEBUGGING,
        difficulty=TaskDifficulty.EASY,
        prompt="Fix the AttributeError: 'NoneType' object has no attribute 'get' in the user authentication code.",
        expected_outcome="No AttributeError, proper null checking implemented",
        estimated_time_seconds=180
    ))

    tasks.append(BenchmarkTask(
        name="Debug Infinite Loop",
        description="Identify and fix an infinite loop in iteration",
        category=TaskCategory.DEBUGGING,
        difficulty=TaskDifficulty.MEDIUM,
        prompt="The while loop in process_queue() never terminates. Find and fix the issue.",
        expected_outcome="Loop terminates correctly with proper exit condition",
        estimated_time_seconds=240
    ))

    tasks.append(BenchmarkTask(
        name="Debug Memory Leak",
        description="Find and fix a memory leak in caching code",
        category=TaskCategory.DEBUGGING,
        difficulty=TaskDifficulty.HARD,
        prompt="The application's memory usage grows indefinitely. Find the memory leak in the caching system.",
        expected_outcome="Memory usage stable, cache properly bounded",
        estimated_time_seconds=600
    ))

    tasks.append(BenchmarkTask(
        name="Debug Race Condition",
        description="Fix a race condition in multi-threaded code",
        category=TaskCategory.DEBUGGING,
        difficulty=TaskDifficulty.EXPERT,
        prompt="Intermittent failures occur in multi-threaded file processor. Find and fix the race condition.",
        expected_outcome="Thread-safe implementation, no race conditions",
        estimated_time_seconds=900
    ))

    # Additional debugging tasks
    for i, (name, difficulty, time) in enumerate([
        ("Debug JSON Parsing Error", TaskDifficulty.EASY, 120),
        ("Debug SQL Query Error", TaskDifficulty.MEDIUM, 300),
        ("Debug Async/Await Issue", TaskDifficulty.HARD, 480),
        ("Debug Performance Bottleneck", TaskDifficulty.HARD, 540),
        ("Debug Security Vulnerability", TaskDifficulty.EXPERT, 720)
    ], start=6):
        tasks.append(BenchmarkTask(
            name=name,
            category=TaskCategory.DEBUGGING,
            difficulty=difficulty,
            prompt=f"Debug task {i}: {name}",
            expected_outcome="Issue resolved",
            estimated_time_seconds=time
        ))

    # ========== FEATURE IMPLEMENTATION TASKS (10 tasks) ==========

    tasks.append(BenchmarkTask(
        name="Implement User Authentication",
        description="Add basic username/password authentication",
        category=TaskCategory.FEATURE_IMPLEMENTATION,
        difficulty=TaskDifficulty.MEDIUM,
        prompt="Implement a user authentication system with login, logout, and session management.",
        expected_outcome="Working authentication with secure password storage",
        estimated_time_seconds=600
    ))

    tasks.append(BenchmarkTask(
        name="Implement REST API Endpoint",
        description="Create a new REST API endpoint",
        category=TaskCategory.FEATURE_IMPLEMENTATION,
        difficulty=TaskDifficulty.MEDIUM,
        prompt="Create a REST API endpoint /api/users that supports GET, POST, PUT, DELETE operations.",
        expected_outcome="CRUD operations work correctly with proper HTTP status codes",
        estimated_time_seconds=480
    ))

    tasks.append(BenchmarkTask(
        name="Implement File Upload",
        description="Add file upload functionality",
        category=TaskCategory.FEATURE_IMPLEMENTATION,
        difficulty=TaskDifficulty.MEDIUM,
        prompt="Implement file upload with validation (file type, size limits) and storage.",
        expected_outcome="Files upload successfully with proper validation",
        estimated_time_seconds=420
    ))

    tasks.append(BenchmarkTask(
        name="Implement Pagination",
        description="Add pagination to a list endpoint",
        category=TaskCategory.FEATURE_IMPLEMENTATION,
        difficulty=TaskDifficulty.EASY,
        prompt="Add pagination to the /api/items endpoint with page and limit parameters.",
        expected_outcome="Pagination works correctly with total count",
        estimated_time_seconds=240
    ))

    tasks.append(BenchmarkTask(
        name="Implement Search Functionality",
        description="Add full-text search to the application",
        category=TaskCategory.FEATURE_IMPLEMENTATION,
        difficulty=TaskDifficulty.HARD,
        prompt="Implement full-text search across multiple fields with ranking and filters.",
        expected_outcome="Search returns relevant results ranked by relevance",
        estimated_time_seconds=720
    ))

    # Additional feature tasks
    for i, (name, difficulty, time) in enumerate([
        ("Implement Email Notifications", TaskDifficulty.MEDIUM, 360),
        ("Implement Caching Layer", TaskDifficulty.MEDIUM, 420),
        ("Implement Rate Limiting", TaskDifficulty.HARD, 480),
        ("Implement Real-time Updates (WebSocket)", TaskDifficulty.EXPERT, 900),
        ("Implement Data Export (CSV/JSON)", TaskDifficulty.EASY, 300)
    ], start=6):
        tasks.append(BenchmarkTask(
            name=name,
            category=TaskCategory.FEATURE_IMPLEMENTATION,
            difficulty=difficulty,
            prompt=f"Feature task {i}: {name}",
            expected_outcome="Feature works as specified",
            estimated_time_seconds=time
        ))

    # ========== REFACTORING TASKS (10 tasks) ==========

    tasks.append(BenchmarkTask(
        name="Extract Duplicate Code",
        description="Remove code duplication by extracting common functions",
        category=TaskCategory.REFACTORING,
        difficulty=TaskDifficulty.EASY,
        prompt="Extract the duplicate validation logic into a reusable function.",
        expected_outcome="No code duplication, single source of truth",
        estimated_time_seconds=180
    ))

    tasks.append(BenchmarkTask(
        name="Simplify Complex Function",
        description="Break down a complex function into smaller parts",
        category=TaskCategory.REFACTORING,
        difficulty=TaskDifficulty.MEDIUM,
        prompt="The process_order() function is 200 lines long. Break it into smaller, focused functions.",
        expected_outcome="Function split into logical components, each <50 lines",
        estimated_time_seconds=360
    ))

    tasks.append(BenchmarkTask(
        name="Improve Variable Names",
        description="Rename variables for better clarity",
        category=TaskCategory.REFACTORING,
        difficulty=TaskDifficulty.EASY,
        prompt="Rename variables (x, data, tmp, result) to descriptive names throughout the module.",
        expected_outcome="All variables have clear, descriptive names",
        estimated_time_seconds=240
    ))

    tasks.append(BenchmarkTask(
        name="Convert to Object-Oriented",
        description="Refactor procedural code to OOP",
        category=TaskCategory.REFACTORING,
        difficulty=TaskDifficulty.HARD,
        prompt="Convert the procedural data processing code to an object-oriented design with classes.",
        expected_outcome="Clean OOP design with proper encapsulation",
        estimated_time_seconds=600
    ))

    tasks.append(BenchmarkTask(
        name="Apply Design Pattern",
        description="Refactor to use appropriate design pattern",
        category=TaskCategory.REFACTORING,
        difficulty=TaskDifficulty.EXPERT,
        prompt="Refactor the notification system to use the Observer pattern for better extensibility.",
        expected_outcome="Observer pattern correctly implemented",
        estimated_time_seconds=720
    ))

    # Additional refactoring tasks
    for i, (name, difficulty, time) in enumerate([
        ("Remove Dead Code", TaskDifficulty.EASY, 120),
        ("Consolidate Conditional Logic", TaskDifficulty.MEDIUM, 300),
        ("Extract Configuration", TaskDifficulty.MEDIUM, 240),
        ("Improve Error Handling", TaskDifficulty.MEDIUM, 360),
        ("Modularize Monolithic Function", TaskDifficulty.HARD, 480)
    ], start=6):
        tasks.append(BenchmarkTask(
            name=name,
            category=TaskCategory.REFACTORING,
            difficulty=difficulty,
            prompt=f"Refactoring task {i}: {name}",
            expected_outcome="Code improved and maintainable",
            estimated_time_seconds=time
        ))

    # ========== TESTING TASKS (10 tasks) ==========

    tasks.append(BenchmarkTask(
        name="Write Unit Tests for Function",
        description="Add comprehensive unit tests",
        category=TaskCategory.TESTING,
        difficulty=TaskDifficulty.EASY,
        prompt="Write unit tests for the calculate_discount() function covering all edge cases.",
        expected_outcome="Tests cover normal, edge, and error cases",
        estimated_time_seconds=240
    ))

    tasks.append(BenchmarkTask(
        name="Write Integration Tests",
        description="Create integration tests for API",
        category=TaskCategory.TESTING,
        difficulty=TaskDifficulty.MEDIUM,
        prompt="Write integration tests for the /api/orders endpoint testing create, read, update, delete.",
        expected_outcome="Integration tests cover all CRUD operations",
        estimated_time_seconds=420
    ))

    tasks.append(BenchmarkTask(
        name="Add Test Coverage",
        description="Increase test coverage to 90%",
        category=TaskCategory.TESTING,
        difficulty=TaskDifficulty.MEDIUM,
        prompt="Current coverage is 65%. Add tests to reach 90% coverage.",
        expected_outcome="Test coverage ≥ 90%",
        estimated_time_seconds=600
    ))

    tasks.append(BenchmarkTask(
        name="Write Mocks for External Services",
        description="Mock external API calls in tests",
        category=TaskCategory.TESTING,
        difficulty=TaskDifficulty.MEDIUM,
        prompt="Create mocks for the payment gateway API to enable offline testing.",
        expected_outcome="Tests run without external dependencies",
        estimated_time_seconds=360
    ))

    tasks.append(BenchmarkTask(
        name="Add Performance Tests",
        description="Create performance/load tests",
        category=TaskCategory.TESTING,
        difficulty=TaskDifficulty.HARD,
        prompt="Write performance tests to ensure the API handles 1000 requests/second.",
        expected_outcome="Performance tests validate throughput requirements",
        estimated_time_seconds=540
    ))

    # Additional testing tasks
    for i, (name, difficulty, time) in enumerate([
        ("Write Parameterized Tests", TaskDifficulty.EASY, 180),
        ("Add Fixture Setup", TaskDifficulty.EASY, 200),
        ("Write End-to-End Tests", TaskDifficulty.HARD, 720),
        ("Add Property-Based Tests", TaskDifficulty.EXPERT, 600),
        ("Create Test Data Factory", TaskDifficulty.MEDIUM, 300)
    ], start=6):
        tasks.append(BenchmarkTask(
            name=name,
            category=TaskCategory.TESTING,
            difficulty=difficulty,
            prompt=f"Testing task {i}: {name}",
            expected_outcome="Tests implemented correctly",
            estimated_time_seconds=time
        ))

    # ========== SECURITY TASKS (5 tasks) ==========

    tasks.append(BenchmarkTask(
        name="Fix SQL Injection",
        description="Secure SQL queries against injection",
        category=TaskCategory.SECURITY,
        difficulty=TaskDifficulty.MEDIUM,
        prompt="The user search query is vulnerable to SQL injection. Fix with parameterized queries.",
        expected_outcome="No SQL injection possible, using prepared statements",
        estimated_time_seconds=300
    ))

    tasks.append(BenchmarkTask(
        name="Add Input Validation",
        description="Validate and sanitize user input",
        category=TaskCategory.SECURITY,
        difficulty=TaskDifficulty.EASY,
        prompt="Add input validation to prevent XSS and injection attacks.",
        expected_outcome="All inputs validated and sanitized",
        estimated_time_seconds=240
    ))

    tasks.append(BenchmarkTask(
        name="Implement CSRF Protection",
        description="Add CSRF tokens to forms",
        category=TaskCategory.SECURITY,
        difficulty=TaskDifficulty.MEDIUM,
        prompt="Implement CSRF protection for all POST/PUT/DELETE endpoints.",
        expected_outcome="CSRF tokens generated and validated",
        estimated_time_seconds=360
    ))

    tasks.append(BenchmarkTask(
        name="Secure Password Storage",
        description="Implement proper password hashing",
        category=TaskCategory.SECURITY,
        difficulty=TaskDifficulty.MEDIUM,
        prompt="Passwords are stored in plaintext. Implement bcrypt hashing.",
        expected_outcome="Passwords securely hashed with bcrypt",
        estimated_time_seconds=300
    ))

    tasks.append(BenchmarkTask(
        name="Add Rate Limiting for Security",
        description="Prevent brute force attacks",
        category=TaskCategory.SECURITY,
        difficulty=TaskDifficulty.HARD,
        prompt="Add rate limiting to login endpoint to prevent brute force attacks.",
        expected_outcome="Rate limiting active, blocks excessive attempts",
        estimated_time_seconds=420
    ))

    # ========== OPTIMIZATION TASKS (5 tasks) ==========

    tasks.append(BenchmarkTask(
        name="Optimize Database Queries",
        description="Reduce N+1 query problem",
        category=TaskCategory.OPTIMIZATION,
        difficulty=TaskDifficulty.MEDIUM,
        prompt="The user list view makes N+1 queries. Optimize with eager loading.",
        expected_outcome="Single query instead of N+1",
        estimated_time_seconds=360
    ))

    tasks.append(BenchmarkTask(
        name="Add Caching for Performance",
        description="Cache expensive computations",
        category=TaskCategory.OPTIMIZATION,
        difficulty=TaskDifficulty.MEDIUM,
        prompt="The dashboard calculation is slow. Add caching to improve performance.",
        expected_outcome="Response time reduced by ≥50%",
        estimated_time_seconds=300
    ))

    tasks.append(BenchmarkTask(
        name="Optimize Algorithm Complexity",
        description="Improve O(n²) to O(n log n)",
        category=TaskCategory.OPTIMIZATION,
        difficulty=TaskDifficulty.HARD,
        prompt="The sorting algorithm is O(n²). Optimize to O(n log n).",
        expected_outcome="Algorithm complexity improved",
        estimated_time_seconds=480
    ))

    tasks.append(BenchmarkTask(
        name="Reduce Memory Usage",
        description="Optimize memory consumption",
        category=TaskCategory.OPTIMIZATION,
        difficulty=TaskDifficulty.HARD,
        prompt="The data processing uses excessive memory. Optimize to use streaming.",
        expected_outcome="Memory usage reduced by ≥50%",
        estimated_time_seconds=540
    ))

    tasks.append(BenchmarkTask(
        name="Parallelize Processing",
        description="Use multi-threading for speed",
        category=TaskCategory.OPTIMIZATION,
        difficulty=TaskDifficulty.EXPERT,
        prompt="Parallelize the batch processor to utilize multiple CPU cores.",
        expected_outcome="Processing time reduced by ≥60% on multi-core",
        estimated_time_seconds=720
    ))

    # ========== MIXED COMPLEXITY TASKS (5 tasks) ==========

    tasks.append(BenchmarkTask(
        name="Build Complete CRUD Module",
        description="Implement full CRUD with tests",
        category=TaskCategory.MIXED,
        difficulty=TaskDifficulty.HARD,
        prompt="Build a complete product catalog module with CRUD, validation, tests, and documentation.",
        expected_outcome="Full CRUD with ≥90% test coverage",
        estimated_time_seconds=900
    ))

    tasks.append(BenchmarkTask(
        name="Migrate Database Schema",
        description="Update schema and migrate data",
        category=TaskCategory.MIXED,
        difficulty=TaskDifficulty.EXPERT,
        prompt="Add new fields to users table and migrate existing data without downtime.",
        expected_outcome="Schema updated, data migrated, no data loss",
        estimated_time_seconds=720
    ))

    tasks.append(BenchmarkTask(
        name="Fix Critical Production Bug",
        description="Debug and fix under pressure",
        category=TaskCategory.MIXED,
        difficulty=TaskDifficulty.EXPERT,
        prompt="Production payments failing intermittently. Find and fix within 30 minutes.",
        expected_outcome="Bug identified and fixed, payments working",
        estimated_time_seconds=1800
    ))

    tasks.append(BenchmarkTask(
        name="Implement Feature from Spec",
        description="Build feature from requirements doc",
        category=TaskCategory.MIXED,
        difficulty=TaskDifficulty.HARD,
        prompt="Implement the shopping cart feature as described in spec.md.",
        expected_outcome="Feature matches specification completely",
        estimated_time_seconds=1200
    ))

    tasks.append(BenchmarkTask(
        name="Refactor Legacy Code",
        description="Modernize old codebase",
        category=TaskCategory.MIXED,
        difficulty=TaskDifficulty.EXPERT,
        prompt="Refactor the legacy order processing system to use modern patterns while maintaining compatibility.",
        expected_outcome="Code modernized, all tests pass, no regressions",
        estimated_time_seconds=1800
    ))

    return tasks


def get_task_suite_by_category(category: TaskCategory) -> list[BenchmarkTask]:
    """Get all tasks for a specific category.

    Args:
        category: Task category

    Returns:
        List of tasks in that category
    """
    all_tasks = get_standard_task_suite()
    return [t for t in all_tasks if t.category == category]


def get_task_suite_by_difficulty(difficulty: TaskDifficulty) -> list[BenchmarkTask]:
    """Get all tasks of a specific difficulty.

    Args:
        difficulty: Task difficulty level

    Returns:
        List of tasks at that difficulty
    """
    all_tasks = get_standard_task_suite()
    return [t for t in all_tasks if t.difficulty == difficulty]


def get_quick_test_suite() -> list[BenchmarkTask]:
    """Get a small suite for quick testing (10 tasks, easy-medium difficulty).

    Returns:
        List of 10 representative tasks
    """
    all_tasks = get_standard_task_suite()
    # Get 2 tasks from each category (easy or medium)
    quick_tasks = []
    for category in TaskCategory:
        category_tasks = [t for t in all_tasks
                         if t.category == category
                         and t.difficulty in [TaskDifficulty.EASY, TaskDifficulty.MEDIUM]]
        quick_tasks.extend(category_tasks[:2])

    return quick_tasks[:10]
