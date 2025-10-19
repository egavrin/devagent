#!/bin/bash
# Automated test runner for DevAgent project
# Provides multiple test execution profiles and comprehensive reporting

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
COVERAGE_THRESHOLD=${COVERAGE_THRESHOLD:-95}
PARALLEL=${PARALLEL:-auto}
TIMEOUT=${TIMEOUT:-300}

# Print formatted message
print_msg() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Print section header
print_header() {
    echo ""
    print_msg "$BLUE" "========================================"
    print_msg "$BLUE" "$1"
    print_msg "$BLUE" "========================================"
    echo ""
}

# Show usage information
show_usage() {
    cat << EOF
Usage: $0 [PROFILE] [OPTIONS]

Test Profiles:
  fast          Run fast unit tests only (default)
  full          Run all tests including integration
  integration   Run integration tests only
  coverage      Run tests with coverage reporting
  watch         Run tests in watch mode
  benchmark     Run performance benchmarks
  ci            Run full CI test suite

Options:
  --parallel N     Number of parallel workers (default: auto)
  --threshold N    Coverage threshold percentage (default: 95)
  --timeout N      Test timeout in seconds (default: 300)
  --no-cov         Skip coverage reporting
  --verbose        Verbose output
  --markers M      Run tests with specific markers
  --failed         Rerun only failed tests from last run
  --clean          Clean test artifacts before running
  --help           Show this help message

Examples:
  $0 fast                    # Run fast tests
  $0 full --parallel 4       # Run all tests with 4 workers
  $0 coverage --threshold 90 # Run with 90% coverage threshold
  $0 --markers integration   # Run integration tests only
  $0 --failed                # Rerun last failed tests

Environment Variables:
  COVERAGE_THRESHOLD   Coverage threshold (default: 95)
  PARALLEL            Parallel workers (default: auto)
  TIMEOUT             Test timeout in seconds (default: 300)
  PYTEST_ARGS         Additional pytest arguments

EOF
    exit 0
}

# Clean test artifacts
clean_artifacts() {
    print_header "Cleaning Test Artifacts"
    rm -rf .pytest_cache
    rm -rf htmlcov
    rm -rf .coverage
    rm -rf coverage.json
    rm -rf coverage.xml
    rm -rf test-results
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    print_msg "$GREEN" "✓ Test artifacts cleaned"
}

# Run fast unit tests
run_fast() {
    print_header "Running Fast Unit Tests"
    python -m pytest tests/ \
        -m "not integration and not slow" \
        -n "$PARALLEL" \
        --timeout="$TIMEOUT" \
        --tb=short \
        -v \
        ${PYTEST_ARGS}
}

# Run full test suite
run_full() {
    print_header "Running Full Test Suite"
    python -m pytest tests/ \
        -n "$PARALLEL" \
        --timeout="$TIMEOUT" \
        --tb=short \
        -v \
        ${PYTEST_ARGS}
}

# Run integration tests only
run_integration() {
    print_header "Running Integration Tests"
    python -m pytest tests/integration/ \
        -m integration \
        -n "$PARALLEL" \
        --timeout="$TIMEOUT" \
        --tb=long \
        -v \
        ${PYTEST_ARGS}
}

# Run tests with coverage
run_coverage() {
    print_header "Running Tests with Coverage"
    python -m pytest tests/ \
        --cov=ai_dev_agent \
        --cov-config=.coveragerc \
        --cov-report=term-missing \
        --cov-report=html \
        --cov-report=json \
        --cov-report=xml \
        --cov-fail-under="$COVERAGE_THRESHOLD" \
        -n "$PARALLEL" \
        --timeout="$TIMEOUT" \
        -v \
        ${PYTEST_ARGS}

    print_msg "$GREEN" "\n✓ Coverage report generated in htmlcov/index.html"
}

# Run tests in watch mode
run_watch() {
    print_header "Running Tests in Watch Mode"
    print_msg "$YELLOW" "Watching for file changes... (Press Ctrl+C to exit)"

    if ! command -v pytest-watch &> /dev/null; then
        print_msg "$RED" "Error: pytest-watch not installed"
        print_msg "$YELLOW" "Install with: pip install pytest-watch"
        exit 1
    fi

    pytest-watch -- tests/ -m "not integration and not slow" -v
}

# Run benchmark tests
run_benchmark() {
    print_header "Running Performance Benchmarks"

    if [ -d "benchmarks" ]; then
        python -m pytest benchmarks/ \
            --benchmark-only \
            --benchmark-autosave \
            --benchmark-compare \
            -v \
            ${PYTEST_ARGS}
    else
        print_msg "$YELLOW" "No benchmarks directory found"
    fi
}

# Run CI test suite
run_ci() {
    print_header "Running CI Test Suite"

    # Step 1: Fast tests first
    print_msg "$BLUE" "Step 1/4: Fast unit tests"
    python -m pytest tests/ \
        -m "not integration and not slow" \
        -n "$PARALLEL" \
        --timeout="$TIMEOUT" \
        -v

    # Step 2: Integration tests
    print_msg "$BLUE" "Step 2/4: Integration tests"
    python -m pytest tests/integration/ \
        -m integration \
        --timeout="$TIMEOUT" \
        -v

    # Step 3: Coverage check
    print_msg "$BLUE" "Step 3/4: Coverage analysis"
    python -m pytest tests/ \
        --cov=ai_dev_agent \
        --cov-config=.coveragerc \
        --cov-report=term-missing \
        --cov-report=html \
        --cov-report=xml \
        --cov-fail-under="$COVERAGE_THRESHOLD" \
        -n "$PARALLEL" \
        --quiet

    # Step 4: Type checking (if mypy available)
    if command -v mypy &> /dev/null; then
        print_msg "$BLUE" "Step 4/4: Type checking"
        mypy ai_dev_agent --ignore-missing-imports || true
    else
        print_msg "$YELLOW" "Step 4/4: Skipped (mypy not installed)"
    fi

    print_msg "$GREEN" "\n✓ CI test suite completed successfully"
}

# Rerun failed tests
run_failed() {
    print_header "Rerunning Failed Tests"
    python -m pytest --lf -v ${PYTEST_ARGS}
}

# Parse command line arguments
PROFILE="fast"
SKIP_COVERAGE=false
VERBOSE=false
MARKERS=""
FAILED_ONLY=false
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        fast|full|integration|coverage|watch|benchmark|ci)
            PROFILE=$1
            shift
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --threshold)
            COVERAGE_THRESHOLD="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --no-cov)
            SKIP_COVERAGE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            PYTEST_ARGS="${PYTEST_ARGS} -vv"
            shift
            ;;
        --markers)
            MARKERS="$2"
            PYTEST_ARGS="${PYTEST_ARGS} -m $2"
            shift 2
            ;;
        --failed)
            FAILED_ONLY=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --help)
            show_usage
            ;;
        *)
            print_msg "$RED" "Unknown option: $1"
            show_usage
            ;;
    esac
done

# Show configuration
print_header "Test Configuration"
echo "Profile:           $PROFILE"
echo "Parallel Workers:  $PARALLEL"
echo "Coverage Threshold: $COVERAGE_THRESHOLD%"
echo "Timeout:           ${TIMEOUT}s"
echo "Project Root:      $PROJECT_ROOT"
echo ""

# Clean if requested
if [ "$CLEAN" = true ]; then
    clean_artifacts
fi

# Run tests based on profile
START_TIME=$(date +%s)

if [ "$FAILED_ONLY" = true ]; then
    run_failed
else
    case $PROFILE in
        fast)
            run_fast
            ;;
        full)
            run_full
            ;;
        integration)
            run_integration
            ;;
        coverage)
            run_coverage
            ;;
        watch)
            run_watch
            ;;
        benchmark)
            run_benchmark
            ;;
        ci)
            run_ci
            ;;
        *)
            print_msg "$RED" "Unknown profile: $PROFILE"
            show_usage
            ;;
    esac
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Show summary
print_header "Test Summary"
print_msg "$GREEN" "✓ Tests completed successfully"
print_msg "$BLUE" "Duration: ${DURATION}s"

# Show coverage report link if available
if [ -f "htmlcov/index.html" ]; then
    print_msg "$BLUE" "Coverage report: file://$PROJECT_ROOT/htmlcov/index.html"
fi

exit 0