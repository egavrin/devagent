.PHONY: help quality format lint type-check security test coverage clean install pre-commit

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install project dependencies
	pip install -e ".[dev]"
	pre-commit install

quality: format lint type-check security test ## Run all quality checks

format: ## Format code with black and isort
	@echo "🎨 Formatting code..."
	black ai_dev_agent tests
	isort ai_dev_agent tests
	@echo "✅ Code formatted!"

lint: ## Run linting with ruff
	@echo "🔍 Linting code..."
	ruff check ai_dev_agent tests --fix
	@echo "✅ Linting complete!"

type-check: ## Run type checking with mypy
	@echo "📝 Type checking..."
	mypy ai_dev_agent || true
	@echo "✅ Type checking complete!"

security: ## Run security checks with bandit
	@echo "🔒 Security scanning..."
	bandit -r ai_dev_agent -ll -x tests || true
	@echo "✅ Security scan complete!"

test: ## Run tests with pytest
	@echo "🧪 Running tests..."
	pytest
	@echo "✅ Tests complete!"

coverage: ## Run tests with coverage report
	@echo "📊 Running tests with coverage..."
	pytest --cov=ai_dev_agent --cov-report=html --cov-report=term
	@echo "✅ Coverage report generated in htmlcov/"

pre-commit: ## Run pre-commit hooks on all files
	@echo "🔗 Running pre-commit hooks..."
	pre-commit run --all-files
	@echo "✅ Pre-commit hooks complete!"

clean: ## Clean up temporary files and caches
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov/ .coverage coverage.json
	rm -rf build/ dist/ *.egg-info
	@echo "✅ Cleanup complete!"

# Quick checks for development
quick: format lint ## Quick format and lint check

# Full validation before commit
validate: quality ## Full validation suite

# Check what would be formatted without making changes
dry-run: ## Show what would be formatted
	@echo "📋 Checking format changes..."
	black --check --diff ai_dev_agent tests
	isort --check-only --diff ai_dev_agent tests

# Update all dependencies
update-deps: ## Update project dependencies
	pip install --upgrade pip setuptools wheel
	pip install -e ".[dev]" --upgrade

# Initialize pre-commit hooks
init-hooks: ## Initialize pre-commit hooks
	pre-commit install
	pre-commit install --hook-type commit-msg
	@echo "✅ Pre-commit hooks installed!"

# Run specific quality tool
black: ## Run only black formatter
	black ai_dev_agent tests

isort: ## Run only isort
	isort ai_dev_agent tests

ruff: ## Run only ruff linter
	ruff check ai_dev_agent tests

mypy: ## Run only mypy
	mypy ai_dev_agent

bandit: ## Run only bandit
	bandit -r ai_dev_agent -ll -x tests

# Development workflow helpers
fix: format lint ## Auto-fix formatting and linting issues

check: dry-run type-check ## Check code without modifying

ci: ## Run CI pipeline checks
	@echo "🚀 Running CI checks..."
	black --check ai_dev_agent tests
	isort --check-only ai_dev_agent tests
	ruff check ai_dev_agent tests
	mypy ai_dev_agent
	bandit -r ai_dev_agent -ll -x tests
	pytest --cov=ai_dev_agent --cov-fail-under=69
	@echo "✅ CI checks passed!"

# Show current code statistics
stats: ## Show code statistics
	@echo "📈 Code Statistics:"
	@echo "Lines of Python code:"
	@find ai_dev_agent -name "*.py" -exec wc -l {} + | tail -1
	@echo "\nNumber of Python files:"
	@find ai_dev_agent -name "*.py" | wc -l
	@echo "\nTest files:"
	@find tests -name "test_*.py" | wc -l
	@echo "\nTODO comments:"
	@grep -r "TODO" ai_dev_agent --include="*.py" | wc -l || echo "0"
