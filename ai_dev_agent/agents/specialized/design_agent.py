"""Design Agent for creating technical designs and analyzing references."""
from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..base import BaseAgent, AgentContext, AgentResult, AgentCapability


class DesignAgent(BaseAgent):
    """Agent specialized in creating technical designs and architecture."""

    def __init__(self):
        """Initialize Design Agent."""
        super().__init__(
            name="design_agent",
            description="Creates technical designs, analyzes references, and documents architecture",
            capabilities=[
                "technical_design",
                "reference_analysis",
                "architecture_design",
                "pattern_extraction"
            ],
            tools=["read", "write", "grep", "find", "symbols"],
            max_iterations=30
        )

        # Register capabilities
        self._register_capabilities()

    def _register_capabilities(self):
        """Register agent capabilities."""
        capabilities = [
            AgentCapability(
                name="technical_design",
                description="Create technical design documents",
                required_tools=["read", "write"],
                optional_tools=["grep", "find"]
            ),
            AgentCapability(
                name="reference_analysis",
                description="Analyze reference implementations",
                required_tools=["read", "grep"],
                optional_tools=["symbols"]
            ),
            AgentCapability(
                name="architecture_design",
                description="Design system architecture",
                required_tools=["write"],
                optional_tools=["read", "grep"]
            ),
            AgentCapability(
                name="pattern_extraction",
                description="Extract design patterns from code",
                required_tools=["read", "grep"],
                optional_tools=["symbols"]
            )
        ]

        for capability in capabilities:
            self.register_capability(capability)

    def analyze_requirements(
        self,
        requirements: str,
        context: AgentContext
    ) -> Dict[str, Any]:
        """
        Analyze requirements for a feature.

        Args:
            requirements: Requirements text
            context: Execution context

        Returns:
            Analysis results with extracted features
        """
        # Extract key features from requirements
        features = set()

        # Common requirement patterns
        patterns = {
            "user_registration": r"(user|account)\s+(registration|signup|create)",
            "authentication": r"(auth|login|jwt|token|password)",
            "crud_operations": r"(crud|create|read|update|delete|manage)",
            "api": r"(api|rest|endpoint|route)",
            "database": r"(database|db|storage|persist)",
            "security": r"(security|encrypt|hash|protect)",
            "validation": r"(validat|verify|check)",
            "testing": r"(test|spec|coverage)",
        }

        req_lower = requirements.lower()
        for feature, pattern in patterns.items():
            if re.search(pattern, req_lower, re.IGNORECASE):
                features.add(feature)

        return {
            "status": "analyzed",
            "features": list(features),
            "raw_requirements": requirements,
            "timestamp": datetime.now().isoformat()
        }

    def extract_patterns(
        self,
        file_path: str,
        context: AgentContext
    ) -> Dict[str, Any]:
        """
        Extract design patterns from a reference file.

        Args:
            file_path: Path to reference file
            context: Execution context

        Returns:
            Extracted patterns and architecture info
        """
        patterns = []
        architecture = {}

        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Detect common patterns
            pattern_checks = {
                "Controller Pattern": r"class\s+\w*Controller",
                "Service Layer": r"class\s+\w*Service",
                "Repository Pattern": r"class\s+\w*Repository",
                "Factory Pattern": r"class\s+\w*Factory",
                "Observer Pattern": r"(observer|subscriber|event|listener)",
                "Singleton": r"(singleton|instance\s*=\s*None)",
                "Dependency Injection": r"def\s+__init__.*\(.*,.*\)",
                "MVC": r"(model|view|controller)",
            }

            for pattern_name, regex in pattern_checks.items():
                if re.search(regex, content, re.IGNORECASE):
                    patterns.append(pattern_name)

            # Detect architecture style
            if "async def" in content:
                architecture["async"] = True
            if "class" in content:
                architecture["style"] = "object-oriented"
            elif "def " in content:
                architecture["style"] = "functional"

            # Count components
            classes = len(re.findall(r"^class\s+", content, re.MULTILINE))
            functions = len(re.findall(r"^def\s+", content, re.MULTILINE))

            architecture["components"] = {
                "classes": classes,
                "functions": functions
            }

        except Exception as e:
            return {
                "error": str(e),
                "patterns": [],
                "architecture": {}
            }

        return {
            "patterns": patterns,
            "architecture": architecture,
            "file": file_path
        }

    def create_design_document(
        self,
        design_data: Dict[str, Any],
        output_path: str,
        context: AgentContext
    ) -> Dict[str, Any]:
        """
        Create a design document in markdown format.

        Args:
            design_data: Design information
            output_path: Path to save the document
            context: Execution context

        Returns:
            Result of document creation
        """
        try:
            # Create document content
            doc_lines = [
                f"# Design: {design_data.get('feature', 'Feature')}",
                f"\n**Created**: {datetime.now().isoformat()}",
                f"\n**Author**: {self.name}",
                "\n## Requirements\n"
            ]

            # Add requirements
            for req in design_data.get("requirements", []):
                doc_lines.append(f"- {req}")

            # Add architecture section
            doc_lines.append("\n## Architecture\n")
            arch = design_data.get("architecture", {})
            if "components" in arch:
                doc_lines.append("### Components")
                for comp in arch["components"]:
                    doc_lines.append(f"- **{comp}**")

            if "flow" in arch:
                doc_lines.append(f"\n### Data Flow\n```\n{arch['flow']}\n```")

            # Add patterns section
            doc_lines.append("\n## Patterns\n")
            for pattern in design_data.get("patterns", []):
                doc_lines.append(f"- {pattern}")

            # Add references
            if "references" in design_data:
                doc_lines.append("\n## References\n")
                for ref in design_data["references"]:
                    doc_lines.append(f"- `{ref}`")

            # Add implementation notes
            if "notes" in design_data:
                doc_lines.append("\n## Implementation Notes\n")
                doc_lines.append(design_data["notes"])

            # Write document
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write('\n'.join(doc_lines))

            return {
                "success": True,
                "path": output_path,
                "lines": len(doc_lines)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def analyze_references(
        self,
        feature: str,
        reference_paths: List[str],
        context: AgentContext
    ) -> Dict[str, Any]:
        """
        Analyze reference implementations for a feature.

        Args:
            feature: Feature to analyze
            reference_paths: Paths to reference implementations
            context: Execution context

        Returns:
            Analysis results with patterns and recommendations
        """
        references = []
        all_patterns = set()
        recommendations = []

        for ref_path in reference_paths:
            if not os.path.exists(ref_path):
                continue

            ref_analysis = {
                "path": ref_path,
                "patterns": [],
                "relevant_files": []
            }

            # Search for relevant files
            if os.path.isdir(ref_path):
                for root, dirs, files in os.walk(ref_path):
                    for file in files:
                        if file.endswith(('.py', '.js', '.ts')):
                            file_path = os.path.join(root, file)
                            # Simple relevance check based on filename
                            if feature.lower() in file.lower():
                                ref_analysis["relevant_files"].append(file_path)

                                # Extract patterns from relevant files
                                patterns = self.extract_patterns(file_path, context)
                                ref_analysis["patterns"].extend(patterns.get("patterns", []))
                                all_patterns.update(patterns.get("patterns", []))

            references.append(ref_analysis)

        # Generate recommendations based on patterns found
        if "Controller Pattern" in all_patterns:
            recommendations.append("Use Controller pattern for handling requests")
        if "Service Layer" in all_patterns:
            recommendations.append("Implement Service layer for business logic")
        if "Repository Pattern" in all_patterns:
            recommendations.append("Use Repository pattern for data access")

        return {
            "references": references,
            "patterns": list(all_patterns),
            "recommendations": recommendations,
            "feature": feature
        }

    def assess_compatibility(
        self,
        proposed_changes: Dict[str, Any],
        context: AgentContext
    ) -> Dict[str, Any]:
        """
        Assess compatibility impact of proposed changes.

        Args:
            proposed_changes: Dictionary of proposed changes
            context: Execution context

        Returns:
            Compatibility assessment with risk level
        """
        impacts = []
        risk_score = 0

        # Check new modules
        new_modules = proposed_changes.get("new_modules", [])
        if new_modules:
            impacts.append(f"Adding {len(new_modules)} new modules")
            risk_score += len(new_modules) * 0.1

        # Check modified modules
        modified_modules = proposed_changes.get("modified_modules", [])
        if modified_modules:
            impacts.append(f"Modifying {len(modified_modules)} existing modules")
            risk_score += len(modified_modules) * 0.2

        # Check API changes
        api_changes = proposed_changes.get("api_changes", [])
        if api_changes:
            impacts.append(f"{len(api_changes)} API changes")
            risk_score += len(api_changes) * 0.3

        # Check new dependencies
        new_deps = proposed_changes.get("new_dependencies", [])
        if new_deps:
            impacts.append(f"Adding {len(new_deps)} new dependencies")
            risk_score += len(new_deps) * 0.15

        # Determine risk level
        if risk_score < 0.3:
            risk_level = "low"
        elif risk_score < 0.7:
            risk_level = "medium"
        else:
            risk_level = "high"

        recommendations = []
        if risk_level == "high":
            recommendations.append("Consider breaking changes into smaller increments")
            recommendations.append("Add comprehensive tests before implementation")
        elif risk_level == "medium":
            recommendations.append("Ensure backward compatibility tests are in place")

        return {
            "risk_level": risk_level,
            "risk_score": min(risk_score, 1.0),
            "impacts": impacts,
            "recommendations": recommendations
        }

    def validate_design(self, design: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a design against best practices.

        Args:
            design: Design to validate

        Returns:
            Validation results with score
        """
        score = 0.0
        suggestions = []
        checks_passed = []

        # Check for architecture layers
        if "architecture" in design:
            arch = design["architecture"]
            if "layers" in arch:
                layers = arch["layers"]
                if all(l in layers for l in ["presentation", "business", "data"]):
                    score += 0.2
                    checks_passed.append("Proper layer separation")
                else:
                    suggestions.append("Consider using standard architecture layers")

            if "components" in arch:
                score += 0.1
                checks_passed.append("Components defined")

        # Check for design patterns
        if "patterns" in design:
            patterns = design["patterns"]
            good_patterns = ["MVC", "Repository", "Service Layer", "SOLID"]
            pattern_score = sum(0.1 for p in patterns if p in good_patterns)
            score += min(pattern_score, 0.3)
            if pattern_score > 0:
                checks_passed.append("Using recognized design patterns")

        # Check for principles
        if "principles" in design:
            principles = design["principles"]
            good_principles = ["SOLID", "DRY", "KISS", "YAGNI"]
            principle_score = sum(0.1 for p in principles if p in good_principles)
            score += min(principle_score, 0.2)
            if principle_score > 0:
                checks_passed.append("Following design principles")

        # Add base score for having a design
        score += 0.2

        # Ensure score is between 0 and 1
        score = min(score, 1.0)

        if score < 0.7:
            suggestions.append("Consider adding more architectural details")
            suggestions.append("Document design patterns being used")

        return {
            "valid": score >= 0.5,
            "score": score,
            "checks_passed": checks_passed,
            "suggestions": suggestions
        }

    def execute(self, prompt: str, context: AgentContext) -> AgentResult:
        """
        Execute design agent task using ReAct workflow.

        Args:
            prompt: Design task description
            context: Execution context

        Returns:
            AgentResult with design artifacts
        """
        # Import executor bridge
        from .executor_bridge import execute_agent_with_react

        # Execute using ReAct workflow with LLM and real tools
        return execute_agent_with_react(
            agent=self,
            prompt=prompt,
            context=context
        )