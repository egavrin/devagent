"""Enhanced agent registry with dynamic discovery and BaseAgent support."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
from typing import Any

from .base import BaseAgent
from .registry import AgentRegistry, AgentSpec


class EnhancedAgentRegistry:
    """Enhanced registry for managing agents and specs with dynamic discovery."""

    _instance: EnhancedAgentRegistry | None = None

    def __init__(self):
        """Initialize enhanced registry."""
        self._agents: dict[str, BaseAgent] = {}
        self._specs: dict[str, AgentSpec] = {}
        self._agent_classes: dict[str, type[BaseAgent]] = {}

        # Register base agent class
        self._agent_classes["BaseAgent"] = BaseAgent

        # Import default agents from original registry
        self._import_default_agents()

    @classmethod
    def get_instance(cls) -> EnhancedAgentRegistry:
        """Get singleton instance of registry."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _import_default_agents(self) -> None:
        """Import default agents from original registry."""
        # Import manager and reviewer specs
        try:
            manager_spec = AgentRegistry.get("manager")
            reviewer_spec = AgentRegistry.get("reviewer")
            self.register_spec(manager_spec)
            self.register_spec(reviewer_spec)
        except KeyError:
            pass  # Default agents not yet registered

    def register_agent(self, agent: BaseAgent) -> None:
        """Register a BaseAgent instance."""
        self._agents[agent.name] = agent

    def register_spec(self, spec: AgentSpec) -> None:
        """Register an agent spec."""
        self._specs[spec.name] = spec

    def register_agent_class(self, name: str, agent_class: type[BaseAgent]) -> None:
        """Register an agent class for dynamic instantiation."""
        self._agent_classes[name] = agent_class

    def has_agent(self, name: str) -> bool:
        """Check if an agent instance is registered."""
        return name in self._agents

    def has_spec(self, name: str) -> bool:
        """Check if an agent spec is registered."""
        return name in self._specs

    def get_agent(self, name: str) -> BaseAgent:
        """Get a registered agent instance."""
        if name not in self._agents:
            raise KeyError(f"Unknown agent: {name}")
        return self._agents[name]

    def get_spec(self, name: str) -> AgentSpec:
        """Get a registered agent spec."""
        if name not in self._specs:
            raise KeyError(f"Unknown agent spec: {name}")
        return self._specs[name]

    def create_agent(self, name: str, **kwargs) -> BaseAgent:
        """
        Create an agent instance from a spec or class.

        Args:
            name: Agent name/spec name
            **kwargs: Additional parameters to override

        Returns:
            New BaseAgent instance
        """
        # Try to create from spec first
        if self.has_spec(name):
            spec = self.get_spec(name)
            return self._create_agent_from_spec(spec, **kwargs)

        # Try to create from registered class
        if name in self._agent_classes:
            agent_class = self._agent_classes[name]
            return agent_class(**kwargs)

        # Try to create from already registered agent (clone)
        if self.has_agent(name):
            template = self.get_agent(name)
            return self._clone_agent(template, **kwargs)

        raise KeyError(f"Cannot create agent: {name} not found")

    def _create_agent_from_spec(self, spec: AgentSpec, **kwargs) -> BaseAgent:
        """Create a BaseAgent from an AgentSpec."""
        # Merge spec attributes with kwargs
        agent_kwargs = {
            "name": spec.name,
            "description": spec.description or "",
            "tools": spec.tools,
            "max_iterations": spec.max_iterations,
            "metadata": spec.metadata.copy(),
        }

        # Add system prompt as metadata if present
        if spec.system_prompt_suffix:
            agent_kwargs["metadata"]["system_prompt_suffix"] = spec.system_prompt_suffix

        # Override with any provided kwargs
        agent_kwargs.update(kwargs)

        return BaseAgent(**agent_kwargs)

    def _clone_agent(self, template: BaseAgent, **kwargs) -> BaseAgent:
        """Clone an existing agent with optional overrides."""
        agent_kwargs = {
            "name": template.name,
            "description": template.description,
            "capabilities": template.capabilities.copy(),
            "tools": template.tools.copy(),
            "max_iterations": template.max_iterations,
            "permissions": template.permissions.copy(),
            "metadata": template.metadata.copy(),
        }
        agent_kwargs.update(kwargs)
        return BaseAgent(**agent_kwargs)

    def find_agents_by_capability(self, capability: str) -> list[str]:
        """Find all agents with a specific capability."""
        matching = []
        for name, agent in self._agents.items():
            if capability in agent.capabilities:
                matching.append(name)
        return matching

    def find_agents_with_tools(self, required_tools: list[str]) -> list[str]:
        """Find agents that have all required tools."""
        matching = []
        for name, agent in self._agents.items():
            if all(tool in agent.tools for tool in required_tools):
                matching.append(name)
        return matching

    def list_all_agents(self) -> list[str]:
        """List all registered agent names."""
        return list(self._agents.keys())

    def list_all_specs(self) -> list[str]:
        """List all registered spec names."""
        return list(self._specs.keys())

    def save_to_file(self, filepath: str | Path) -> None:
        """Save registry specs to a JSON file."""
        data = {"specs": {}}

        for name, spec in self._specs.items():
            data["specs"][name] = {
                "name": spec.name,
                "tools": spec.tools,
                "max_iterations": spec.max_iterations,
                "description": spec.description,
                "system_prompt_suffix": spec.system_prompt_suffix,
                "metadata": spec.metadata,
            }

        filepath = Path(filepath)
        with filepath.open("w") as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filepath: str | Path) -> None:
        """Load registry specs from a JSON file."""
        filepath = Path(filepath)
        with filepath.open() as f:
            data = json.load(f)

        for _name, spec_data in data.get("specs", {}).items():
            spec = AgentSpec(
                name=spec_data["name"],
                tools=spec_data["tools"],
                max_iterations=spec_data["max_iterations"],
                description=spec_data.get("description"),
                system_prompt_suffix=spec_data.get("system_prompt_suffix"),
                metadata=spec_data.get("metadata", {}),
            )
            self.register_spec(spec)

    def clear(self) -> None:
        """Clear all registered agents and specs."""
        self._agents.clear()
        self._specs.clear()
        # Keep agent classes


class AgentLoader:
    """Loads agents from various sources."""

    def load_agent_class(self, module_name: str, class_name: str) -> type[BaseAgent]:
        """
        Load an agent class from a module.

        Args:
            module_name: Python module path
            class_name: Name of the agent class

        Returns:
            Agent class type
        """
        module = importlib.import_module(module_name)
        agent_class = getattr(module, class_name)

        if not issubclass(agent_class, BaseAgent):
            raise TypeError(f"{class_name} is not a BaseAgent subclass")

        return agent_class

    def load_from_config(self, config_path: str) -> BaseAgent:
        """
        Load an agent from a configuration file.

        Args:
            config_path: Path to JSON config file

        Returns:
            Configured BaseAgent instance
        """
        config_path = Path(config_path)
        with config_path.open() as f:
            config = json.load(f)

        # Get agent type or default to BaseAgent
        agent_type = config.pop("type", "BaseAgent")

        # Create agent based on type
        if agent_type == "BaseAgent":
            return BaseAgent(**config)
        else:
            # Try to load custom agent type
            if "." in agent_type:
                module_name, class_name = agent_type.rsplit(".", 1)
                agent_class = self.load_agent_class(module_name, class_name)
                return agent_class(**config)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")

    def load_from_file(self, filepath: str) -> type[BaseAgent]:
        """
        Load an agent class from a Python file.

        Args:
            filepath: Path to Python file containing agent class

        Returns:
            First BaseAgent subclass found in the file
        """
        spec = importlib.util.spec_from_file_location("custom_agent", filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find BaseAgent subclasses
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseAgent) and obj is not BaseAgent:
                return obj

        raise ValueError(f"No BaseAgent subclass found in {filepath}")


class AgentDiscovery:
    """Discovers and auto-registers agents."""

    def discover_in_directory(self, directory: str) -> list[str]:
        """
        Discover agent files in a directory.

        Args:
            directory: Directory path to search

        Returns:
            List of discovered agent file paths
        """
        agent_files = []
        path = Path(directory)

        for py_file in path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            # Check if file likely contains an agent
            with py_file.open() as f:
                content = f.read()
                if "BaseAgent" in content or "AgentSpec" in content:
                    agent_files.append(str(py_file))

        return agent_files

    def discover_builtin_agents(self) -> dict[str, AgentSpec]:
        """
        Discover built-in agents from the registry.

        Returns:
            Dictionary of built-in agent specs
        """
        builtin = {}

        # Get from original registry
        for name in AgentRegistry.list_agents():
            try:
                spec = AgentRegistry.get(name)
                builtin[name] = spec
            except KeyError:
                pass

        return builtin

    def is_valid_agent_class(self, obj: Any) -> bool:
        """Check if an object is a valid agent class."""
        return inspect.isclass(obj) and issubclass(obj, BaseAgent) and obj is not BaseAgent

    def register_agent_class(
        self, registry: EnhancedAgentRegistry, agent_class: type[BaseAgent]
    ) -> None:
        """
        Register an agent class with the registry.

        Args:
            registry: Registry to register with
            agent_class: Agent class to register
        """
        # Create a test instance to get the name
        try:
            test_instance = agent_class()
            name = test_instance.name
            registry.register_agent_class(name, agent_class)

            # Also register the instance for immediate use
            registry.register_agent(test_instance)
        except Exception:
            # If can't instantiate with no args, just register the class
            class_name = agent_class.__name__
            registry.register_agent_class(class_name, agent_class)

    def auto_discover_and_register(
        self, registry: EnhancedAgentRegistry, search_paths: list[str] | None = None
    ) -> int:
        """
        Auto-discover and register agents from search paths.

        Args:
            registry: Registry to register with
            search_paths: List of paths to search (defaults to built-in paths)

        Returns:
            Number of agents registered
        """
        count = 0

        if search_paths is None:
            # Default search paths
            search_paths = [
                str(Path(__file__).parent / "builtin"),
                str(Path("~/.devagent/agents").expanduser()),
            ]

        loader = AgentLoader()

        for path in search_paths:
            path_obj = Path(path)
            if not path_obj.exists():
                continue

            if path_obj.is_dir():
                # Discover in directory
                agent_files = self.discover_in_directory(path)

                for filepath in agent_files:
                    try:
                        agent_class = loader.load_from_file(filepath)
                        self.register_agent_class(registry, agent_class)
                        count += 1
                    except Exception:
                        pass  # Skip files that don't load properly

            elif path.endswith(".json"):
                # Load from config
                try:
                    agent = loader.load_from_config(path)
                    registry.register_agent(agent)
                    count += 1
                except Exception:
                    pass

        return count
