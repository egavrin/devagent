"""Universal symbol interface for tree-sitter and LSP integration.

This module provides a unified abstraction for symbol extraction and analysis,
supporting both tree-sitter (offline, fast) and LSP (real-time, accurate) backends.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Protocol, Any
from abc import ABC, abstractmethod

__all__ = [
    "Symbol",
    "SymbolKind",
    "SymbolSource",
    "SymbolGraph",
    "SymbolProvider",
    "TreeSitterSymbolProvider",
    "LSPSymbolProvider",
    "HybridSymbolProvider",
]


class SymbolKind(Enum):
    """Universal symbol types across languages."""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    VARIABLE = "variable"
    CONSTANT = "constant"
    INTERFACE = "interface"
    ENUM = "enum"
    STRUCT = "struct"
    MODULE = "module"
    NAMESPACE = "namespace"
    PROPERTY = "property"
    FIELD = "field"
    PARAMETER = "parameter"
    TYPE_PARAMETER = "type_parameter"


class SymbolSource(Enum):
    """Source of symbol information."""
    TREE_SITTER = "tree_sitter"
    LSP = "lsp"
    HYBRID = "hybrid"
    FALLBACK = "fallback"


@dataclass
class Symbol:
    """Universal symbol representation."""
    name: str
    kind: SymbolKind
    file: Path
    line: int
    column: int
    source: SymbolSource
    is_definition: bool = True
    type_info: Optional[str] = None  # From LSP
    documentation: Optional[str] = None  # From LSP
    references: Set[Path] = field(default_factory=set)

    def __hash__(self) -> int:
        """Make symbols hashable for use in sets."""
        return hash((self.name, str(self.file), self.line, self.column))

    def __eq__(self, other: object) -> bool:
        """Compare symbols by location."""
        if not isinstance(other, Symbol):
            return False
        return (
            self.name == other.name and
            self.file == other.file and
            self.line == other.line and
            self.column == other.column
        )


@dataclass
class SymbolGraph:
    """Graph of symbol definitions and references."""
    symbols: List[Symbol]
    definitions: Dict[str, Set[Path]]  # symbol_name -> defining files
    references: Dict[str, List[Path]]  # symbol_name -> referencing files

    @classmethod
    def from_symbols(cls, symbols: List[Symbol]) -> "SymbolGraph":
        """Build graph from symbol list."""
        definitions: Dict[str, Set[Path]] = {}
        references: Dict[str, List[Path]] = {}

        for symbol in symbols:
            if symbol.is_definition:
                if symbol.name not in definitions:
                    definitions[symbol.name] = set()
                definitions[symbol.name].add(symbol.file)
            else:
                if symbol.name not in references:
                    references[symbol.name] = []
                references[symbol.name].append(symbol.file)

        return cls(symbols=symbols, definitions=definitions, references=references)


class SymbolProvider(ABC):
    """Abstract interface for symbol providers."""

    @abstractmethod
    async def get_symbols(self, file: Path) -> List[Symbol]:
        """Get all symbols in a file."""
        raise NotImplementedError

    @abstractmethod
    async def get_workspace_symbols(self, query: str = "") -> List[Symbol]:
        """Search symbols across workspace."""
        raise NotImplementedError

    @abstractmethod
    async def get_references(self, symbol_name: str, file: Path) -> List[Symbol]:
        """Find all references to a symbol."""
        raise NotImplementedError

    @property
    @abstractmethod
    def source_type(self) -> SymbolSource:
        """Get the source type of this provider."""
        raise NotImplementedError


class TreeSitterSymbolProvider(SymbolProvider):
    """Symbol extraction via tree-sitter parsing."""

    def __init__(self, root: Path):
        self.root = root
        self._init_tree_sitter()

    def _init_tree_sitter(self) -> None:
        """Initialize tree-sitter components."""
        from ai_dev_agent.core.tree_sitter import TreeSitterManager, ensure_parser
        from ai_dev_agent.core.cache import SymbolCache

        self.manager = TreeSitterManager()
        self.cache = SymbolCache(self.root)

    async def get_symbols(self, file: Path) -> List[Symbol]:
        """Extract symbols using tree-sitter queries."""
        # Check cache first
        cached = self.cache.get_symbols(file)
        if cached:
            return cached

        # Parse with tree-sitter
        from ai_dev_agent.core.tree_sitter import ensure_parser, slice_bytes

        parser_handle = ensure_parser(file)
        if not parser_handle:
            return []

        content = file.read_bytes()
        tree = parser_handle.parser.parse(content)

        # Extract using language-specific queries
        symbols = await self._extract_symbols_from_tree(
            tree, parser_handle.language, file, content
        )

        # Cache results
        self.cache.set_symbols(file, symbols)

        return symbols

    async def _extract_symbols_from_tree(
        self, tree: Any, language: str, file: Path, content: bytes
    ) -> List[Symbol]:
        """Extract symbols from parsed tree."""
        symbols = []

        # Load language-specific query
        query = self._load_query_for_language(language)
        if not query:
            return []

        # Execute query
        from ai_dev_agent.core.tree_sitter import language_object
        lang_obj = language_object(language)
        if not lang_obj:
            return []

        captures = query.captures(tree.root_node)

        for node, tag in captures:
            if tag.startswith("name.definition."):
                kind_str = tag.replace("name.definition.", "")
                kind = self._map_tree_sitter_kind(kind_str)
                symbols.append(Symbol(
                    name=node.text.decode("utf-8"),
                    kind=kind,
                    file=file,
                    line=node.start_point[0],
                    column=node.start_point[1],
                    source=SymbolSource.TREE_SITTER,
                    is_definition=True
                ))
            elif tag.startswith("name.reference."):
                symbols.append(Symbol(
                    name=node.text.decode("utf-8"),
                    kind=SymbolKind.VARIABLE,  # Default for references
                    file=file,
                    line=node.start_point[0],
                    column=node.start_point[1],
                    source=SymbolSource.TREE_SITTER,
                    is_definition=False
                ))

        return symbols

    def _load_query_for_language(self, language: str) -> Optional[Any]:
        """Load tree-sitter query for language."""
        from ai_dev_agent.core.tree_sitter.queries import get_scm_file_path
        from ai_dev_agent.core.tree_sitter import language_object

        query_path = get_scm_file_path(language)
        if not query_path or not query_path.exists():
            return None

        query_text = query_path.read_text()
        lang_obj = language_object(language)

        if lang_obj:
            return lang_obj.query(query_text)
        return None

    def _map_tree_sitter_kind(self, kind_str: str) -> SymbolKind:
        """Map tree-sitter kinds to universal SymbolKind."""
        mapping = {
            "function": SymbolKind.FUNCTION,
            "class": SymbolKind.CLASS,
            "method": SymbolKind.METHOD,
            "variable": SymbolKind.VARIABLE,
            "constant": SymbolKind.CONSTANT,
            "interface": SymbolKind.INTERFACE,
            "enum": SymbolKind.ENUM,
            "struct": SymbolKind.STRUCT,
            "module": SymbolKind.MODULE,
            "namespace": SymbolKind.NAMESPACE,
            "property": SymbolKind.PROPERTY,
            "field": SymbolKind.FIELD,
        }
        return mapping.get(kind_str, SymbolKind.VARIABLE)

    async def get_workspace_symbols(self, query: str = "") -> List[Symbol]:
        """Search symbols across workspace (slow with tree-sitter)."""
        symbols = []

        # Find all relevant files
        for ext in [".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".c"]:
            for file in self.root.rglob(f"*{ext}"):
                if any(part.startswith(".") for part in file.parts):
                    continue  # Skip hidden directories

                file_symbols = await self.get_symbols(file)
                if query:
                    # Filter by query
                    file_symbols = [
                        s for s in file_symbols
                        if query.lower() in s.name.lower()
                    ]
                symbols.extend(file_symbols)

        return symbols

    async def get_references(self, symbol_name: str, file: Path) -> List[Symbol]:
        """Find references to a symbol."""
        references = []

        # Search in current file first
        symbols = await self.get_symbols(file)
        references.extend([
            s for s in symbols
            if s.name == symbol_name and not s.is_definition
        ])

        # Could expand to search other files if needed
        return references

    @property
    def source_type(self) -> SymbolSource:
        return SymbolSource.TREE_SITTER


class LSPSymbolProvider(SymbolProvider):
    """Symbol extraction via Language Server Protocol."""

    def __init__(self, root: Path):
        self.root = root
        self.registry = None  # Will be initialized lazily

    async def _ensure_registry(self) -> Any:
        """Lazily initialize LSP registry."""
        if self.registry is None:
            from ai_dev_agent.lsp.servers import ServerRegistry
            self.registry = ServerRegistry(self.root)
        return self.registry

    async def get_symbols(self, file: Path) -> List[Symbol]:
        """Get symbols via LSP documentSymbol request."""
        registry = await self._ensure_registry()
        client = await registry.get_or_create_client(file)

        if not client:
            return []

        # Open file in LSP
        await client.notify.open({"path": str(file)})

        # Request document symbols
        lsp_symbols = await client.connection.sendRequest(
            "textDocument/documentSymbol",
            {"textDocument": {"uri": file.as_uri()}}
        )

        return [self._convert_lsp_symbol(s, file) for s in lsp_symbols]

    def _convert_lsp_symbol(self, lsp_symbol: Dict[str, Any], file: Path) -> Symbol:
        """Convert LSP symbol to universal Symbol."""
        return Symbol(
            name=lsp_symbol["name"],
            kind=self._map_lsp_kind(lsp_symbol.get("kind", 0)),
            file=file,
            line=lsp_symbol["range"]["start"]["line"],
            column=lsp_symbol["range"]["start"]["character"],
            source=SymbolSource.LSP,
            is_definition=True,
            type_info=lsp_symbol.get("detail"),
            documentation=lsp_symbol.get("documentation")
        )

    def _map_lsp_kind(self, lsp_kind: int) -> SymbolKind:
        """Map LSP SymbolKind numbers to universal SymbolKind."""
        mapping = {
            5: SymbolKind.CLASS,
            6: SymbolKind.METHOD,
            7: SymbolKind.PROPERTY,
            8: SymbolKind.FIELD,
            9: SymbolKind.FUNCTION,  # Constructor
            10: SymbolKind.ENUM,
            11: SymbolKind.INTERFACE,
            12: SymbolKind.FUNCTION,
            13: SymbolKind.VARIABLE,
            14: SymbolKind.CONSTANT,
            23: SymbolKind.STRUCT,
        }
        return mapping.get(lsp_kind, SymbolKind.VARIABLE)

    async def get_workspace_symbols(self, query: str = "") -> List[Symbol]:
        """Fast workspace-wide symbol search via LSP."""
        registry = await self._ensure_registry()
        all_symbols = []

        for client in registry.active_clients():
            try:
                lsp_symbols = await client.connection.sendRequest(
                    "workspace/symbol",
                    {"query": query}
                )

                for s in lsp_symbols:
                    file = Path(s["location"]["uri"].replace("file://", ""))
                    all_symbols.append(Symbol(
                        name=s["name"],
                        kind=self._map_lsp_kind(s.get("kind", 0)),
                        file=file,
                        line=s["location"]["range"]["start"]["line"],
                        column=s["location"]["range"]["start"]["character"],
                        source=SymbolSource.LSP,
                        is_definition=True
                    ))
            except Exception:
                continue

        return all_symbols

    async def get_references(self, symbol_name: str, file: Path) -> List[Symbol]:
        """Find references via LSP."""
        registry = await self._ensure_registry()
        client = await registry.get_or_create_client(file)

        if not client:
            return []

        # Find symbol position first
        symbols = await self.get_symbols(file)
        target = next((s for s in symbols if s.name == symbol_name), None)

        if not target:
            return []

        # Request references
        references = await client.connection.sendRequest(
            "textDocument/references",
            {
                "textDocument": {"uri": file.as_uri()},
                "position": {"line": target.line, "character": target.column},
                "context": {"includeDeclaration": False}
            }
        )

        return [
            Symbol(
                name=symbol_name,
                kind=SymbolKind.VARIABLE,
                file=Path(ref["uri"].replace("file://", "")),
                line=ref["range"]["start"]["line"],
                column=ref["range"]["start"]["character"],
                source=SymbolSource.LSP,
                is_definition=False
            )
            for ref in references
        ]

    @property
    def source_type(self) -> SymbolSource:
        return SymbolSource.LSP


class HybridSymbolProvider(SymbolProvider):
    """Combines tree-sitter and LSP for best results."""

    def __init__(self, root: Path):
        self.root = root
        self.ts_provider = TreeSitterSymbolProvider(root)
        self.lsp_provider = LSPSymbolProvider(root)

    async def get_symbols(self, file: Path) -> List[Symbol]:
        """Get symbols using best available method."""
        # Try LSP first (has type info)
        try:
            lsp_symbols = await self.lsp_provider.get_symbols(file)
            if lsp_symbols:
                # Enhance with tree-sitter references
                ts_symbols = await self.ts_provider.get_symbols(file)
                return self._merge_symbols(lsp_symbols, ts_symbols)
        except Exception:
            pass

        # Fallback to tree-sitter
        return await self.ts_provider.get_symbols(file)

    def _merge_symbols(
        self, lsp_symbols: List[Symbol], ts_symbols: List[Symbol]
    ) -> List[Symbol]:
        """Merge LSP definitions with tree-sitter references."""
        # Index by name and location
        symbol_map = {(s.name, s.line): s for s in lsp_symbols}

        # Add references from tree-sitter
        for ts_sym in ts_symbols:
            if not ts_sym.is_definition:
                # Find matching definition
                for lsp_sym in lsp_symbols:
                    if lsp_sym.name == ts_sym.name:
                        lsp_sym.references.add(ts_sym.file)

        return list(symbol_map.values())

    async def get_workspace_symbols(self, query: str = "") -> List[Symbol]:
        """Use LSP for fast workspace search, fallback to tree-sitter."""
        try:
            lsp_symbols = await self.lsp_provider.get_workspace_symbols(query)
            if lsp_symbols:
                return lsp_symbols
        except Exception:
            pass

        return await self.ts_provider.get_workspace_symbols(query)

    async def get_references(self, symbol_name: str, file: Path) -> List[Symbol]:
        """Find references using best available method."""
        try:
            lsp_refs = await self.lsp_provider.get_references(symbol_name, file)
            if lsp_refs:
                return lsp_refs
        except Exception:
            pass

        return await self.ts_provider.get_references(symbol_name, file)

    @property
    def source_type(self) -> SymbolSource:
        return SymbolSource.HYBRID

    @classmethod
    def create(cls, root: Path) -> "HybridSymbolProvider":
        """Factory method to create hybrid provider."""
        return cls(root)