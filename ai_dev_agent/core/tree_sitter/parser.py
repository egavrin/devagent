"""Tree-sitter based parsing for symbol extraction."""

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import logging

try:
    import tree_sitter_languages as tsl
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

logger = logging.getLogger(__name__)


class TreeSitterParser:
    """Parser using tree-sitter for accurate symbol extraction."""

    # Language mapping for tree-sitter
    LANGUAGE_MAP = {
        'python': 'python',
        'javascript': 'javascript',
        'typescript': 'typescript',
        'tsx': 'tsx',
        'c': 'c',
        'cpp': 'cpp',
        'c_sharp': 'c_sharp',
        'java': 'java',
        'go': 'go',
        'rust': 'rust',
        'ruby': 'ruby',
        'php': 'php',
        'bash': 'bash',
        'lua': 'lua',
        'scala': 'scala',
        'kotlin': 'kotlin',
        'swift': 'swift',
        'r': 'r',
        'julia': 'julia',
        'haskell': 'haskell',
        'ocaml': 'ocaml',
        'perl': 'perl',
        'elixir': 'elixir',
        'elm': 'elm',
        'clojure': 'clojure'
    }

    def __init__(self):
        """Initialize tree-sitter parser."""
        self.parsers = {}
        if not TREE_SITTER_AVAILABLE:
            logger.warning("tree-sitter-languages not available, falling back to regex parsing")

    def get_parser(self, language: str):
        """Get or create a parser for the given language."""
        if not TREE_SITTER_AVAILABLE:
            return None

        ts_lang = self.LANGUAGE_MAP.get(language)
        if not ts_lang:
            return None

        if ts_lang not in self.parsers:
            try:
                self.parsers[ts_lang] = tsl.get_parser(ts_lang)
            except Exception as e:
                logger.debug(f"Could not get tree-sitter parser for {ts_lang}: {e}")
                return None

        return self.parsers[ts_lang]

    def extract_symbols(self, file_path: Path, language: str) -> Dict[str, List[Any]]:
        """Extract symbols from a file using tree-sitter."""
        if not TREE_SITTER_AVAILABLE:
            return {'symbols': [], 'imports': [], 'references': []}

        parser = self.get_parser(language)
        if not parser:
            return {'symbols': [], 'imports': [], 'references': []}

        try:
            with open(file_path, 'rb') as f:
                content = f.read()

            tree = parser.parse(content)

            symbols = []
            imports = []
            references = []

            # Extract based on language
            if language == 'python':
                symbols, imports, references = self._extract_python_symbols(tree, content)
            elif language in ['javascript', 'typescript']:
                symbols, imports, references = self._extract_javascript_symbols(tree, content)
            elif language in ['c', 'cpp']:
                symbols, imports, references = self._extract_cpp_symbols(tree, content)
            elif language == 'java':
                symbols, imports, references = self._extract_java_symbols(tree, content)
            elif language == 'go':
                symbols, imports, references = self._extract_go_symbols(tree, content)
            elif language == 'rust':
                symbols, imports, references = self._extract_rust_symbols(tree, content)
            else:
                # Generic extraction for other languages
                symbols = self._extract_generic_symbols(tree, content)

            return {
                'symbols': symbols,
                'imports': imports,
                'references': references
            }

        except Exception as e:
            logger.debug(f"Tree-sitter parsing failed for {file_path}: {e}")
            return {'symbols': [], 'imports': [], 'references': []}

    def _extract_python_symbols(self, tree, content: bytes) -> Tuple[List[str], List[str], List[str]]:
        """Extract Python symbols using tree-sitter."""
        symbols = []
        imports = []
        references = []

        query_text = """
        (class_definition name: (identifier) @class)
        (function_definition name: (identifier) @function)
        (decorated_definition
            definition: (function_definition name: (identifier) @decorated_function))
        (decorated_definition
            definition: (class_definition name: (identifier) @decorated_class))
        (import_statement) @import
        (import_from_statement) @import_from
        (call function: (identifier) @function_call)
        (call function: (attribute attribute: (identifier) @method_call))
        """

        try:
            language = tsl.get_language('python')
            query = language.query(query_text)
            captures = query.captures(tree.root_node)

            for node, capture_name in captures:
                text = content[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')

                if capture_name in ['class', 'decorated_class']:
                    symbols.append(text)
                elif capture_name in ['function', 'decorated_function']:
                    symbols.append(text)
                elif capture_name in ['import', 'import_from']:
                    imports.append(text)
                elif capture_name in ['function_call', 'method_call']:
                    references.append(text)

        except Exception as e:
            logger.debug(f"Python symbol extraction failed: {e}")

        return symbols, imports, references

    def _extract_javascript_symbols(self, tree, content: bytes) -> Tuple[List[str], List[str], List[str]]:
        """Extract JavaScript/TypeScript symbols using tree-sitter."""
        symbols = []
        imports = []
        references = []

        query_text = """
        (class_declaration name: (identifier) @class)
        (function_declaration name: (identifier) @function)
        (method_definition name: (property_identifier) @method)
        (variable_declarator name: (identifier) @variable)
        (interface_declaration name: (type_identifier) @interface)
        (type_alias_declaration name: (type_identifier) @type_alias)
        (import_statement) @import
        (call_expression function: (identifier) @function_call)
        """

        try:
            lang_name = 'javascript' if tree.root_node.type == 'program' else 'typescript'
            language = tsl.get_language(lang_name)
            query = language.query(query_text)
            captures = query.captures(tree.root_node)

            for node, capture_name in captures:
                text = content[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')

                if capture_name in ['class', 'interface', 'type_alias']:
                    symbols.append(text)
                elif capture_name in ['function', 'method']:
                    symbols.append(text)
                elif capture_name == 'variable':
                    # Only include if it looks like a significant variable
                    if text[0].isupper() or len(text) > 3:
                        symbols.append(text)
                elif capture_name == 'import':
                    imports.append(text)
                elif capture_name == 'function_call':
                    references.append(text)

        except Exception as e:
            logger.debug(f"JavaScript/TypeScript symbol extraction failed: {e}")

        return symbols, imports, references

    def _extract_cpp_symbols(self, tree, content: bytes) -> Tuple[List[str], List[str], List[str]]:
        """Extract C/C++ symbols using tree-sitter."""
        symbols = []
        imports = []
        references = []

        query_text = """
        (class_specifier name: (type_identifier) @class)
        (struct_specifier name: (type_identifier) @struct)
        (function_definition declarator: (function_declarator declarator: (identifier) @function))
        (declaration declarator: (function_declarator declarator: (identifier) @function_decl))
        (namespace_definition name: (identifier) @namespace)
        (preproc_include path: (_) @include)
        (call_expression function: (identifier) @function_call)
        """

        try:
            language = tsl.get_language('cpp')
            query = language.query(query_text)
            captures = query.captures(tree.root_node)

            for node, capture_name in captures:
                text = content[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')

                if capture_name in ['class', 'struct', 'namespace']:
                    symbols.append(text)
                elif capture_name in ['function', 'function_decl']:
                    symbols.append(text)
                elif capture_name == 'include':
                    # Clean up include path
                    text = text.strip('"<>')
                    imports.append(text)
                elif capture_name == 'function_call':
                    references.append(text)

        except Exception as e:
            logger.debug(f"C/C++ symbol extraction failed: {e}")

        return symbols, imports, references

    def _extract_java_symbols(self, tree, content: bytes) -> Tuple[List[str], List[str], List[str]]:
        """Extract Java symbols using tree-sitter."""
        symbols = []
        imports = []
        references = []

        query_text = """
        (class_declaration name: (identifier) @class)
        (interface_declaration name: (identifier) @interface)
        (enum_declaration name: (identifier) @enum)
        (method_declaration name: (identifier) @method)
        (import_declaration) @import
        (method_invocation name: (identifier) @method_call)
        """

        try:
            language = tsl.get_language('java')
            query = language.query(query_text)
            captures = query.captures(tree.root_node)

            for node, capture_name in captures:
                text = content[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')

                if capture_name in ['class', 'interface', 'enum']:
                    symbols.append(text)
                elif capture_name == 'method':
                    symbols.append(text)
                elif capture_name == 'import':
                    imports.append(text)
                elif capture_name == 'method_call':
                    references.append(text)

        except Exception as e:
            logger.debug(f"Java symbol extraction failed: {e}")

        return symbols, imports, references

    def _extract_go_symbols(self, tree, content: bytes) -> Tuple[List[str], List[str], List[str]]:
        """Extract Go symbols using tree-sitter."""
        symbols = []
        imports = []
        references = []

        query_text = """
        (function_declaration name: (identifier) @function)
        (method_declaration name: (field_identifier) @method)
        (type_declaration (type_spec name: (type_identifier) @type))
        (import_declaration) @import
        (call_expression function: (identifier) @function_call)
        """

        try:
            language = tsl.get_language('go')
            query = language.query(query_text)
            captures = query.captures(tree.root_node)

            for node, capture_name in captures:
                text = content[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')

                if capture_name in ['function', 'method', 'type']:
                    symbols.append(text)
                elif capture_name == 'import':
                    imports.append(text)
                elif capture_name == 'function_call':
                    references.append(text)

        except Exception as e:
            logger.debug(f"Go symbol extraction failed: {e}")

        return symbols, imports, references

    def _extract_rust_symbols(self, tree, content: bytes) -> Tuple[List[str], List[str], List[str]]:
        """Extract Rust symbols using tree-sitter."""
        symbols = []
        imports = []
        references = []

        query_text = """
        (function_item name: (identifier) @function)
        (struct_item name: (type_identifier) @struct)
        (enum_item name: (type_identifier) @enum)
        (trait_item name: (type_identifier) @trait)
        (impl_item trait: (type_identifier) @impl_trait)
        (use_declaration) @use
        (call_expression function: (identifier) @function_call)
        """

        try:
            language = tsl.get_language('rust')
            query = language.query(query_text)
            captures = query.captures(tree.root_node)

            for node, capture_name in captures:
                text = content[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')

                if capture_name in ['function', 'struct', 'enum', 'trait', 'impl_trait']:
                    symbols.append(text)
                elif capture_name == 'use':
                    imports.append(text)
                elif capture_name == 'function_call':
                    references.append(text)

        except Exception as e:
            logger.debug(f"Rust symbol extraction failed: {e}")

        return symbols, imports, references

    def _extract_generic_symbols(self, tree, content: bytes) -> List[str]:
        """Extract generic symbols for any language."""
        symbols = []

        # Try to find common patterns across languages
        try:
            # Walk the tree and look for identifier nodes
            def visit(node):
                if node.type in ['identifier', 'type_identifier', 'field_identifier', 'property_identifier']:
                    text = content[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')
                    if len(text) > 2 and not text.startswith('_'):
                        symbols.append(text)

                for child in node.children:
                    visit(child)

            visit(tree.root_node)

            # Deduplicate while preserving order
            seen = set()
            unique_symbols = []
            for s in symbols:
                if s not in seen:
                    seen.add(s)
                    unique_symbols.append(s)

            return unique_symbols[:100]  # Limit to top 100 symbols

        except Exception as e:
            logger.debug(f"Generic symbol extraction failed: {e}")
            return []