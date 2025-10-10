"""RepoMap-based file discovery tool for intelligent context gathering."""

from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import logging

from ai_dev_agent.tools.base import Tool, tool, ToolSpec, ToolContext
from ai_dev_agent.core.repo_map import RepoMapManager

logger = logging.getLogger(__name__)


@tool
class RepoMapTool(Tool):
    """Find relevant files using PageRank-based RepoMap."""

    spec = ToolSpec(
        name="repomap",
        description="Find files related to symbols or concepts using PageRank-based importance ranking",
        parameters={
            "query": {
                "type": "string",
                "description": "Symbols, classes, or concepts to search for (e.g., 'BytecodeOptimizer', 'memory allocation')",
                "required": True
            },
            "max_files": {
                "type": "integer",
                "description": "Maximum number of files to return (default: 20)",
                "required": False
            },
            "include_scores": {
                "type": "boolean",
                "description": "Include PageRank scores in output",
                "required": False
            }
        }
    )

    def _execute(self, context: ToolContext) -> Dict[str, Any]:
        """Execute RepoMap-based file discovery."""
        query = context.parameters["query"]
        max_files = context.parameters.get("max_files", 20)
        include_scores = context.parameters.get("include_scores", False)

        # Get repository root
        repo_root = context.session.get_workspace_root() if context.session else Path.cwd()

        # Get or create RepoMap instance
        rm = RepoMapManager.get_instance(repo_root)

        # Ensure RepoMap is initialized
        if not rm.context.files:
            logger.info(f"Initializing RepoMap for {repo_root}")
            rm.scan_repository()

        # Extract symbols from query
        import re
        words = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', query)
        symbols = set()
        for word in words:
            if len(word) > 2:  # Skip very short words
                symbols.add(word)
                symbols.add(word.lower())
                symbols.add(word.upper())
                # Add variations
                if word[0].isupper():
                    symbols.add(word[0].lower() + word[1:])  # camelCase variant

        # Get ranked files
        try:
            ranked_files = rm.get_ranked_files(
                mentioned_files=set(),
                mentioned_symbols=symbols,
                max_files=max_files
            )
        except Exception as e:
            logger.error(f"RepoMap query failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "files": []
            }

        # Format results
        results = []
        for file_path, score in ranked_files:
            file_info = rm.context.files.get(file_path)

            result = {
                "path": file_path,
                "language": file_info.language if file_info else "unknown",
                "symbols": len(file_info.symbols) if file_info else 0
            }

            if include_scores:
                result["score"] = round(score, 3)
                if rm.context.pagerank_scores:
                    result["pagerank"] = round(rm.context.pagerank_scores.get(file_path, 0), 6)

            # Add relevant symbols found
            if file_info and symbols:
                matching_symbols = [s for s in file_info.symbols
                                  if any(sym.lower() in s.lower() for sym in symbols)][:5]
                if matching_symbols:
                    result["matching_symbols"] = matching_symbols

            results.append(result)

        # Group by language for better overview
        by_language = {}
        for result in results:
            lang = result["language"]
            if lang not in by_language:
                by_language[lang] = []
            by_language[lang].append(result)

        return {
            "status": "success",
            "query": query,
            "symbols_extracted": list(symbols)[:10],  # Show first 10
            "total_files": len(results),
            "files": results,
            "by_language": {lang: len(files) for lang, files in by_language.items()},
            "repomap_stats": {
                "total_indexed": len(rm.context.files),
                "languages": len(set(f.language for f in rm.context.files.values() if f.language)),
                "has_pagerank": bool(rm.context.pagerank_scores)
            }
        }


# Optionally create an even simpler find tool that uses RepoMap
@tool
class SmartFindTool(Tool):
    """Smart file finder that uses RepoMap for instant, ranked results."""

    spec = ToolSpec(
        name="smartfind",
        description="Instantly find files related to any symbol or concept, ranked by importance",
        parameters={
            "pattern": {
                "type": "string",
                "description": "Symbol, class name, or concept to find (e.g., 'BytecodeOptimizer')",
                "required": True
            }
        }
    )

    def _execute(self, context: ToolContext) -> Dict[str, Any]:
        """Execute smart find using RepoMap."""
        pattern = context.parameters["pattern"]

        # Delegate to RepoMapTool with simplified output
        repo_tool = RepoMapTool()
        result = repo_tool._execute(ToolContext(
            session=context.session,
            parameters={"query": pattern, "max_files": 15, "include_scores": True}
        ))

        if result["status"] != "success":
            return result

        # Simplify output for agent
        output_lines = [f"Found {result['total_files']} files related to '{pattern}':"]

        # Show top files
        for i, file_data in enumerate(result["files"][:10], 1):
            path = Path(file_data["path"])
            score = file_data.get("score", 0)
            lang = file_data.get("language", "?")
            output_lines.append(f"{i:2}. {path.name:40} [{lang:8}] score: {score:.2f}")

            # Show matching symbols if any
            if file_data.get("matching_symbols"):
                symbols = ", ".join(file_data["matching_symbols"][:3])
                output_lines.append(f"    Contains: {symbols}")

        # Summary by language
        if result.get("by_language"):
            output_lines.append("\nBy language:")
            for lang, count in result["by_language"].items():
                output_lines.append(f"  {lang}: {count} files")

        return {
            "status": "success",
            "output": "\n".join(output_lines),
            "files": [f["path"] for f in result["files"]],
            "count": result["total_files"]
        }