import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import ts from "typescript";
import { describe, expect, it } from "vitest";

interface StartupSafeModuleRule {
  readonly relativePath: string;
  readonly allowedDynamicImports?: ReadonlyArray<string>;
  readonly forbiddenRuntimeImports?: ReadonlyArray<string>;
}

const STARTUP_SAFE_MODULES: ReadonlyArray<StartupSafeModuleRule> = [
  {
    relativePath: "packages/runtime/src/core/index.ts",
  },
  {
    relativePath: "packages/runtime/src/core/bun-sqlite.ts",
  },
  {
    relativePath: "packages/runtime/src/core/session.ts",
  },
  {
    relativePath: "packages/runtime/src/tools/index.ts",
    forbiddenRuntimeImports: ["../core/index.js"],
  },
  {
    relativePath: "packages/runtime/src/tools/registry.ts",
    forbiddenRuntimeImports: ["../core/index.js"],
  },
  {
    relativePath: "packages/runtime/src/tools/builtins/index.ts",
    forbiddenRuntimeImports: ["../../core/index.js"],
  },
  {
    relativePath: "packages/cli/src/index.ts",
    allowedDynamicImports: ["./main.js", "@devagent/runtime"],
  },
  {
    relativePath: "packages/cli/src/runtime-version.ts",
  },
];

const REPO_ROOT = resolve(import.meta.dirname, "../../..");

describe("startup-safe modules", () => {
  it("avoid top-level await, unexpected dynamic imports, and broad barrel imports", () => {
    for (const rule of STARTUP_SAFE_MODULES) {
      const filePath = join(REPO_ROOT, rule.relativePath);
      const sourceText = readFileSync(filePath, "utf-8");
      const sourceFile = ts.createSourceFile(
        filePath,
        sourceText,
        ts.ScriptTarget.Latest,
        true,
        ts.ScriptKind.TS,
      );

      const dynamicImports = collectDynamicImports(sourceFile);
      const allowedDynamicImports = new Set(rule.allowedDynamicImports ?? []);
      const unexpectedDynamicImports = dynamicImports.filter(
        (specifier) => !allowedDynamicImports.has(specifier),
      );
      expect(unexpectedDynamicImports, `${rule.relativePath} dynamic imports`).toEqual([]);

      const topLevelAwaits = collectTopLevelAwaitLocations(sourceFile);
      expect(topLevelAwaits, `${rule.relativePath} top-level await`).toEqual([]);

      const forbiddenImports = new Set(rule.forbiddenRuntimeImports ?? []);
      if (forbiddenImports.size > 0) {
        const runtimeImports = collectRuntimeImports(sourceFile).filter(
          (specifier) => forbiddenImports.has(specifier),
        );
        expect(runtimeImports, `${rule.relativePath} broad barrel imports`).toEqual([]);
      }
    }
  });
});

function collectDynamicImports(sourceFile: ts.SourceFile): string[] {
  const imports: string[] = [];

  function visit(node: ts.Node): void {
    if (ts.isCallExpression(node) && node.expression.kind === ts.SyntaxKind.ImportKeyword) {
      const [firstArg] = node.arguments;
      if (firstArg && ts.isStringLiteralLike(firstArg)) {
        imports.push(firstArg.text);
      } else {
        imports.push("<non-literal>");
      }
    }
    ts.forEachChild(node, visit);
  }

  visit(sourceFile);
  return imports;
}

function collectTopLevelAwaitLocations(sourceFile: ts.SourceFile): string[] {
  const awaits: string[] = [];

  function visit(node: ts.Node, insideFunction: boolean): void {
    if (ts.isAwaitExpression(node) && !insideFunction) {
      const { line, character } = sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile));
      awaits.push(`${line + 1}:${character + 1}`);
    }

    const nextInsideFunction =
      insideFunction ||
      ts.isFunctionLike(node) ||
      ts.isClassStaticBlockDeclaration(node);

    ts.forEachChild(node, (child) => visit(child, nextInsideFunction));
  }

  visit(sourceFile, false);
  return awaits;
}

function collectRuntimeImports(sourceFile: ts.SourceFile): string[] {
  return sourceFile.statements.flatMap((statement) => {
    if (!ts.isImportDeclaration(statement)) return [];
    if (statement.importClause?.isTypeOnly) return [];
    if (!ts.isStringLiteralLike(statement.moduleSpecifier)) return [];
    return [statement.moduleSpecifier.text];
  });
}
