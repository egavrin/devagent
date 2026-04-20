import tsEslintPlugin from "@typescript-eslint/eslint-plugin";
import tsParser from "@typescript-eslint/parser";
import importPlugin from "eslint-plugin-import";
import sonarjs from "eslint-plugin-sonarjs";

const baseFiles = ["**/*.ts", "**/*.tsx"];
const testFiles = [
  "**/*.test.ts",
  "**/*.test.tsx",
  "**/*.spec.ts",
  "**/*.spec.tsx",
  "scripts/**/*.test.ts",
];
const ignoredPatterns = [
  "**/dist/**",
  "**/node_modules/**",
  "**/.turbo/**",
  ".devagent/workspaces/**",
  "dist/**",
];

export default [
  {
    ignores: ignoredPatterns,
  },
  {
    files: baseFiles,
    languageOptions: {
      parser: tsParser,
      ecmaVersion: "latest",
      sourceType: "module",
      parserOptions: {
        tsconfigRootDir: import.meta.dirname,
      },
    },
    plugins: {
      "@typescript-eslint": tsEslintPlugin,
      import: importPlugin,
      sonarjs,
    },
    settings: {
      "import/resolver": {
        typescript: {
          project: ["./tsconfig.json", "./packages/*/tsconfig.json"],
        },
      },
    },
    rules: {
      "complexity": ["warn", 10],
      "max-depth": ["warn", 4],
      "max-lines": ["warn", { max: 600, skipBlankLines: true, skipComments: true }],
      "max-lines-per-function": [
        "warn",
        { max: 60, skipBlankLines: true, skipComments: true, IIFEs: true },
      ],
      "max-params": ["warn", 5],
      "no-console": ["warn", { allow: ["error", "warn"] }],
      "no-warning-comments": ["warn", { terms: ["todo", "fixme", "hack"], location: "start" }],
      "no-unused-vars": "off",
      "@typescript-eslint/consistent-type-imports": [
        "warn",
        { prefer: "type-imports", disallowTypeAnnotations: false },
      ],
      "@typescript-eslint/no-unused-vars": ["warn", { argsIgnorePattern: "^_", varsIgnorePattern: "^_" }],
      "import/no-cycle": "warn",
      "import/no-duplicates": "warn",
      "import/order": [
        "warn",
        {
          alphabetize: { order: "asc", caseInsensitive: true },
          groups: [["builtin", "external"], ["internal", "parent", "sibling", "index", "object", "type"]],
          "newlines-between": "always",
        },
      ],
      "sonarjs/cognitive-complexity": ["warn", 15],
      "sonarjs/no-identical-functions": "warn",
      "sonarjs/no-small-switch": "warn",
    },
  },
  {
    files: baseFiles,
    ignores: testFiles,
    languageOptions: {
      parser: tsParser,
      ecmaVersion: "latest",
      sourceType: "module",
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      "@typescript-eslint/no-floating-promises": "error",
      "@typescript-eslint/no-misused-promises": ["error", { checksVoidReturn: { attributes: false } }],
      "@typescript-eslint/no-unnecessary-type-assertion": "warn",
      "@typescript-eslint/switch-exhaustiveness-check": "warn",
    },
  },
  {
    files: testFiles,
    rules: {
      "max-lines": ["warn", { max: 1200, skipBlankLines: true, skipComments: true }],
      "max-lines-per-function": [
        "warn",
        { max: 120, skipBlankLines: true, skipComments: true, IIFEs: true },
      ],
      "sonarjs/cognitive-complexity": ["warn", 25],
    },
  },
];
