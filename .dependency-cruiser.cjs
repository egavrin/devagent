module.exports = {
  forbidden: [
    {
      name: "no-circular",
      severity: "warn",
      from: {},
      to: { circular: true },
    },
    {
      name: "runtime-no-cli-imports",
      severity: "error",
      from: { path: "^packages/runtime/src" },
      to: { path: "^packages/cli/src" },
    },
    {
      name: "runtime-no-executor-imports",
      severity: "error",
      from: { path: "^packages/runtime/src" },
      to: { path: "^packages/executor/src" },
    },
    {
      name: "runtime-no-providers-imports",
      severity: "error",
      from: { path: "^packages/runtime/src" },
      to: { path: "^packages/providers/src" },
    },
    {
      name: "providers-no-cli-imports",
      severity: "error",
      from: { path: "^packages/providers/src" },
      to: { path: "^packages/cli/src" },
    },
    {
      name: "providers-no-executor-imports",
      severity: "error",
      from: { path: "^packages/providers/src" },
      to: { path: "^packages/executor/src" },
    },
    {
      name: "executor-no-cli-imports",
      severity: "error",
      from: { path: "^packages/executor/src" },
      to: { path: "^packages/cli/src" },
    },
    {
      name: "no-cross-package-src-imports",
      severity: "warn",
      from: { path: "^packages/([^/]+)/src" },
      to: {
        path: "^packages/([^/]+)/src",
        pathNot: "^packages/$1/src",
      },
    },
  ],
  options: {
    tsConfig: {
      fileName: "./tsconfig.json",
    },
    exclude: {
      path: ["^node_modules", "^dist", "^\.devagent/workspaces"],
    },
    doNotFollow: {
      path: "^node_modules",
    },
    includeOnly: "^packages",
    reporterOptions: {
      dot: {
        collapsePattern: "node_modules/[^/]+",
      },
    },
  },
};
