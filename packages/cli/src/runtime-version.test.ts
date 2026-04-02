import { describe, expect, it } from "vitest";

import {
  getUnsupportedRuntimeMessage,
  MINIMUM_BUN_VERSION,
  MINIMUM_NODE_MAJOR,
  parseNodeMajor,
  renderRuntimeBootstrap,
} from "./runtime-version.js";

describe("parseNodeMajor", () => {
  it("reads the major version from a Node version string", () => {
    expect(parseNodeMajor("v20.11.1")).toBe(20);
  });

  it("returns null when the version string is malformed", () => {
    expect(parseNodeMajor("not-a-version")).toBeNull();
  });
});

describe("getUnsupportedRuntimeMessage", () => {
  it("rejects unsupported Node.js runtimes with an actionable message", () => {
    expect(getUnsupportedRuntimeMessage({ nodeVersion: "v19.9.0", hasBun: false })).toContain(
      `DevAgent requires Node.js >= ${MINIMUM_NODE_MAJOR} or Bun >= ${MINIMUM_BUN_VERSION}.`,
    );
  });

  it("allows supported Node.js runtimes", () => {
    expect(getUnsupportedRuntimeMessage({ nodeVersion: "v20.0.0", hasBun: false })).toBeNull();
  });

  it("allows Bun regardless of the embedded Node.js version", () => {
    expect(getUnsupportedRuntimeMessage({ nodeVersion: "v18.0.0", hasBun: true })).toBeNull();
  });
});

describe("renderRuntimeBootstrap", () => {
  it("generates a bootstrap script that checks the Node.js version before importing the bundle", () => {
    const script = renderRuntimeBootstrap("./devagent.js");

    expect(script).toContain('import("./devagent.js")');
    expect(script).toContain(`minimumNodeMajor = ${MINIMUM_NODE_MAJOR}`);
    expect(script).toContain('DevAgent requires Node.js >= " + minimumNodeMajor');
  });
});
