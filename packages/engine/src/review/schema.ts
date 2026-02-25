/**
 * Review violation schema -- interfaces and JSON Schema for structured LLM output.
 */

// ── Interfaces ──────────────────────────────────────────────────────────────

export type Severity = "error" | "warning" | "info";
export type ReviewChangeType = "added" | "removed";

export interface Violation {
  file: string;
  line: number;
  severity: Severity;
  message: string;
  rule?: string;
  changeType: ReviewChangeType;
  codeSnippet?: string;
}

export interface ReviewSummary {
  totalViolations: number;
  filesReviewed: number;
  ruleName: string;
  discardedViolations?: number;
  unverifiedChunks?: number;
}

export interface ReviewResult {
  violations: Violation[];
  summary: ReviewSummary;
}

// ── JSON Schema for constraining LLM output ─────────────────────────────────

export const VIOLATION_SCHEMA = {
  type: "object" as const,
  properties: {
    violations: {
      type: "array" as const,
      items: {
        type: "object" as const,
        properties: {
          file: {
            type: "string" as const,
            description: "File path where the violation was found",
          },
          line: {
            type: "number" as const,
            description: "Line number of the violation (1-indexed)",
          },
          severity: {
            type: "string" as const,
            enum: ["error", "warning", "info"] as const,
            description: "Severity level of the violation",
          },
          message: {
            type: "string" as const,
            description: "Human-readable description of the violation",
          },
          rule: {
            type: "string" as const,
            description: "Name or ID of the rule that was violated",
          },
          changeType: {
            type: "string" as const,
            enum: ["added", "removed"] as const,
            description:
              "Whether the violation is on an added or removed line",
          },
          codeSnippet: {
            type: "string" as const,
            description: "The code that triggered the violation",
          },
        },
        required: [
          "file",
          "line",
          "severity",
          "message",
          "changeType",
        ] as const,
        additionalProperties: false as const,
      },
      description: "List of violations found during the review",
    },
    summary: {
      type: "object" as const,
      properties: {
        totalViolations: {
          type: "number" as const,
          description: "Total number of violations found",
        },
        filesReviewed: {
          type: "number" as const,
          description: "Number of files that were reviewed",
        },
        ruleName: {
          type: "string" as const,
          description: "Name of the rule used for the review",
        },
        discardedViolations: {
          type: "number" as const,
          description:
            "Number of violations that were discarded during verification",
        },
        unverifiedChunks: {
          type: "number" as const,
          description:
            "Number of chunks that could not be verified against the rule",
        },
      },
      required: [
        "totalViolations",
        "filesReviewed",
        "ruleName",
      ] as const,
      additionalProperties: false as const,
    },
  },
  required: ["violations", "summary"] as const,
  additionalProperties: false as const,
} as const;
