import { describe, it, expect } from "vitest";
import { isReviewQuery } from "./main.js";

describe("isReviewQuery", () => {
  it("matches 'perform detailed code review of uncommitted changes'", () => {
    expect(isReviewQuery("perform detailed code review of uncommitted changes")).toBe("unstaged");
  });

  it("matches 'review my staged changes'", () => {
    expect(isReviewQuery("review my staged changes")).toBe("staged");
  });

  it("matches 'code review the diff'", () => {
    expect(isReviewQuery("code review the diff")).toBe("unstaged");
  });

  it("does not match 'review this PR'", () => {
    expect(isReviewQuery("review this PR")).toBe(false);
  });

  it("does not match 'fix the bug'", () => {
    expect(isReviewQuery("fix the bug")).toBe(false);
  });

  it("matches 'review uncommitted code changes'", () => {
    expect(isReviewQuery("review uncommitted code changes")).toBe("unstaged");
  });

});
