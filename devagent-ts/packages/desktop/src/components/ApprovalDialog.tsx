/**
 * ApprovalDialog — modal that shows pending tool approval requests.
 *
 * Displays tool name, details (file path, content preview), and
 * Approve / Deny buttons. Stacks if multiple requests arrive.
 */

import { useCallback } from "react";
import type { ApprovalRequest } from "../types";

interface ApprovalDialogProps {
  readonly requests: ReadonlyArray<ApprovalRequest>;
  readonly onRespond: (id: string, approved: boolean) => void;
}

export function ApprovalDialog({
  requests,
  onRespond,
}: ApprovalDialogProps): React.JSX.Element | null {
  const current = requests[0];
  if (!current) return null;

  const handleApprove = useCallback(() => {
    onRespond(current.id, true);
  }, [current.id, onRespond]);

  const handleDeny = useCallback(() => {
    onRespond(current.id, false);
  }, [current.id, onRespond]);

  // Try to parse details as JSON for nicer display
  let detailsContent: React.JSX.Element;
  try {
    const parsed = JSON.parse(current.details) as Record<string, unknown>;
    detailsContent = (
      <div className="approval-details-structured">
        {Object.entries(parsed).map(([key, value]) => (
          <div key={key} className="approval-detail-row">
            <span className="approval-detail-key">{key}:</span>
            <span className="approval-detail-value">
              {typeof value === "string" && value.length > 200
                ? value.substring(0, 200) + "..."
                : String(value)}
            </span>
          </div>
        ))}
      </div>
    );
  } catch {
    detailsContent = (
      <pre className="approval-details-raw">{current.details}</pre>
    );
  }

  return (
    <div className="approval-overlay">
      <div className="approval-dialog">
        <div className="approval-header">
          <span className="approval-icon">&#x26A0;&#xFE0F;</span>
          <h3>Approval Required</h3>
          {requests.length > 1 && (
            <span className="approval-badge">
              +{requests.length - 1} more
            </span>
          )}
        </div>

        <div className="approval-body">
          <div className="approval-tool-name">
            <span className="approval-tool-label">Tool:</span>
            <code>{current.toolName}</code>
          </div>

          <div className="approval-details">{detailsContent}</div>
        </div>

        <div className="approval-actions">
          <button
            className="approval-btn approval-btn-deny"
            onClick={handleDeny}
          >
            Deny
          </button>
          <button
            className="approval-btn approval-btn-approve"
            onClick={handleApprove}
          >
            Approve
          </button>
        </div>
      </div>
    </div>
  );
}
