import { extname } from "node:path";

export function buildSavedFilename(input: {
  readonly disposition: string;
  readonly finalUrl: string;
  readonly contentType: string;
  readonly sha256: string;
}): string {
  const rawName = getDispositionFilename(input.disposition)
    ?? getUrlFilename(input.finalUrl)
    ?? "download";
  const safeName = sanitizeFilename(rawName, input.contentType);
  const timestamp = new Date().toISOString().replace(/[-:.TZ]/g, "");
  return `${timestamp}-${input.sha256.slice(0, 12)}-${safeName}`;
}

export function sanitizePathSegment(value: string): string {
  return value.replace(/[^a-zA-Z0-9._-]/g, "-") || "session";
}

function getDispositionFilename(disposition: string): string | null {
  return getEncodedDispositionFilename(disposition)
    ?? getPlainDispositionFilename(disposition);
}

function getEncodedDispositionFilename(disposition: string): string | null {
  const encodedMatch = disposition.match(/filename\*\s*=\s*([^;]+)/i);
  if (!encodedMatch) return null;

  const rawValue = encodedMatch[1]?.trim().replace(/^"(.*)"$/, "$1") ?? "";
  const encodedPart = rawValue.split("''")[1] ?? rawValue;
  if (!encodedPart) return null;

  try {
    return decodeURIComponent(encodedPart) || null;
  } catch {
    return encodedPart;
  }
}

function getPlainDispositionFilename(disposition: string): string | null {
  const plainMatch = disposition.match(/filename\s*=\s*(?:"([^"]+)"|([^;]+))/i);
  const filename = plainMatch?.[1] ?? plainMatch?.[2] ?? "";
  return filename.trim() || null;
}

function getUrlFilename(finalUrl: string): string | null {
  try {
    const pathname = new URL(finalUrl).pathname;
    const decoded = decodeURIComponent(pathname);
    const candidate = decoded.split(/[\\/]/).filter(Boolean).pop() ?? "";
    return candidate || null;
  } catch {
    return null;
  }
}

function sanitizeFilename(filename: string, contentType: string): string {
  const basename = filename.split(/[\\/]/).filter(Boolean).pop() ?? filename;
  const sanitizedBase = sanitizeBasename(basename);
  const providedExtension = extname(sanitizedBase);
  const stem = sanitizeStem(
    providedExtension ? sanitizedBase.slice(0, -providedExtension.length) : sanitizedBase,
  );
  const extension = sanitizeExtension(providedExtension || extensionFromMime(contentType));

  return `${stem}${extension}`;
}

function sanitizeBasename(basename: string): string {
  return basename
    .replace(/[\u0000-\u001f\u007f]/g, "")
    .trim()
    .replace(/[<>:"/\\|?*\x00-\x1f]/g, "-")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^\.+/, "")
    .replace(/\.+$/, "");
}

function sanitizeStem(stem: string): string {
  return stem
    .replace(/[^\w.-]/g, "-")
    .replace(/-+/g, "-")
    .replace(/^\.+/, "")
    .replace(/\.+$/, "")
    || "download";
}

function sanitizeExtension(extension: string): string {
  if (!extension) return "";
  const safe = extension
    .replace(/[^a-zA-Z0-9.]/g, "")
    .replace(/^\.+/, ".");
  return safe === "." ? "" : safe;
}

function extensionFromMime(contentType: string): string {
  const mime = contentType.split(";")[0]?.trim().toLowerCase() ?? "";
  return MIME_EXTENSIONS[mime] ?? "";
}

const MIME_EXTENSIONS: Record<string, string> = {
  "application/gzip": ".gz",
  "application/json": ".json",
  "application/pdf": ".pdf",
  "application/zip": ".zip",
  "image/gif": ".gif",
  "image/jpeg": ".jpg",
  "image/png": ".png",
  "image/svg+xml": ".svg",
  "image/webp": ".webp",
};
