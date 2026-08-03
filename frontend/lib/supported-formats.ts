/** Formats implemented by the FastAPI ingestion pipeline. */
export const SUPPORTED_UPLOAD_EXTENSIONS = [
  ".pdf",
  ".docx",
  ".html",
  ".htm",
  ".md",
  ".markdown",
  ".csv",
  ".pptx",
] as const;

export const SUPPORTED_UPLOAD_MIME = [
  "application/pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "text/html",
  "application/xhtml+xml",
  "text/markdown",
  "text/x-markdown",
  "text/csv",
  "application/csv",
  "application/vnd.openxmlformats-officedocument.presentationml.presentation",
] as const;

export const UPLOAD_ACCEPT = [
  ...SUPPORTED_UPLOAD_EXTENSIONS,
  ...SUPPORTED_UPLOAD_MIME,
].join(",");

/** Extensions-first accept string — some browsers filter more reliably this way. */
export const UPLOAD_ACCEPT_EXTENSIONS = SUPPORTED_UPLOAD_EXTENSIONS.join(",");

export const SUPPORTED_FORMAT_LABEL = "PDF, DOCX, HTML, MD, CSV, PPTX";

export function isSupportedUploadFile(file: File): boolean {
  const lower = file.name.toLowerCase();
  // Prefer extension — browsers often report markdown/CSV as text/plain or
  // octet-stream, which must not reject a valid supported file.
  if (SUPPORTED_UPLOAD_EXTENSIONS.some((ext) => lower.endsWith(ext))) {
    return true;
  }
  if (!file.type) return false;
  return SUPPORTED_UPLOAD_MIME.includes(
    file.type as (typeof SUPPORTED_UPLOAD_MIME)[number],
  );
}

export function fileTypeLabel(
  filename: string,
  contentType?: string | null,
): string {
  const lower = filename.toLowerCase();
  if (lower.endsWith(".pdf") || contentType === "application/pdf") return "PDF";
  if (lower.endsWith(".docx")) return "DOCX";
  if (lower.endsWith(".html") || lower.endsWith(".htm")) return "HTML";
  if (lower.endsWith(".md") || lower.endsWith(".markdown")) return "MD";
  if (lower.endsWith(".csv")) return "CSV";
  if (lower.endsWith(".pptx")) return "PPTX";
  const ext = lower.includes(".") ? lower.split(".").pop() : "";
  return (ext || "FILE").toUpperCase();
}
