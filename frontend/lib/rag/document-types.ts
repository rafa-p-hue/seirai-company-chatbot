export type DocumentProcessingStatus =
  | "uploaded"
  | "extracting"
  | "cleaning"
  | "chunking"
  | "embedding"
  | "ready"
  | "failed";

export type ExtractedSection = {
  pageNumber?: number;
  slideNumber?: number;
  sheetName?: string;
  heading?: string;
  text: string;
};

export type ExtractedDocument = {
  documentId: string;
  fileName: string;
  mimeType: string;
  sections: ExtractedSection[];
};

export type ChunkContentType =
  | "paragraph"
  | "list"
  | "table"
  | "heading-section"
  | "profile-entry"
  | "policy-section"
  | "unknown";

export type StructuredRecordType =
  | "profile"
  | "experience"
  | "education"
  | "role"
  | "project"
  | "skill"
  | "organization"
  | "policy"
  | "section"
  | "internship"
  | "unknown";

export type StructuredRecord = {
  id: string;
  documentId: string;
  recordType: StructuredRecordType;
  title?: string;
  organization?: string;
  dateText?: string;
  startDate?: string;
  endDate?: string;
  location?: string;
  heading?: string;
  employmentType?: string;
  descriptionLines: string[];
  skills?: string[];
  pageStart?: number;
  pageEnd?: number;
  rawText: string;
};

export type RejectedChunk = {
  content: string;
  reason: string;
  pageStart?: number;
  pageEnd?: number;
  qualityScore?: number;
};

export type ChunkMetadata = {
  documentId: string;
  documentName: string;
  chunkIndex: number;
  pageStart?: number;
  pageEnd?: number;
  heading?: string;
  sectionPath?: string[];
  contentType?: ChunkContentType;
  dateText?: string;
  dateStart?: number;
  dateEnd?: number;
  isCurrent?: boolean;
  recordId?: string;
  recordType?: StructuredRecordType;
  title?: string;
  organization?: string;
  location?: string;
  employmentType?: string;
  qualityScore?: number;
};

export type DocumentChunk = {
  id: string;
  documentId: string;
  documentName: string;
  chunkIndex: number;
  pageStart?: number;
  pageEnd?: number;
  heading?: string;
  sectionPath?: string[];
  contentType?: ChunkContentType;
  content: string;
  originalContent: string;
  characterCount: number;
  tokenCount: number;
  metadata: ChunkMetadata;
  embeddingStatus: "pending" | "complete" | "failed";
  embedding?: number[];
  embeddingError?: string;
  record?: StructuredRecord;
};

export type DocumentOverview = {
  apparentDocumentType: string;
  mainTopics: string[];
  majorSections: string[];
  namedEntities: string[];
  summary: string;
};

export type CleaningInspection = {
  rawSections: ExtractedSection[];
  cleanedSections: ExtractedSection[];
  removedRepeatedLines: string[];
  detectedHeadings: string[];
  warnings: string[];
  structuredRecords?: StructuredRecord[];
  rejectedChunks?: RejectedChunk[];
};

export type ProcessingLogEntry = {
  status: DocumentProcessingStatus;
  message: string;
  progress: number;
  timestamp: string;
  durationMs?: number;
};

export type StoredDocument = {
  id: string;
  name: string;
  mimeType: string;
  fileSize: number;
  pageCount: number;
  characterCount: number;
  chunkCount: number;
  embeddingCount: number;
  status: DocumentProcessingStatus;
  progress: number;
  message: string;
  error?: string;
  uploadedAt: string;
  readyAt?: string;
  overview?: DocumentOverview;
  chunks: DocumentChunk[];
  records?: StructuredRecord[];
  rejectedChunks?: RejectedChunk[];
  inspection: CleaningInspection;
  processingLog: ProcessingLogEntry[];
};

export type PublicDocumentStatus = {
  id: string;
  name: string;
  mimeType: string;
  fileSize: number;
  pageCount: number;
  characterCount: number;
  chunkCount: number;
  embeddingCount: number;
  status: DocumentProcessingStatus;
  progress: number;
  message: string;
  error?: string;
  uploadedAt: string;
  readyAt?: string;
  overview?: DocumentOverview;
};
