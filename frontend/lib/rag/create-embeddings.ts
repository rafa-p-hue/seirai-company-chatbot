import type { DocumentChunk } from "./document-types";
import type { EmbeddingProvider } from "./providers";
import { tokenize } from "./token-utils";

export const EMBEDDING_DIMENSIONS = 384;
export const LOCAL_EMBEDDING_MODEL_NAME = "local-hashed-ngram-embedding-v1";

export type EmbeddingProgress = {
  completed: number;
  total: number;
};

export async function createEmbeddings(
  chunks: DocumentChunk[],
  onProgress?: (progress: EmbeddingProgress) => void,
  provider: EmbeddingProvider = localEmbeddingProvider,
): Promise<DocumentChunk[]> {
  const embeddedChunks: DocumentChunk[] = [];
  const embeddings = await provider.embedBatch(
    chunks.map((chunk) => buildEmbeddingText(chunk)),
  );

  for (const [index, chunk] of chunks.entries()) {
    try {
      const embedding = embeddings[index];

      validateEmbedding(embedding, provider.dimensions);

      embeddedChunks.push({
        ...chunk,
        embedding,
        embeddingStatus: "complete",
      });
    } catch (error) {
      embeddedChunks.push({
        ...chunk,
        embeddingStatus: "failed",
        embeddingError:
          error instanceof Error ? error.message : "Embedding failed.",
      });
    }

    onProgress?.({
      completed: embeddedChunks.length,
      total: chunks.length,
    });
  }

  return embeddedChunks;
}

export function createQuestionEmbedding(question: string): number[] {
  return createLocalEmbedding(question);
}

export function createLocalEmbedding(text: string): number[] {
  const vector = Array.from({ length: EMBEDDING_DIMENSIONS }, () => 0);
  const terms = tokenize(text).filter((term) => term.length > 1);

  for (let index = 0; index < terms.length; index += 1) {
    addFeature(vector, terms[index], 1);

    if (terms[index + 1]) {
      addFeature(vector, `${terms[index]} ${terms[index + 1]}`, 1.4);
    }

    if (terms[index + 2]) {
      addFeature(
        vector,
        `${terms[index]} ${terms[index + 1]} ${terms[index + 2]}`,
        1.8,
      );
    }
  }

  return normalizeVector(vector);
}

export const localEmbeddingProvider: EmbeddingProvider = {
  name: LOCAL_EMBEDDING_MODEL_NAME,
  dimensions: EMBEDDING_DIMENSIONS,
  modelDownloadSize: "0 MB (deterministic local provider; no external model download)",
  async embedText(text: string) {
    return createLocalEmbedding(text);
  },
  async embedBatch(texts: string[]) {
    return texts.map((text) => createLocalEmbedding(text));
  },
};

export function cosineSimilarity(first: number[], second: number[]): number {
  if (first.length !== second.length) {
    return 0;
  }

  let dot = 0;
  let firstMagnitude = 0;
  let secondMagnitude = 0;

  for (let index = 0; index < first.length; index += 1) {
    dot += first[index] * second[index];
    firstMagnitude += first[index] ** 2;
    secondMagnitude += second[index] ** 2;
  }

  if (firstMagnitude === 0 || secondMagnitude === 0) {
    return 0;
  }

  return dot / (Math.sqrt(firstMagnitude) * Math.sqrt(secondMagnitude));
}

function normalizeVector(vector: number[]): number[] {
  const magnitude = Math.sqrt(
    vector.reduce((total, value) => total + value ** 2, 0),
  );

  if (magnitude === 0) {
    return vector;
  }

  return vector.map((value) => value / magnitude);
}

function addFeature(vector: number[], feature: string, weight: number): void {
  const index = hashTerm(feature) % EMBEDDING_DIMENSIONS;
  vector[index] += weight;
}

function buildEmbeddingText(chunk: DocumentChunk): string {
  if (chunk.record) {
    const lines = [
      chunk.record.title ? `Title: ${chunk.record.title}` : undefined,
      chunk.record.organization
        ? `Organization: ${chunk.record.organization}`
        : undefined,
      chunk.record.dateText ? `Dates: ${chunk.record.dateText}` : undefined,
      chunk.record.location ? `Location: ${chunk.record.location}` : undefined,
      chunk.record.heading ? `Section: ${chunk.record.heading}` : undefined,
      chunk.record.employmentType
        ? `Employment type: ${chunk.record.employmentType}`
        : undefined,
      chunk.record.skills?.length
        ? `Skills: ${chunk.record.skills.join(", ")}`
        : undefined,
      chunk.record.descriptionLines.length > 0
        ? `Description:\n${chunk.record.descriptionLines.join("\n")}`
        : undefined,
    ].filter(Boolean);

    if (lines.length > 0) {
      return lines.join("\n");
    }
  }

  return [
    chunk.metadata.recordType,
    chunk.metadata.title,
    chunk.metadata.organization,
    chunk.metadata.heading,
    chunk.metadata.sectionPath?.join(" "),
    chunk.metadata.contentType,
    chunk.originalContent || chunk.content,
  ]
    .filter(Boolean)
    .join("\n");
}

function validateEmbedding(embedding: number[], dimensions: number): void {
  if (!Array.isArray(embedding) || embedding.length !== dimensions) {
    throw new Error(
      `Embedding dimension mismatch. Expected ${dimensions}, received ${embedding?.length ?? 0}.`,
    );
  }
}

function hashTerm(term: string): number {
  let hash = 2166136261;

  for (let index = 0; index < term.length; index += 1) {
    hash ^= term.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }

  return Math.abs(hash);
}
