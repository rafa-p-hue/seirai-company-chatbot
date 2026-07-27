import type { GroundedAnswer } from "./generate-answer";
import type { RetrievalInspection } from "./retrieve-documents";

export type GroundedAnswerInput = {
  question: string;
  retrieval: RetrievalInspection;
};

export interface EmbeddingProvider {
  readonly name: string;
  readonly dimensions: number;
  readonly modelDownloadSize: string;
  embedText(text: string): Promise<number[]>;
  embedBatch(texts: string[]): Promise<number[][]>;
}

export interface AnswerGenerator {
  readonly name: string;
  generateAnswer(input: GroundedAnswerInput): Promise<GroundedAnswer>;
}
