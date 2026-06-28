/**
 * Core model type definitions shared across the frontend.
 */

export type ModelTaskKind =
  | "text_generation"
  | "embeddings"
  | "rerank"
  | "image_generation"
  | "text2speech"
  | "speech2text";

export type EmbeddingsPooling = "CLS" | "LAST" | "MEAN";

export interface EmbeddingsParams {
  normalize: boolean;
  pooling: EmbeddingsPooling;
  truncate: boolean;
  num_streams: number;
}
