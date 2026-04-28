export type SharedChatMessage = {
  id: string | number;
  content: string;
  type?: "text" | "image" | "pdf" | "system" | "audio";
  senderRole?: string;
  senderName?: string | null;
  timestamp: string;
  threadId?: string | null;
  imageUrl?: string | null;
  pdfUrl?: string | null;
  pdfFilename?: string | null;
  /** Fichier audio local (m4a) ou URL résolue côté serveur. */
  audioUrl?: string | null;
  mimeType?: string | null;
};
