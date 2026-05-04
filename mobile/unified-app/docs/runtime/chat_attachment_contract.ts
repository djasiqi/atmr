export type ChatAttachmentMime =
  | "image/jpeg"
  | "image/png"
  | "application/pdf";

export type ChatAttachmentContract = {
  maxFileSizeBytes: number;
  mimeWhitelist: ChatAttachmentMime[];
  offlineRetryEnabled: boolean;
  resumableUploadEnabled: boolean;
  previewMode: "cdn" | "base64";
};

export const chatAttachmentContract: ChatAttachmentContract = {
  maxFileSizeBytes: 10 * 1024 * 1024,
  mimeWhitelist: ["image/jpeg", "image/png", "application/pdf"],
  offlineRetryEnabled: true,
  resumableUploadEnabled: false,
  previewMode: "cdn",
};
