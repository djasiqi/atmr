import { z } from "zod";

export const accountStatusSchema = z.enum(["active", "inactive", "suspended"]);
export type AccountStatus = z.infer<typeof accountStatusSchema>;

export const contextTypeSchema = z.enum([
  "client",
  "driver",
  "company",
  "institution",
]);
export type ContextType = z.infer<typeof contextTypeSchema>;

export const contextSchema = z.object({
  context_id: z.string().min(1),
  context_type: contextTypeSchema,
  label: z.string().min(1),
  organization_id: z.union([z.number(), z.string(), z.null()]).optional(),
  organization_name: z.string().nullable().optional(),
  permissions: z.array(z.string()),
  is_default: z.boolean(),
  /** true si l’espace (entreprise ou chauffeur lié) est une entreprise de transport (dispatch) éligible à la bascule mobile. */
  allow_mobile_context_switch: z.boolean().optional(),
});
export type AuthContext = z.infer<typeof contextSchema>;

export const onboardingStatusSchema = z
  .object({
    required: z.boolean().optional(),
  })
  .passthrough();

export const bootstrapResponseSchema = z.object({
  bootstrap_version: z.string().min(1),
  is_authenticated: z.boolean(),
  user: z
    .object({
      id: z.union([z.number(), z.string()]),
      public_id: z.string().optional(),
      username: z.string().optional(),
      email: z.string().email().nullable().optional(),
      first_name: z.string().nullable().optional(),
      last_name: z.string().nullable().optional(),
      full_name: z.string().nullable().optional(),
      /** Bascule entreprise/chauffeur: seuls les comptes `COMPANY` côté API. */
      role: z.string().optional(),
    })
    .nullable()
    .optional(),
  account_status: accountStatusSchema,
  onboarding_status: onboardingStatusSchema.optional(),
  available_contexts: z.array(contextSchema),
  active_context_id: z.string().nullable(),
  feature_flags: z.record(z.union([z.boolean(), z.number(), z.string(), z.null()])),
  min_supported_app_version: z.string(),
  maintenance_mode: z.boolean(),
  degraded_mode: z.boolean(),
  server_time: z.string(),
  request_id: z.string().min(8),
  status_dictionary_version: z.string().optional(),
  pricing_contract_version: z.string().optional(),
  canonical_address_contract_version: z.string().optional(),
  preview_contract_version: z.string().optional(),
  mission_status_version: z.string().optional(),
  mission_snapshot_version: z.string().optional(),
  driver_socket_contract_version: z.string().optional(),
  driver_tracking_contract_version: z.string().optional(),
});
export type BootstrapResponse = z.infer<typeof bootstrapResponseSchema>;

export const switchContextRequestSchema = z.object({
  target_context_id: z.string().min(1),
});
export type SwitchContextRequest = z.infer<typeof switchContextRequestSchema>;

export const switchContextResponseSchema = z.object({
  success: z.boolean(),
  active_context_id: z.string(),
  available_contexts: z.array(contextSchema).optional(),
  feature_flags: z
    .record(z.union([z.boolean(), z.number(), z.string(), z.null()]))
    .optional(),
  preview_contract_version: z.string().optional(),
  request_id: z.string(),
});
export type SwitchContextResponse = z.infer<typeof switchContextResponseSchema>;

export const apiErrorSchema = z.object({
  error_code: z.string(),
  error_message: z.string().optional(),
  action_hint: z
    .enum(["retry", "logout", "contact_support", "force_update", "open_context_selector"])
    .optional(),
  retryable: z.boolean().optional(),
});
export type ApiError = z.infer<typeof apiErrorSchema>;

export function resolveDefaultContext(
  contexts: AuthContext[],
  activeContextId: string | null
): AuthContext | null {
  if (activeContextId) {
    const active = contexts.find((ctx) => ctx.context_id === activeContextId);
    if (active) return active;
  }
  return contexts.find((ctx) => ctx.is_default) ?? contexts[0] ?? null;
}

export function hasPermission(
  context: AuthContext | null | undefined,
  permission: string
): boolean {
  if (!context) return false;
  return context.permissions.includes(permission);
}
