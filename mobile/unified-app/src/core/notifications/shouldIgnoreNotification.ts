type NotificationFilterContext = {
  contextType: string | null | undefined;
  userId: string | number | null | undefined;
  companyId?: string | number | null | undefined;
};

type NotificationFilterDecision = {
  ignore: boolean;
  reason?: string;
};

function normalizeIdentifier(value: unknown): string | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value);
  }
  if (typeof value === "string") {
    const normalized = value.trim();
    return normalized.length > 0 ? normalized : null;
  }
  return null;
}

function asObject(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object") return null;
  return value as Record<string, unknown>;
}

export function shouldIgnoreNotification(
  payload: unknown,
  context: NotificationFilterContext
): NotificationFilterDecision {
  if (context.contextType !== "driver") {
    return { ignore: false };
  }

  const data = asObject(payload);
  if (!data) {
    return { ignore: true, reason: "invalid_payload" };
  }

  const recipientRole = typeof data.recipient_role === "string" ? data.recipient_role : null;
  if (recipientRole && recipientRole !== "driver") {
    return { ignore: true, reason: "recipient_role_mismatch" };
  }

  const actor = asObject(data.actor);
  const actorId = normalizeIdentifier(data.actor_id ?? actor?.id);
  const actorRoleRaw = data.actor_role ?? actor?.role;
  const actorRole = typeof actorRoleRaw === "string" ? actorRoleRaw : null;
  const currentUserId = normalizeIdentifier(context.userId);
  if (
    actorId &&
    currentUserId &&
    actorId === currentUserId &&
    (actorRole === "driver" || actorRole == null)
  ) {
    return { ignore: true, reason: "self_actor" };
  }

  const payloadCompanyId = normalizeIdentifier(
    data.company_id ?? data.companyId ?? asObject(data.company)?.id
  );
  const contextCompanyId = normalizeIdentifier(context.companyId);
  if (payloadCompanyId && contextCompanyId && payloadCompanyId !== contextCompanyId) {
    return { ignore: true, reason: "company_mismatch" };
  }

  return { ignore: false };
}
