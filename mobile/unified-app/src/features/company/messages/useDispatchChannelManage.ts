import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  addConversationParticipant,
  clearChannelHistory,
  fetchChannelManageDetail,
  removeConversationParticipant,
  updateChannelManageDetail,
  type ChannelManagePayload,
} from "./conversationManageApi";
import { HUB_KEY } from "./hooks";
import { useCompanyNumericId } from "./companyId";

const manageKey = (conversationId: number) =>
  ["company", "conversation-manage", conversationId] as const;

function patchDispatchThreadTitle(
  queryClient: ReturnType<typeof useQueryClient>,
  companyId: number,
  conversationId: number,
  payload: ChannelManagePayload
) {
  queryClient.setQueriesData<{ threads: { thread_id?: string; conversation_id?: number; title?: string; subtitle?: string }[] }>(
    { queryKey: [...HUB_KEY, "threads", companyId] },
    (old) => {
      if (!old?.threads?.length) return old;
      return {
        ...old,
        threads: old.threads.map((t) => {
          const isDispatch =
            t.thread_id === "dispatch" || t.conversation_id === conversationId;
          if (!isDispatch) return t;
          return {
            ...t,
            title: payload.channel.title,
            subtitle: payload.channel.description || t.subtitle,
          };
        }),
      };
    }
  );
}

function invalidateDispatchHub(
  queryClient: ReturnType<typeof useQueryClient>,
  companyId: number,
  threadId = "dispatch"
) {
  void queryClient.invalidateQueries({ queryKey: [...HUB_KEY, "threads", companyId] });
  void queryClient.invalidateQueries({
    queryKey: [...HUB_KEY, "messages", companyId, threadId],
  });
  void queryClient.invalidateQueries({ queryKey: [...HUB_KEY, "unread", companyId] });
}

export function useDispatchChannelManage(conversationId: number | null) {
  const queryClient = useQueryClient();
  const companyId = useCompanyNumericId();

  const detailQuery = useQuery({
    queryKey: [...manageKey(conversationId ?? 0), "detail"],
    enabled: conversationId != null,
    queryFn: () => fetchChannelManageDetail(conversationId as number),
  });

  const invalidate = () => {
    if (!conversationId) return;
    void queryClient.invalidateQueries({ queryKey: manageKey(conversationId) });
  };

  const updateChannel = useMutation({
    mutationFn: (body: { title?: string; description?: string }) =>
      updateChannelManageDetail(conversationId as number, body),
    onSuccess: (payload) => {
      invalidate();
      if (companyId && conversationId) {
        patchDispatchThreadTitle(queryClient, companyId, conversationId, payload);
        invalidateDispatchHub(queryClient, companyId);
      }
    },
  });

  const clearHistory = useMutation({
    mutationFn: () => clearChannelHistory(conversationId as number),
    onSuccess: () => {
      invalidate();
      if (companyId) {
        invalidateDispatchHub(queryClient, companyId);
      }
    },
  });

  const addParticipant = useMutation({
    mutationFn: (driverId: number) =>
      addConversationParticipant(conversationId as number, driverId),
    onSuccess: () => invalidate(),
  });

  const removeParticipant = useMutation({
    mutationFn: (userId: number) =>
      removeConversationParticipant(conversationId as number, userId),
    onSuccess: () => invalidate(),
  });

  return {
    detailQuery,
    updateChannel,
    addParticipant,
    removeParticipant,
    clearHistory,
  };
}
