import { useCallback, useEffect, useLayoutEffect, useMemo, useRef } from "react";
import { InteractionManager, type FlatList } from "react-native";
import type { HubChatMessage } from "./types";
import {
  resolveHubFeedScrollPlan,
  type HubFeedInitialAnchorMode,
  type HubFeedScrollItem,
  type HubFeedScrollPlan,
} from "./hubFeedInitialScroll";

const SETTLE_MS = 140;
const FALLBACK_MS = 400;
const MAX_MEDIA_RETRIES = 12;

type Params = {
  listRef: React.RefObject<FlatList<HubFeedScrollItem> | null>;
  feedItems: HubFeedScrollItem[];
  messages: HubChatMessage[];
  ownSenderId: string | null;
  listAnchorKey: number;
  loading: boolean;
  loadingMore: boolean;
  initialAnchorMode: HubFeedInitialAnchorMode;
  onInitialScrollSettled?: () => void;
};

export function useHubFeedInitialScroll({
  listRef,
  feedItems,
  messages,
  ownSenderId,
  listAnchorKey,
  loading,
  loadingMore,
  initialAnchorMode,
  onInitialScrollSettled,
}: Params) {
  const scrollRequestRef = useRef(0);
  const satisfiedRequestRef = useRef(0);
  const scrollPlanRef = useRef<HubFeedScrollPlan | null>(null);
  const prevAnchorKeyRef = useRef<number | null>(null);
  const hadMessagesRef = useRef(false);
  const settleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const fallbackTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mediaRetryRef = useRef(0);
  const settledNotifiedRef = useRef(false);

  const scrollPlan = useMemo(() => {
    if (loading || feedItems.length === 0) return null;
    return resolveHubFeedScrollPlan(messages, feedItems, ownSenderId, initialAnchorMode);
  }, [feedItems, initialAnchorMode, loading, messages, ownSenderId]);

  const useLatestAnchor = initialAnchorMode === "latest";
  const initialScrollIndex =
    useLatestAnchor && feedItems.length > 0
      ? feedItems.length - 1
      : scrollPlan?.mode === "index"
        ? scrollPlan.feedIndex
        : undefined;

  const notifySettled = useCallback(() => {
    if (settledNotifiedRef.current) return;
    settledNotifiedRef.current = true;
    onInitialScrollSettled?.();
  }, [onInitialScrollSettled]);

  const applyPlan = useCallback(
    (plan: HubFeedScrollPlan): boolean => {
      const list = listRef.current;
      if (!list || feedItems.length === 0) return false;
      try {
        if (plan.mode === "index") {
          list.scrollToIndex({
            index: plan.feedIndex,
            animated: false,
            viewPosition: 0.2,
          });
        } else {
          list.scrollToEnd({ animated: false });
        }
        return true;
      } catch {
        try {
          list.scrollToEnd({ animated: false });
          return true;
        } catch {
          return false;
        }
      }
    },
    [feedItems.length, listRef]
  );

  const attemptScroll = useCallback(
    (requestId: number) => {
      if (loading || loadingMore || feedItems.length === 0) return;
      if (requestId === 0 || satisfiedRequestRef.current >= requestId) return;
      const plan = scrollPlanRef.current;
      if (!plan) return;

      const run = () => {
        if (satisfiedRequestRef.current >= requestId) return;
        const applied = applyPlan(plan);
        if (!applied) return;
        satisfiedRequestRef.current = requestId;
        notifySettled();

        if (settleTimerRef.current) clearTimeout(settleTimerRef.current);
        settleTimerRef.current = setTimeout(() => {
          settleTimerRef.current = null;
          const list = listRef.current;
          if (!list || scrollRequestRef.current !== requestId) return;
          applyPlan(plan);
        }, SETTLE_MS);
      };

      InteractionManager.runAfterInteractions(() => {
        requestAnimationFrame(() => {
          requestAnimationFrame(run);
        });
      });
    },
    [applyPlan, feedItems.length, listRef, loading, loadingMore, notifySettled]
  );

  const scheduleRestore = useCallback(
    (requestId: number) => {
      attemptScroll(requestId);
      if (fallbackTimerRef.current) clearTimeout(fallbackTimerRef.current);
      fallbackTimerRef.current = setTimeout(() => {
        fallbackTimerRef.current = null;
        if (satisfiedRequestRef.current < requestId) {
          attemptScroll(requestId);
        }
      }, FALLBACK_MS);
    },
    [attemptScroll]
  );

  const scheduleMediaRelayout = useCallback(() => {
    if (loadingMore || loading) return;
    const requestId = scrollRequestRef.current;
    if (requestId === 0) return;
    if (initialAnchorMode === "latest") {
      // Team-only behavior: once initial restore is done, never re-anchor.
      if (satisfiedRequestRef.current >= requestId) return;
      mediaRetryRef.current += 1;
      if (mediaRetryRef.current > MAX_MEDIA_RETRIES) return;
      const plan = scrollPlanRef.current;
      if (!plan) return;
      const list = listRef.current;
      if (!list) return;
      applyPlan(plan);
      return;
    }

    // Default behavior for mission/company_driver/dispatch/support.
    if (satisfiedRequestRef.current < requestId) {
      mediaRetryRef.current += 1;
      if (mediaRetryRef.current > MAX_MEDIA_RETRIES) return;
      const plan = scrollPlanRef.current;
      if (!plan) return;
      const list = listRef.current;
      if (!list) return;
      applyPlan(plan);
      return;
    }
    if (mediaRetryRef.current >= MAX_MEDIA_RETRIES) return;
    mediaRetryRef.current += 1;
    satisfiedRequestRef.current = requestId - 1;
    attemptScroll(requestId);
  }, [applyPlan, attemptScroll, initialAnchorMode, listRef, loading, loadingMore]);

  useLayoutEffect(() => {
    if (loading) return;
    if (feedItems.length === 0) {
      hadMessagesRef.current = false;
      scrollPlanRef.current = null;
      return;
    }

    const anchorBumped = prevAnchorKeyRef.current !== listAnchorKey;
    const firstDataPaint = !hadMessagesRef.current;
    if (!anchorBumped && !firstDataPaint) return;

    hadMessagesRef.current = true;
    prevAnchorKeyRef.current = listAnchorKey;
    settledNotifiedRef.current = false;
    mediaRetryRef.current = 0;
    satisfiedRequestRef.current = 0;

    const plan =
      scrollPlan ??
      resolveHubFeedScrollPlan(messages, feedItems, ownSenderId, initialAnchorMode);
    scrollPlanRef.current = plan;

    scrollRequestRef.current += 1;
    const requestId = scrollRequestRef.current;
    scheduleRestore(requestId);
  }, [
    feedItems,
    feedItems.length,
    listAnchorKey,
    loading,
    messages,
    ownSenderId,
    initialAnchorMode,
    scheduleRestore,
    scrollPlan,
  ]);

  useEffect(
    () => () => {
      if (settleTimerRef.current) clearTimeout(settleTimerRef.current);
      if (fallbackTimerRef.current) clearTimeout(fallbackTimerRef.current);
    },
    []
  );

  /** Marquer lu / fallback si le fil est vide ou le scroll ne se déclenche pas. */
  useEffect(() => {
    if (loading) return;
    const timer = setTimeout(() => notifySettled(), 2200);
    return () => clearTimeout(timer);
  }, [listAnchorKey, loading, notifySettled]);

  const onContentSizeChange = useCallback(() => {
    if (loading || loadingMore || feedItems.length === 0) return;
    const requestId = scrollRequestRef.current;
    if (requestId === 0) return;
    if (satisfiedRequestRef.current >= requestId) {
      if (initialAnchorMode === "latest") return;
      scheduleMediaRelayout();
      return;
    }
    attemptScroll(requestId);
  }, [
    attemptScroll,
    feedItems.length,
    initialAnchorMode,
    loading,
    loadingMore,
    scheduleMediaRelayout,
  ]);

  const onScrollToIndexFailed = useCallback(
    (info: { index: number; averageItemLength: number }) => {
      const list = listRef.current;
      if (!list) return;
      const offset = Math.max(0, info.index * (info.averageItemLength || 88));
      list.scrollToOffset({ offset, animated: false });
      setTimeout(() => {
        try {
          list.scrollToIndex({ index: info.index, animated: false, viewPosition: 0.2 });
        } catch {
          list.scrollToEnd({ animated: false });
        }
        const requestId = scrollRequestRef.current;
        if (requestId > 0) {
          satisfiedRequestRef.current = requestId;
          notifySettled();
        }
      }, 50);
    },
    [listRef, notifySettled]
  );

  const initialNumToRender =
    initialScrollIndex != null
      ? Math.min(feedItems.length, 18)
      : Math.min(feedItems.length, 24);

  return {
    initialScrollIndex,
    initialNumToRender,
    onContentSizeChange,
    onScrollToIndexFailed,
    scheduleMediaRelayout,
  };
}
