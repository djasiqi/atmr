import { useState } from "react";
import { Pressable, StyleSheet, View, type LayoutChangeEvent } from "react-native";
import { AppText } from "../../../../design/ui/AppText";
import { M } from "../../../messaging/messagingTheme";
import type { InboxTab } from "../inboxDisplay";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";
import { AnimatedTabIndicator } from "../../../../design/navigation/AnimatedTabIndicator";

const TABS: { id: InboxTab; label: string }[] = [
  { id: "all", label: "TOUTES" },
  { id: "missions", label: "MISSIONS" },
  { id: "contacts", label: "CONTACTS" },
];

type Props = {
  active: InboxTab;
  onChange: (tab: InboxTab) => void;
  /** Conservé pour accessibilité (annonce du nombre de non-lus par onglet). */
  unreadByTab?: Record<InboxTab, number>;
};

type TabRect = { x: number; width: number };

export function MessagesInboxTabs({ active, onChange, unreadByTab }: Props) {
  const [rects, setRects] = useState<Array<TabRect | null>>(() => TABS.map(() => null));
  const activeIndex = Math.max(0, TABS.findIndex((t) => t.id === active));

  const handleLayout = (idx: number) => (event: LayoutChangeEvent) => {
    const { x, width } = event.nativeEvent.layout;
    setRects((prev) => {
      const current = prev[idx];
      if (current && current.x === x && current.width === width) return prev;
      const next = prev.slice();
      next[idx] = { x, width };
      return next;
    });
  };

  return (
    <View style={styles.wrap}>
      {TABS.map((tab, idx) => {
        const selected = active === tab.id;
        const unread = unreadByTab?.[tab.id] ?? 0;
        const a11y =
          unread > 0
            ? `${tab.label}, ${unread} non lu${unread > 1 ? "s" : ""}`
            : tab.label;
        return (
          <Pressable
            key={tab.id}
            style={styles.tab}
            onPress={() => onChange(tab.id)}
            onLayout={handleLayout(idx)}
            accessibilityRole="tab"
            accessibilityState={{ selected }}
            accessibilityLabel={a11y}
          >
            <AppText variant="caption" style={[styles.label, selected && styles.labelActive]}>
              {tab.label}
            </AppText>
          </Pressable>
        );
      })}
      <AnimatedTabIndicator
        activeIndex={activeIndex}
        rects={rects}
        color={M.BRAND}
        screen="messages.inbox"
      />
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    position: "relative",
    flexDirection: "row",
    backgroundColor: M.CARD,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: M.BORDER,
  },
  tab: {
    flex: 1,
    alignItems: "center",
    paddingTop: 10,
    paddingBottom: 13,
  },
  label: {
    fontSize: FONT_SIZE.px13,
    fontWeight: "600",
    letterSpacing: 0.5,
    color: M.TEXT_MUTED,
  },
  labelActive: {
    color: M.BRAND,
    fontWeight: "700",
  },
});
