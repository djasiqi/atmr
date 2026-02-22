import React, { useEffect, useRef, useState, useCallback } from "react";
import {
  View,
  Text,
  StyleSheet,
  Animated,
  TouchableOpacity,
  Dimensions,
} from "react-native";

const TOAST_DURATION = 4000;
const SLIDE_DURATION = 300;
const { width: SCREEN_WIDTH } = Dimensions.get("window");

export type InAppNotification = {
  title: string;
  body: string;
  data?: Record<string, unknown>;
};

type Listener = (notif: InAppNotification) => void;
const listeners = new Set<Listener>();

export function emitInAppNotification(notif: InAppNotification): void {
  listeners.forEach((fn) => fn(notif));
}

export function InAppNotificationToast() {
  const [current, setCurrent] = useState<InAppNotification | null>(null);
  const translateY = useRef(new Animated.Value(-120)).current;
  const hideTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  const dismiss = useCallback(() => {
    Animated.timing(translateY, {
      toValue: -120,
      duration: SLIDE_DURATION,
      useNativeDriver: true,
    }).start(() => setCurrent(null));
  }, [translateY]);

  const show = useCallback(
    (notif: InAppNotification) => {
      if (hideTimeout.current) clearTimeout(hideTimeout.current);

      setCurrent(notif);
      translateY.setValue(-120);

      Animated.timing(translateY, {
        toValue: 0,
        duration: SLIDE_DURATION,
        useNativeDriver: true,
      }).start();

      hideTimeout.current = setTimeout(dismiss, TOAST_DURATION);
    },
    [translateY, dismiss]
  );

  useEffect(() => {
    listeners.add(show);
    return () => {
      listeners.delete(show);
      if (hideTimeout.current) clearTimeout(hideTimeout.current);
    };
  }, [show]);

  if (!current) return null;

  return (
    <Animated.View
      style={[styles.container, { transform: [{ translateY }] }]}
      pointerEvents="box-none"
    >
      <TouchableOpacity
        style={styles.toast}
        activeOpacity={0.9}
        onPress={dismiss}
      >
        <View style={styles.accent} />
        <View style={styles.content}>
          <Text style={styles.title} numberOfLines={1}>
            {current.title}
          </Text>
          <Text style={styles.body} numberOfLines={2}>
            {current.body}
          </Text>
        </View>
      </TouchableOpacity>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  container: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    zIndex: 9999,
    paddingTop: 48,
    paddingHorizontal: 12,
  },
  toast: {
    flexDirection: "row",
    backgroundColor: "#FFFFFF",
    borderRadius: 12,
    overflow: "hidden",
    maxWidth: SCREEN_WIDTH - 24,
    elevation: 8,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.15,
    shadowRadius: 12,
  },
  accent: {
    width: 4,
    backgroundColor: "#0A7F59",
  },
  content: {
    flex: 1,
    paddingVertical: 12,
    paddingHorizontal: 14,
  },
  title: {
    fontSize: 14,
    fontWeight: "700",
    color: "#15362B",
    marginBottom: 2,
  },
  body: {
    fontSize: 13,
    color: "#5F7369",
    lineHeight: 18,
  },
});
