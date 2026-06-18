import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";

import { Dimensions, Platform, View, type StyleProp, type ViewStyle } from "react-native";

import { computeFooterLiftCorrection } from "../keyboardLayoutMetrics";

import { useChatFooterLayout } from "../useKeyboardBottomInset";

import { KeyboardLayoutProvider, useKeyboardLayout } from "../useKeyboardLayout";

import { useChatLayoutKpisPublisher } from "../../../design/responsive/chatLayoutKpis";



const LOG_SHELL_LAYOUT =
  __DEV__ &&
  Platform.OS === "android" &&
  process.env.EXPO_PUBLIC_CHAT_LAYOUT_DEBUG === "1";



type ChatConversationShellProps = {

  header: ReactNode;

  feed: ReactNode;

  footer: ReactNode;

  horizontalPadding: number;

  pageBackground?: string;

  style?: StyleProp<ViewStyle>;

};



function useShellWindowGap() {

  const shellRef = useRef<View>(null);

  const [gap, setGap] = useState(0);



  const remeasure = useCallback(() => {

    shellRef.current?.measureInWindow((_x, y, _w, height) => {

      const windowH = Dimensions.get("window").height;

      const next = Math.max(0, Math.round(windowH - (y + height)));

      setGap((prev) => (prev !== next ? next : prev));

    });

  }, []);



  return { shellRef, shellBottomGap: gap, remeasureShellGap: remeasure };

}



export function ChatConversationShell(props: ChatConversationShellProps) {

  return (

    <KeyboardLayoutProvider>

      <ChatConversationShellInner {...props} />

    </KeyboardLayoutProvider>

  );

}



function ChatConversationShellInner({

  header,

  feed,

  footer,

  horizontalPadding,

  pageBackground = "transparent",

  style,

}: ChatConversationShellProps) {

  const [footerHeight, setFooterHeight] = useState(88);

  const [liftCorrection, setLiftCorrection] = useState(0);

  const footerRef = useRef<View>(null);

  const keyboardLayout = useKeyboardLayout();

  const publishChatKpis = useChatLayoutKpisPublisher();

  const { shellRef, shellBottomGap, remeasureShellGap } = useShellWindowGap();

  const { positionStyle, bottomOffset: baseBottomOffset } = useChatFooterLayout({

    horizontalPadding,

    backgroundColor: pageBackground,

    shellBottomGap,

  });



  const effectiveTopY = keyboardLayout.metrics?.effectiveKeyboardTopY ?? keyboardLayout.screenY;

  const bottomOffset = Math.max(0, baseBottomOffset + liftCorrection);



  /** Mesure réelle : aligner le bas du footer sur le haut clavier effectif. */

  useEffect(() => {

    if (

      !keyboardLayout.keyboardVisible ||

      effectiveTopY == null ||

      Platform.OS !== "android"

    ) {

      setLiftCorrection(0);

      return undefined;

    }

    const targetTopY = effectiveTopY;

    const timers = [48, 160, 320].map((ms) =>

      setTimeout(() => {

        footerRef.current?.measureInWindow((_x, y, _w, h) => {

          const correction = computeFooterLiftCorrection(y + h, targetTopY);

          setLiftCorrection((prev) => (prev !== correction ? correction : prev));

        });

      }, ms)

    );

    return () => {

      timers.forEach(clearTimeout);

    };

  }, [effectiveTopY, keyboardLayout.keyboardVisible]);



  useEffect(() => {

    remeasureShellGap();

    const delays = keyboardLayout.keyboardVisible ? [16, 48, 120, 280] : [48, 160];

    const timers = delays.map((ms) => setTimeout(remeasureShellGap, ms));

    return () => {

      timers.forEach(clearTimeout);

    };

  }, [keyboardLayout.keyboardVisible, keyboardLayout.visibleBottomInset, remeasureShellGap]);



  useEffect(() => {

    if (bottomOffset <= 0) return;

    footerRef.current?.measureInWindow((_x, y, _w, h) => {

      const footerBottom = Math.round(y + h);

      const composerKbGap =

        effectiveTopY != null ? Math.round(effectiveTopY - footerBottom) : null;

      publishChatKpis({

        composerKbGap,

        shellBottomGap,

        footerHeight,

      });

      if (!LOG_SHELL_LAYOUT) return;

      console.log("[chat-footer] window", {

        footerBottom,

        effectiveKeyboardTopY: effectiveTopY,

        liftCorrection,

        visibleBottomInset: keyboardLayout.visibleBottomInset,

        shellBottomGap,

        bottomOffset,

        footerHeight,

        composerKbGap,

      });

      if (composerKbGap != null && Math.abs(composerKbGap) > 12) {

        console.warn(

          `[chat-footer] composerKbGap=${composerKbGap}px hors KPI (cible 8 ±4)`

        );

      }

    });

  }, [

    bottomOffset,

    effectiveTopY,

    liftCorrection,

    keyboardLayout.visibleBottomInset,

    shellBottomGap,

    publishChatKpis,
    footerHeight,

  ]);



  const feedBottomReserve = footerHeight + bottomOffset;



  return (

    <View

      ref={shellRef}

      style={[{ flex: 1 }, style]}

      onLayout={() => {

        remeasureShellGap();

      }}

    >

      {header}

      <View style={{ flex: 1, minHeight: 0, paddingBottom: feedBottomReserve }}>{feed}</View>

      <View

        ref={footerRef}

        onLayout={(event) => {

          const next = Math.ceil(event.nativeEvent.layout.height);

          if (next > 0 && next !== footerHeight) {

            setFooterHeight(next);

          }

        }}

        style={[positionStyle, { bottom: bottomOffset }]}

      >

        {footer}

      </View>

    </View>

  );

}


