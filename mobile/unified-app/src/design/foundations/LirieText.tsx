import type { ReactNode } from "react";
import { Text, type TextProps, type TextStyle } from "react-native";
import { lirieColor } from "./tokens";

export type LirieTextVariant = "title" | "subtitle" | "data" | "description" | "message";

const VARIANT: Record<LirieTextVariant, TextStyle> = {
  title: { fontSize: 22, lineHeight: 28, fontWeight: "700", color: lirieColor.ink },
  subtitle: { fontSize: 15, lineHeight: 20, fontWeight: "600", color: lirieColor.inkMuted },
  data: { fontSize: 17, lineHeight: 22, fontWeight: "700", color: lirieColor.ink },
  description: { fontSize: 15, lineHeight: 22, fontWeight: "400", color: lirieColor.ink },
  message: { fontSize: 14, lineHeight: 20, fontWeight: "400", color: lirieColor.inkMuted },
};

type Props = TextProps & {
  variant: LirieTextVariant;
  children: ReactNode;
};

/** Le plafond 2.0 laisse passer la police système. Aucun texte essentiel n'est bridé. */
export function LirieText({ variant, style, children, ...rest }: Props) {
  return (
    <Text
      {...rest}
      maxFontSizeMultiplier={2}
      style={[VARIANT[variant], style]}
    >
      {children}
    </Text>
  );
}
