import type { ReactNode } from "react";
import { View, type ViewProps } from "react-native";
import { useResponsiveTokens } from "../responsive/useResponsiveTokens";
import { AppText } from "./AppText";

export type AppSectionProps = ViewProps & {
  title?: string;
  children: ReactNode;
};

export function AppSection({ title, children, style, ...rest }: AppSectionProps) {
  const t = useResponsiveTokens();

  return (
    <View style={[{ gap: t.sectionGap }, style]} {...rest}>
      {title ? <AppText variant="sectionTitle">{title}</AppText> : null}
      {children}
    </View>
  );
}
