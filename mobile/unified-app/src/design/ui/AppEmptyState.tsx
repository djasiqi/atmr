import type { ReactNode } from "react";
import { View } from "react-native";
import { brandTextMuted } from "../responsive/brand";
import { useResponsiveTokens } from "../responsive/useResponsiveTokens";
import { AppButton } from "./AppButton";
import { AppText } from "./AppText";

export type AppEmptyStateProps = {
  title: string;
  description?: string;
  icon?: ReactNode;
  actionLabel?: string;
  onAction?: () => void;
};

export function AppEmptyState({
  title,
  description,
  icon,
  actionLabel,
  onAction,
}: AppEmptyStateProps) {
  const t = useResponsiveTokens();

  return (
    <View
      style={{
        alignItems: "center",
        justifyContent: "center",
        gap: t.fieldGap,
        paddingVertical: t.spacingMd,
        paddingHorizontal: t.spacingSm,
      }}
    >
      {icon ? <View style={{ marginBottom: t.spacingXs }}>{icon}</View> : null}
      <AppText variant="sectionTitle" style={{ textAlign: "center" }}>
        {title}
      </AppText>
      {description ? (
        <AppText variant="bodyMuted" style={{ textAlign: "center", color: brandTextMuted }}>
          {description}
        </AppText>
      ) : null}
      {actionLabel && onAction ? (
        <View style={{ marginTop: t.spacingSm, alignSelf: "stretch" }}>
          <AppButton title={actionLabel} variant="secondary" onPress={onAction} />
        </View>
      ) : null}
    </View>
  );
}
