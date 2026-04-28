import { Pressable, StyleSheet, View } from 'react-native';

import { ThemedText } from '@/components/ThemedText';

type MobileListCardProps = {
  title: string;
  subtitle?: string;
  meta?: string;
  badge?: string;
  onPress?: () => void;
};

export function MobileListCard({ title, subtitle, meta, badge, onPress }: MobileListCardProps) {
  return (
    <Pressable style={styles.card} onPress={onPress} disabled={!onPress}>
      <View style={styles.row}>
        <ThemedText type="defaultSemiBold" style={styles.title}>
          {title}
        </ThemedText>
        {badge ? (
          <View style={styles.badge}>
            <ThemedText style={styles.badgeText}>{badge}</ThemedText>
          </View>
        ) : null}
      </View>
      {subtitle ? <ThemedText style={styles.subtitle}>{subtitle}</ThemedText> : null}
      {meta ? <ThemedText style={styles.meta}>{meta}</ThemedText> : null}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  card: {
    borderWidth: 1,
    borderColor: '#d7d7d7',
    borderRadius: 12,
    padding: 14,
    gap: 8,
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: 8,
  },
  title: {
    flex: 1,
  },
  subtitle: {
    opacity: 0.85,
  },
  meta: {
    opacity: 0.7,
    fontSize: 13,
  },
  badge: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 99,
    backgroundColor: '#ebf7ff',
  },
  badgeText: {
    color: '#0a7ea4',
    fontSize: 12,
    lineHeight: 18,
  },
});
