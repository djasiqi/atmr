import React from 'react';
import {
  TouchableOpacity,
  View,
  StyleSheet,
  TouchableOpacityProps,
  ViewStyle,
} from 'react-native';
import { useColorScheme } from '@/hooks/useColorScheme';

interface CardProps extends TouchableOpacityProps {
  children: React.ReactNode;
  style?: ViewStyle | ViewStyle[];
  contentStyle?: ViewStyle | ViewStyle[];
}

export const Card: React.FC<CardProps> = ({ children, style, contentStyle, ...rest }) => {
  const colorScheme = useColorScheme();

  return (
    <TouchableOpacity
      activeOpacity={0.8}
      style={[
        baseStyles.card,
        {
          backgroundColor: colorScheme === 'dark' ? '#1F2937' : '#FFFFFF',
          shadowColor: '#000',
        },
        style,
      ]}
      {...rest}
    >
      <View style={[baseStyles.content, contentStyle]}>{children}</View>
    </TouchableOpacity>
  );
};

const baseStyles = StyleSheet.create({
  card: {
    borderRadius: 12,
    elevation: 3,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 6,
    marginVertical: 8,
  },
  content: {
    flex: 1,
  },
});
