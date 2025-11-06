import React from 'react';
import { TouchableOpacity, Text, ActivityIndicator, TouchableOpacityProps } from 'react-native';
import { useColorScheme } from '@/hooks/useColorScheme';

interface ButtonProps extends TouchableOpacityProps {
  variant?: 'primary' | 'secondary';
  loading?: boolean;
}

export const Button: React.FC<ButtonProps> = ({
  children,
  variant = 'primary',
  loading = false,
  disabled,
  style,
  ...rest
}) => {
  const colorScheme = useColorScheme();

  const variants = {
    primary: {
      container: colorScheme === 'dark' ? 'bg-blue-600' : 'bg-blue-500',
      text: 'text-white',
    },
    secondary: {
      container: colorScheme === 'dark' ? 'bg-gray-700' : 'bg-gray-200',
      text: colorScheme === 'dark' ? 'text-white' : 'text-black',
    },
  };

  const containerClass = `rounded-md py-3 px-4 flex-row justify-center items-center ${
    variants[variant].container
  } ${disabled || loading ? 'opacity-50' : ''}`;

  const textClass = `text-base font-semibold ${variants[variant].text}`;

  return (
    <TouchableOpacity
      className={containerClass}
      activeOpacity={0.8}
      disabled={disabled || loading}
      {...rest}
    >
      {loading ? (
        <ActivityIndicator size="small" color={variants[variant].text.includes('white') ? '#fff' : '#000'} />
      ) : (
        <Text className={textClass}>{children}</Text>
      )}
    </TouchableOpacity>
  );
};