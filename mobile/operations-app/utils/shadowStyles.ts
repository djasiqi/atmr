// utils/shadowStyles.ts
// Helper pour créer des styles shadow compatibles avec React Native Web
import { Platform, ViewStyle } from 'react-native';

export interface ShadowConfig {
  color: string;
  offset?: { width: number; height: number };
  opacity?: number;
  radius: number;
  elevation?: number;
}

/**
 * Crée un style shadow compatible avec React Native Web
 * Sur le web, utilise boxShadow. Sur iOS/Android, utilise shadow* properties
 */
export function createShadowStyle(config: ShadowConfig): ViewStyle {
  if (Platform.OS === 'web') {
    const { color, offset = { width: 0, height: 0 }, opacity = 0.25, radius } = config;
    // Convertir rgba hex en rgba string pour boxShadow
    const rgbaColor = color.startsWith('#') 
      ? hexToRgba(color, opacity)
      : color.replace(/rgba?\(([^)]+)\)/, (match, values) => {
          const parts = values.split(',').map((v: string) => v.trim());
          if (parts.length === 3) {
            // rgb -> rgba
            return `rgba(${parts.join(',')},${opacity})`;
          }
          return match;
        });
    
    return {
      boxShadow: `${offset.width}px ${offset.height}px ${radius}px ${rgbaColor}`,
    } as ViewStyle;
  }

  // iOS/Android
  return {
    shadowColor: config.color,
    shadowOffset: config.offset || { width: 0, height: 0 },
    shadowOpacity: config.opacity ?? 0.25,
    shadowRadius: config.radius,
    elevation: config.elevation ?? config.radius / 2,
  };
}

/**
 * Convertit une couleur hex en rgba
 */
function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

