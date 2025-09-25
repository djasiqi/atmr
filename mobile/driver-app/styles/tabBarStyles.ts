import { Platform, ViewStyle, TextStyle } from 'react-native';

type ViewStyleStrict = ViewStyle & {
  alignItems?: 'flex-start' | 'flex-end' | 'center' | 'stretch' | 'baseline';
  justifyContent?: 'flex-start' | 'flex-end' | 'center' | 'space-between' | 'space-around' | 'space-evenly';
};

export const tabBarStyles = {
  tabBarStyle: {
    position: 'absolute',
    borderTopWidth: 1,
    backgroundColor: '#065f46', // vert foncé
    borderTopColor: '#064e3b',
    elevation: 8,
    shadowColor: '#000',
    shadowOpacity: 0.15,
    shadowOffset: { width: 0, height: -3 },
    shadowRadius: 6,
    height: Platform.OS === 'ios' ? 80 : 70,
    paddingBottom: Platform.OS === 'ios' ? 20 : 10,
    paddingTop: 6,
  } as ViewStyle,

  tabBarItemStyle: {
    alignItems: 'center',
    justifyContent: 'center',
    marginHorizontal: 4,
  } as ViewStyleStrict,

  tabBarLabelStyle: {
    fontSize: 12,
    fontWeight: '600',
    marginTop: 4,
  } as TextStyle,
};

