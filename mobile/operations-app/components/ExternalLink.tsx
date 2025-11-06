import { Link } from 'expo-router';
import { openBrowserAsync } from 'expo-web-browser';
import { ComponentProps } from 'react';
import { Platform } from 'react-native';

// Dérive le type href à partir des props du composant Link
type HrefType = ComponentProps<typeof Link>['href'];

type Props = Omit<ComponentProps<typeof Link>, 'href'> & { href: HrefType };

export function ExternalLink({ href, ...rest }: Props) {
  // Détermine si le lien est externe (commence par http)
  const isExternal = typeof href === 'string' && /^https?:\/\//.test(href);

  return (
    <Link
      // Sur web, ouvre dans un nouvel onglet si externe
      target={isExternal ? '_blank' : undefined}
      {...rest}
      href={href}
      onPress={async (event) => {
        if (Platform.OS !== 'web' && isExternal) {
          // Empêche le comportement par défaut sur mobile
          event.preventDefault();
          // Ouvre le navigateur interne
          await openBrowserAsync(href as string);
        }
      }}
    />
  );
}
