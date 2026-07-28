/**
 * Métadonnées SEO des pages publiques LIRIE (périmètre verrouillé : 10 routes).
 * Toute extension nécessite une validation séparée.
 */

import { buildFaqPageStructuredData } from './publicFaq';

export const SEO_BASE_URL = 'https://www.lirie.ch';

export const SEO_DEFAULT_IMAGE = '/logo-lirie.png';

/** Emails publics autorisés dans le HTML pré-rendu. */
export const SEO_EMAIL_ALLOWLIST = Object.freeze([
  'info@lirie.ch',
  'privacy@lirie.ch',
]);

/** Routes publiques indexables (sans slash final). */
export const PUBLIC_SEO_PATHS = Object.freeze([
  '/',
  '/deplacez-vous',
  '/conduire',
  '/professionnel',
  '/a-propos',
  '/aide',
  '/contact',
  '/privacy',
  '/conditions',
  '/mentions-legales',
]);

const ORGANIZATION = {
  '@type': 'Organization',
  '@id': `${SEO_BASE_URL}/#organization`,
  name: 'LIRIE',
  url: `${SEO_BASE_URL}/`,
  logo: {
    '@type': 'ImageObject',
    url: `${SEO_BASE_URL}/logo-lirie.png`,
  },
  description:
    'Plateforme suisse de coordination des transports entre patients, établissements de santé et entreprises de transport.',
  email: 'info@lirie.ch',
  telephone: '+41225520302',
  areaServed: [
    { '@type': 'AdministrativeArea', name: 'Genève' },
    { '@type': 'AdministrativeArea', name: 'Suisse romande' },
  ],
  foundingLocation: {
    '@type': 'Place',
    name: 'Genève, Suisse',
  },
};

const WEBSITE = {
  '@type': 'WebSite',
  '@id': `${SEO_BASE_URL}/#website`,
  url: `${SEO_BASE_URL}/`,
  name: 'LIRIE',
  inLanguage: 'fr-CH',
  publisher: { '@id': `${SEO_BASE_URL}/#organization` },
};

const SOFTWARE_APPLICATION = {
  '@type': 'SoftwareApplication',
  '@id': `${SEO_BASE_URL}/#software`,
  name: 'LIRIE',
  applicationCategory: 'BusinessApplication',
  description:
    'Plateforme de coordination, de suivi et de traçabilité des transports. LIRIE n’exécute pas elle-même les prestations de transport.',
  provider: { '@id': `${SEO_BASE_URL}/#organization` },
};

/**
 * @param {string} path
 * @param {{ includeSoftwareApplication?: boolean }} [options]
 */
export function buildPublicStructuredData(path, options = {}) {
  const canonicalPath = path === '/' ? '/' : path.replace(/\/$/, '') || '/';
  const pageUrl = canonicalPath === '/' ? `${SEO_BASE_URL}/` : `${SEO_BASE_URL}${canonicalPath}`;
  const graph = [
    ORGANIZATION,
    WEBSITE,
    {
      '@type': 'WebPage',
      '@id': `${pageUrl}#webpage`,
      url: pageUrl,
      name: PUBLIC_SEO_BY_PATH[canonicalPath]?.title,
      isPartOf: { '@id': `${SEO_BASE_URL}/#website` },
      about: { '@id': `${SEO_BASE_URL}/#organization` },
      inLanguage: 'fr-CH',
    },
  ];
  if (options.includeSoftwareApplication) {
    graph.push(SOFTWARE_APPLICATION);
  }
  if (options.includeFaqPage) {
    graph.push(buildFaqPageStructuredData());
  }
  return {
    '@context': 'https://schema.org',
    '@graph': graph,
  };
}

/** @type {Record<string, { title: string, description: string, includeSoftwareApplication?: boolean }>} */
export const PUBLIC_SEO_BY_PATH = Object.freeze({
  '/': {
    title: 'LIRIE | Coordination des transports de santé en Suisse romande',
    description:
      'LIRIE centralise les demandes de transport entre patients, établissements de santé et entreprises partenaires en Suisse romande, avec suivi et traçabilité des missions.',
    includeSoftwareApplication: true,
  },
  '/deplacez-vous': {
    title: 'Transport médical et accompagné en Suisse romande | LIRIE',
    description:
      'Organisez un transport médical non urgent ou accompagné en Suisse romande : demande guidée, partenaires habilités et suivi pour les personnes autorisées.',
  },
  '/conduire': {
    title: 'Entreprises de transport adapté et médical | LIRIE',
    description:
      'Rejoignez LIRIE comme entreprise de transport partenaire : missions claires, coordination avec les établissements et suivi des courses.',
    includeSoftwareApplication: true,
  },
  '/professionnel': {
    title: 'Gestion des transports pour EMS, cliniques et institutions | LIRIE',
    description:
      'Centralisez et suivez les transports de votre établissement de santé avec LIRIE : un point d’entrée, traçabilité et collaboration avec vos transporteurs.',
    includeSoftwareApplication: true,
  },
  '/a-propos': {
    title: 'À propos de LIRIE | Plateforme suisse de coordination',
    description:
      'LIRIE est une plateforme suisse de coordination des transports basée à Genève. Elle fournit l’outil de coordination, sans exécuter elle-même les prestations.',
  },
  '/aide': {
    title: 'Questions fréquentes sur les transports et LIRIE',
    description:
      'Réponses aux questions fréquentes sur LIRIE : réservation, suivi, institutions, entreprises partenaires et transports médicaux non urgents.',
    includeFaqPage: true,
  },
  '/contact': {
    title: 'Contacter LIRIE à Genève',
    description:
      'Contactez LIRIE à Genève pour une question institutionnelle, un partenariat transport, une démonstration ou un support.',
  },
  '/privacy': {
    title: 'Confidentialité | LIRIE',
    description:
      'Politique de confidentialité de LIRIE : protection des données personnelles et pratiques de la plateforme de coordination des transports.',
  },
  '/conditions': {
    title: 'Conditions d’utilisation | LIRIE',
    description:
      'Conditions d’utilisation de la plateforme LIRIE pour patients, établissements de santé et entreprises de transport partenaires.',
  },
  '/mentions-legales': {
    title: 'Mentions légales | LIRIE',
    description:
      'Mentions légales de LIRIE, plateforme suisse de coordination des transports basée à Genève.',
  },
});

/**
 * Normalise un pathname React Router vers une clé SEO (sans slash final, sauf `/`).
 * @param {string} pathname
 * @returns {string | null}
 */
export function normalizePublicSeoPath(pathname) {
  if (!pathname || typeof pathname !== 'string') return null;
  const raw = pathname.split('?')[0].split('#')[0];
  if (raw === '/' || raw === '') return '/';
  const trimmed = raw.replace(/\/+$/, '');
  return PUBLIC_SEO_BY_PATH[trimmed] ? trimmed : null;
}

/**
 * @param {string} pathname
 * @returns {{ path: string, title: string, description: string, canonicalUrl: string, structuredData: object } | null}
 */
export function getPublicSeoForPath(pathname) {
  const path = normalizePublicSeoPath(pathname);
  if (!path) return null;
  const entry = PUBLIC_SEO_BY_PATH[path];
  const canonicalUrl = path === '/' ? `${SEO_BASE_URL}/` : `${SEO_BASE_URL}${path}`;
  return {
    path,
    title: entry.title,
    description: entry.description,
    canonicalUrl,
    image: SEO_DEFAULT_IMAGE,
    structuredData: buildPublicStructuredData(path, {
      includeSoftwareApplication: Boolean(entry.includeSoftwareApplication),
      includeFaqPage: Boolean(entry.includeFaqPage),
    }),
  };
}

export function listPublicSeoEntries() {
  return PUBLIC_SEO_PATHS.map((path) => getPublicSeoForPath(path));
}
