import React from 'react';
import { Helmet } from 'react-helmet-async';
import { SEO_BASE_URL, SEO_DEFAULT_IMAGE } from '../../config/publicSeo';

/**
 * Balises SEO pour une page publique indexable.
 */
export default function PublicSeo({
  title,
  description,
  path,
  canonicalUrl,
  image = SEO_DEFAULT_IMAGE,
  structuredData,
}) {
  const resolvedCanonical =
    canonicalUrl ||
    (path === '/' ? `${SEO_BASE_URL}/` : `${SEO_BASE_URL}${path}`);
  const imageUrl = image.startsWith('http') ? image : `${SEO_BASE_URL}${image}`;

  return (
    <Helmet prioritizeSeoTags>
      <html lang="fr-CH" />
      <title>{title}</title>
      <meta name="description" content={description} />
      <link rel="canonical" href={resolvedCanonical} />
      <meta
        name="robots"
        content="index, follow, max-image-preview:large"
      />

      <meta property="og:type" content="website" />
      <meta property="og:locale" content="fr_CH" />
      <meta property="og:site_name" content="LIRIE" />
      <meta property="og:title" content={title} />
      <meta property="og:description" content={description} />
      <meta property="og:url" content={resolvedCanonical} />
      <meta property="og:image" content={imageUrl} />

      <meta name="twitter:card" content="summary_large_image" />
      <meta name="twitter:title" content={title} />
      <meta name="twitter:description" content={description} />
      <meta name="twitter:image" content={imageUrl} />

      {structuredData ? (
        <script type="application/ld+json">
          {JSON.stringify(structuredData)}
        </script>
      ) : null}
    </Helmet>
  );
}
