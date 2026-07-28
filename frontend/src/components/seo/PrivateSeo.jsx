import React from 'react';
import { Helmet } from 'react-helmet-async';

/**
 * Meta robots pour les routes non indexables (fail-closed côté React).
 * Le shell index.html porte aussi noindex (SEO-01B).
 */
export default function PrivateSeo() {
  return (
    <Helmet>
      <meta name="robots" content="noindex, nofollow" />
    </Helmet>
  );
}
