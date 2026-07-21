import React from 'react';

/**
 * Univers Connection — coordination, liens entre acteurs.
 */
export default function Connection({ className, title, ...rest }) {
  const labelled = Boolean(title);
  return (
    <svg
      className={className}
      viewBox="0 0 480 360"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      role={labelled ? 'img' : 'presentation'}
      aria-hidden={labelled ? undefined : true}
      aria-label={title}
      {...rest}
    >
      {title ? <title>{title}</title> : null}
      <rect width="480" height="360" rx="24" fill="#F7FBFA" />
      <circle cx="240" cy="180" r="110" fill="#E0F2F1" opacity="0.45" />
      <circle cx="240" cy="180" r="70" fill="#FFFFFF" opacity="0.75" />

      {/* Nœuds */}
      <circle cx="240" cy="120" r="16" fill="#FFFFFF" stroke="#00796B" strokeWidth="1.75" />
      <circle cx="240" cy="120" r="5" fill="#00796B" />

      <circle cx="160" cy="210" r="16" fill="#FFFFFF" stroke="#00796B" strokeWidth="1.75" />
      <circle cx="160" cy="210" r="5" fill="#00796B" opacity="0.75" />

      <circle cx="320" cy="210" r="16" fill="#FFFFFF" stroke="#00796B" strokeWidth="1.75" />
      <circle cx="320" cy="210" r="5" fill="#00796B" opacity="0.75" />

      <circle cx="240" cy="260" r="14" fill="#FFFFFF" stroke="#26A69A" strokeWidth="1.5" />
      <circle cx="240" cy="260" r="4" fill="#26A69A" />

      {/* Liens */}
      <line x1="240" y1="136" x2="172" y2="198" stroke="#00796B" strokeWidth="1.5" opacity="0.45" />
      <line x1="240" y1="136" x2="308" y2="198" stroke="#00796B" strokeWidth="1.5" opacity="0.45" />
      <line x1="174" y1="218" x2="228" y2="252" stroke="#00796B" strokeWidth="1.25" opacity="0.35" />
      <line x1="306" y1="218" x2="252" y2="252" stroke="#00796B" strokeWidth="1.25" opacity="0.35" />
      <line x1="176" y1="210" x2="304" y2="210" stroke="#91A3A0" strokeWidth="1" opacity="0.3" strokeDasharray="3 6" />

      {/* Accents périphériques */}
      <circle cx="100" cy="100" r="4" fill="#00796B" opacity="0.25" />
      <circle cx="380" cy="90" r="3" fill="#00796B" opacity="0.2" />
      <circle cx="390" cy="280" r="5" fill="#E0F2F1" />
      <circle cx="90" cy="270" r="6" fill="#E8F5F3" />
    </svg>
  );
}
