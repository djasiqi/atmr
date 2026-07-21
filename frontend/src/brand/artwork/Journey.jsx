import React from 'react';

/**
 * Univers Journey — trajet, progression, étapes.
 */
export default function Journey({ className, title, ...rest }) {
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
      <path
        d="M0 220 C80 180, 140 260, 240 210 C340 160, 400 240, 480 200 L480 360 L0 360 Z"
        fill="#E0F2F1"
        opacity="0.55"
      />

      {/* Chemin courbe */}
      <path
        d="M64 250 C140 160, 200 280, 280 170 C340 110, 380 200, 420 150"
        stroke="#00796B"
        strokeWidth="2"
        strokeLinecap="round"
        strokeDasharray="0"
        opacity="0.7"
      />
      <path
        d="M64 250 C140 160, 200 280, 280 170 C340 110, 380 200, 420 150"
        stroke="#26A69A"
        strokeWidth="1"
        strokeLinecap="round"
        opacity="0.35"
        strokeDasharray="4 8"
      />

      {/* Étapes */}
      <circle cx="64" cy="250" r="10" fill="#FFFFFF" stroke="#00796B" strokeWidth="1.75" />
      <circle cx="64" cy="250" r="4" fill="#00796B" />

      <circle cx="180" cy="210" r="10" fill="#FFFFFF" stroke="#00796B" strokeWidth="1.75" />
      <circle cx="180" cy="210" r="4" fill="#00796B" opacity="0.7" />

      <circle cx="280" cy="170" r="12" fill="#FFFFFF" stroke="#00796B" strokeWidth="1.75" />
      <circle cx="280" cy="170" r="5" fill="#00796B" />

      <circle cx="420" cy="150" r="10" fill="#FFFFFF" stroke="#00796B" strokeWidth="1.75" />
      <circle cx="420" cy="150" r="4" fill="#26A69A" />

      {/* Aplat distant */}
      <circle cx="400" cy="60" r="48" fill="#E8F5F3" opacity="0.8" />
      <circle cx="70" cy="80" r="28" fill="#FFFFFF" opacity="0.6" />
    </svg>
  );
}
