import React from 'react';

/**
 * Univers Trust — cadre, clarté, institution.
 */
export default function Trust({ className, title, ...rest }) {
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
      <rect x="48" y="48" width="384" height="264" rx="20" fill="#FFFFFF" stroke="#E0F2F1" strokeWidth="1.5" />

      {/* Cadre intérieur */}
      <rect x="88" y="88" width="304" height="184" rx="14" fill="#E8F5F3" opacity="0.55" />

      {/* Colonne / stabilité */}
      <rect x="210" y="120" width="60" height="120" rx="8" fill="#FFFFFF" stroke="#00796B" strokeWidth="1.5" opacity="0.9" />
      <line x1="222" y1="148" x2="258" y2="148" stroke="#00796B" strokeWidth="1.25" opacity="0.4" />
      <line x1="222" y1="168" x2="258" y2="168" stroke="#00796B" strokeWidth="1.25" opacity="0.35" />
      <line x1="222" y1="188" x2="258" y2="188" stroke="#00796B" strokeWidth="1.25" opacity="0.3" />

      {/* Bouclier abstrait */}
      <path
        d="M240 112 C255 118, 268 122, 268 140 C268 162, 252 178, 240 186 C228 178, 212 162, 212 140 C212 122, 225 118, 240 112 Z"
        fill="#00796B"
        opacity="0.12"
      />
      <path
        d="M240 124 C250 128, 258 131, 258 142 C258 156, 248 167, 240 172 C232 167, 222 156, 222 142 C222 131, 230 128, 240 124 Z"
        stroke="#00796B"
        strokeWidth="1.5"
        fill="none"
        opacity="0.65"
      />

      {/* Points d’équilibre */}
      <circle cx="120" cy="160" r="3" fill="#00796B" opacity="0.3" />
      <circle cx="360" cy="160" r="3" fill="#00796B" opacity="0.3" />
      <line x1="128" y1="160" x2="200" y2="160" stroke="#91A3A0" strokeWidth="1" opacity="0.3" />
      <line x1="280" y1="160" x2="352" y2="160" stroke="#91A3A0" strokeWidth="1" opacity="0.3" />
    </svg>
  );
}
