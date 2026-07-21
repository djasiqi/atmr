import React from 'react';

/**
 * Univers Care — soin, sérénité, accompagnement.
 * Composition éditoriale : espace, aplats menthe, traits fins.
 */
export default function Care({ className, title, ...rest }) {
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
      {/* Fond doux */}
      <rect width="480" height="360" rx="24" fill="#F7FBFA" />
      <circle cx="380" cy="70" r="90" fill="#E0F2F1" opacity="0.85" />
      <circle cx="90" cy="300" r="70" fill="#E8F5F3" opacity="0.9" />

      {/* Halo central */}
      <ellipse cx="240" cy="190" rx="120" ry="88" fill="#FFFFFF" opacity="0.7" />
      <ellipse cx="240" cy="190" rx="88" ry="64" fill="#E0F2F1" opacity="0.55" />

      {/* Forme « soin » abstraite — cercle + arc protecteur */}
      <circle cx="240" cy="178" r="36" stroke="#00796B" strokeWidth="1.5" fill="#FFFFFF" />
      <circle cx="240" cy="178" r="14" fill="#00796B" opacity="0.18" />
      <circle cx="240" cy="178" r="6" fill="#00796B" />

      <path
        d="M170 210 C190 250, 290 250, 310 210"
        stroke="#00796B"
        strokeWidth="1.75"
        strokeLinecap="round"
        opacity="0.55"
      />
      <path
        d="M185 200 C200 230, 280 230, 295 200"
        stroke="#26A69A"
        strokeWidth="1.25"
        strokeLinecap="round"
        opacity="0.45"
      />

      {/* Points de respiration */}
      <circle cx="152" cy="120" r="3" fill="#00796B" opacity="0.35" />
      <circle cx="328" cy="130" r="2.5" fill="#00796B" opacity="0.3" />
      <circle cx="300" cy="260" r="2" fill="#00796B" opacity="0.25" />

      {/* Trait horizon discret */}
      <line x1="120" y1="288" x2="360" y2="288" stroke="#91A3A0" strokeWidth="1" opacity="0.35" />
    </svg>
  );
}
