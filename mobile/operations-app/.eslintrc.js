// driver-app/.eslintrc.js
module.exports = {
  root: true,
  extends: ['expo', 'prettier'],
  plugins: ['prettier'],
  rules: {
    'prettier/prettier': 'warn',
    'react/prop-types': 'off',
    'no-unused-vars': ['warn', { argsIgnorePattern: '^_' }],
  },
  overrides: [
    {
      // Prévention multi-tenant : en contexte enterprise, ne pas utiliser api (token driver)
      files: ['app/(enterprise)/**/*.tsx', 'app/(enterprise)/**/*.ts'],
      rules: {
        'no-restricted-imports': [
          'error',
          {
            paths: [
              {
                name: '@/services/api',
                importNames: ['default'],
                message:
                  'En contexte enterprise, utiliser enterpriseStandardApi ou enterpriseApi. api envoie le token driver.',
              },
            ],
          },
        ],
      },
    },
    {
      // Phase 5 — boundary dispatch mobile (ADR 013) : pas d’API dispatch hors périmètre enterprise
      files: [
        'app/_layout.tsx',
        'app/index.tsx',
        'app/quick-action.tsx',
        'app/(tabs)/**/*.{ts,tsx}',
        'app/(dashboard)/**/*.{ts,tsx}',
        'app/(auth)/**/*.{ts,tsx}',
        'app/(enterprise-auth)/**/*.{ts,tsx}',
      ],
      rules: {
        'no-restricted-imports': [
          'error',
          {
            paths: [
              {
                name: '@/services/enterpriseDispatch',
                message:
                  'API dispatch réservée aux écrans (enterprise) et modules enterprise (voir docs/adr/013).',
              },
            ],
            patterns: [
              {
                group: ['**/services/enterpriseDispatch'],
                message:
                  'Import dispatch : utiliser uniquement depuis app/(enterprise)/, components/enterprise/, hooks métier dispatch (ADR 013).',
              },
            ],
          },
        ],
      },
    },
  ],
};
