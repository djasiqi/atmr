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
  ],
};
