// https://docs.expo.dev/guides/using-eslint/
/* eslint-env node */
const path = require('node:path');

module.exports = {
  root: true,
  extends: 'expo',
  ignorePatterns: ['/dist/*'],
  settings: {
    'import/resolver': {
      typescript: {
        project: path.join(__dirname, 'tsconfig.json'),
      },
      node: {
        paths: [path.join(__dirname)],
        extensions: ['.js', '.jsx', '.ts', '.tsx'],
      },
    },
  },
  overrides: [
    {
      files: ['scripts/**/*.js'],
      env: {
        node: true,
      },
    },
  ],
};
