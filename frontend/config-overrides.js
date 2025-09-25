// config-overrides.js

const SML_EXCLUDE = /node_modules[\\/](svg-engine)/; // adapte le nom du paquet si besoin

module.exports = function override(config, env) {
  const isSmlRule = (rule) => {
    const inUse =
      Array.isArray(rule?.use) &&
      rule.use.some((u) => {
        const loader = typeof u === "string" ? u : u?.loader || "";
        return loader.includes("source-map-loader");
      });
    const inLoader =
      typeof rule?.loader === "string" &&
      rule.loader.includes("source-map-loader");
    return rule?.enforce === "pre" && (inUse || inLoader);
  };

  const appendExclude = (rule, pattern) => {
    if (!rule.exclude) rule.exclude = [pattern];
    else if (Array.isArray(rule.exclude)) rule.exclude.push(pattern);
    else rule.exclude = [rule.exclude, pattern]; // convertit RegExp/Fn/String -> Array
  };

  const visit = (rule) => {
    if (!rule) return;
    if (isSmlRule(rule)) appendExclude(rule, SML_EXCLUDE);
    if (Array.isArray(rule.oneOf)) rule.oneOf.forEach(visit);
    if (Array.isArray(rule.rules)) rule.rules.forEach(visit);
  };

  (config.module?.rules || []).forEach(visit);

  // Ignore “Failed to parse source map” (optionnel)
  config.ignoreWarnings = [
    ...(config.ignoreWarnings || []),
    /Failed to parse source map/,
  ];

  // (optionnel) couper complètement les sourcemaps en dev
  // if (env === "development") config.devtool = false;

  return config;
};

// ❌ NE PAS exporter module.exports.devServer : ça peut casser le démarrage
