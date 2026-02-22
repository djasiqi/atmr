import Constants from "expo-constants";

const isDevelopment =
  __DEV__ || Constants.expoConfig?.extra?.environment !== "production";

type LogLevel = "debug" | "info" | "warn" | "error";

const LEVEL_TAGS: Record<LogLevel, string> = {
  debug: "DBG",
  info: "INF",
  warn: "WRN",
  error: "ERR",
};

function ts(): string {
  return new Date().toISOString().slice(11, 23);
}

function fmt(level: LogLevel, module: string, msg: string): string {
  return `${ts()} ${LEVEL_TAGS[level]} [${module}] ${msg}`;
}

function createModuleLogger(module: string) {
  return {
    debug(msg: string, data?: Record<string, unknown>) {
      if (!isDevelopment) return;
      if (data) console.debug(fmt("debug", module, msg), data);
      else console.debug(fmt("debug", module, msg));
    },
    info(msg: string, data?: Record<string, unknown>) {
      if (!isDevelopment) return;
      if (data) console.log(fmt("info", module, msg), data);
      else console.log(fmt("info", module, msg));
    },
    warn(msg: string, data?: Record<string, unknown>) {
      if (data) console.warn(fmt("warn", module, msg), data);
      else console.warn(fmt("warn", module, msg));
    },
    error(msg: string, data?: Record<string, unknown>) {
      if (data) console.error(fmt("error", module, msg), data);
      else console.error(fmt("error", module, msg));
    },
    success(msg: string, data?: Record<string, unknown>) {
      if (!isDevelopment) return;
      if (data) console.log(fmt("info", module, `OK ${msg}`), data);
      else console.log(fmt("info", module, `OK ${msg}`));
    },
  };
}

export type ModuleLogger = ReturnType<typeof createModuleLogger>;

export function getLogger(module: string): ModuleLogger {
  return createModuleLogger(module);
}

export const logger = {
  log: (...args: unknown[]) => {
    if (isDevelopment) console.log(...args);
  },
  info: (...args: unknown[]) => {
    if (isDevelopment) console.info(...args);
  },
  warn: (...args: unknown[]) => {
    console.warn(...args);
  },
  error: (...args: unknown[]) => {
    console.error(...args);
  },
  debug: (...args: unknown[]) => {
    if (isDevelopment) console.debug(...args);
  },
};

export const isDev = isDevelopment;
