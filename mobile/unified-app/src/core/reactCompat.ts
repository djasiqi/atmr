// eslint-disable-next-line @typescript-eslint/no-require-imports
const ReactRuntime: any = require("react");

export type PropsWithChildren<P = object> = P & { children?: any };

export const useEffect: typeof import("react").useEffect = (...args: any[]) =>
  ReactRuntime.useEffect(...args);
