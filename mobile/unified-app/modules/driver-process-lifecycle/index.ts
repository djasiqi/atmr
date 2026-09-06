export type DriverStartedActivityCountEvent = {
  count: number;
};

export type DriverProcessLifecycleNative = {
  getStartedActivityCount?: () => number;
  addListener?: (
    event: "onStartedActivityCountChanged",
    listener: (event: DriverStartedActivityCountEvent) => void
  ) => { remove: () => void };
};
