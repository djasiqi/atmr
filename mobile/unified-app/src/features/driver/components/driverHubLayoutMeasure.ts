type HubWindowEdges = {
  headerBottom?: number;
  statusTop?: number;
  statusBottom?: number;
  missionTop?: number;
};

let edges: HubWindowEdges = {};

type Measurable = {
  measureInWindow?: (
    callback: (x: number, y: number, width: number, height: number) => void
  ) => void;
};

function logIfComplete(): void {
  const { headerBottom, statusTop, statusBottom, missionTop } = edges;
  if (
    headerBottom == null ||
    statusTop == null ||
    statusBottom == null ||
    missionTop == null
  ) {
    return;
  }
  const gapA = statusTop - headerBottom;
  const gapB = missionTop - statusBottom;
  const visibleGap = missionTop - headerBottom;
  console.log(
    `[driver-shell-layout] headerBottom=${Math.round(headerBottom)} statusTop=${Math.round(statusTop)} statusBottom=${Math.round(statusBottom)} missionTop=${Math.round(missionTop)} GAP_A=${Math.round(gapA)} GAP_B=${Math.round(gapB)} VISIBLE_GAP=${Math.round(visibleGap)}`
  );
}

function record(partial: HubWindowEdges): void {
  edges = { ...edges, ...partial };
  logIfComplete();
}

export function measureDriverHubWindowEdge(
  node: Measurable | null,
  edge: "headerBottom" | "status" | "missionTop"
): void {
  if (!__DEV__) return;
  if (!node || typeof node.measureInWindow !== "function") return;
  node.measureInWindow((_x, y, _width, height) => {
    if (edge === "headerBottom") {
      record({ headerBottom: y + height });
      return;
    }
    if (edge === "status") {
      record({ statusTop: y, statusBottom: y + height });
      return;
    }
    record({ missionTop: y });
  });
}
