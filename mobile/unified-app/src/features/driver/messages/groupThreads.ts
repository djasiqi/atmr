import type { MessageHubSection, MessageHubThread } from "./types";

export function groupThreadsBySection(threads: MessageHubThread[] | undefined) {
  const sections: Record<MessageHubSection, MessageHubThread[]> = {
    mission_active: [],
    urgent: [],
    team: [],
    dispatch: [],
    colleagues: [],
    support: [],
    archives: [],
  };
  (threads ?? []).forEach((thread) => {
    let key = (thread.section as MessageHubSection) || "dispatch";
    if (key === "company" || key === "groups") {
      key = key === "company" ? "dispatch" : "team";
    }
    if (key in sections) {
      sections[key as MessageHubSection].push(thread);
    } else if (key === "dispatch") {
      sections.dispatch.push(thread);
    } else {
      sections.archives.push(thread);
    }
  });
  return sections;
}
