/** Met à jour la liste des champs overlay sans allouer si l’appartenance n’a pas changé. */
export function nextSuggestionFields(
  prev: readonly string[],
  fieldKey: string,
  visible: boolean
): readonly string[] {
  const has = prev.includes(fieldKey);
  if (visible === has) return prev;
  return visible ? [...prev, fieldKey] : prev.filter((key) => key !== fieldKey);
}
