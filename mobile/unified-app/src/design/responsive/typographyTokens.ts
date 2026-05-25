/**
 * Static typography scale used by the layout debt migration.
 *
 * The values intentionally mirror existing UI sizes first. Follow-up design work can consolidate
 * aliases by semantic role (`body`, `caption`, `title`) without reintroducing literal font sizes.
 */
export const FONT_SIZE = {
  px7: 7,
  px8: 8,
  px9: 9,
  px10: 10,
  px11: 11,
  px11_5: 11.5,
  px12: 12,
  px12_5: 12.5,
  px13: 13,
  px13_5: 13.5,
  px14: 14,
  px14_5: 14.5,
  px15: 15,
  px15_5: 15.5,
  px16: 16,
  px17: 17,
  px18: 18,
  px20: 20,
  px21: 21,
  px22: 22,
  px23: 23,
  px24: 24,
  px26: 26,
  px28: 28,
  px29: 29,
  px30: 30,
} as const;
