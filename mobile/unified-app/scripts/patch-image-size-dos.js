/**
 * Patch local image-size@1.2.1 contre DoS ICNS / JXL / HEIF (boxes size=0).
 * Le dépôt upstream est archivé ; pas de release npm corrigée au-delà de 1.2.1/2.0.2.
 */
const fs = require("fs");
const path = require("path");

const root = path.join(
  __dirname,
  "..",
  "node_modules",
  "image-size",
  "dist",
  "types"
);

function patchFile(name, transform) {
  const file = path.join(root, name);
  const before = fs.readFileSync(file, "utf8");
  const after = transform(before);
  if (after === before) {
    console.log(`skip ${name} (déjà patché ou motif introuvable)`);
    return false;
  }
  fs.writeFileSync(file, after);
  console.log(`patched ${name}`);
  return true;
}

patchFile("icns.js", (src) => {
  if (src.includes("Invalid ICNS, bad entry length")) return src;
  let out = src.replace(
    "        imageOffset += imageHeader[1];\n        if (imageOffset === fileLength)",
    [
      "        if (!imageHeader[1] || imageHeader[1] < 8) {",
      "            throw new TypeError('Invalid ICNS, bad entry length');",
      "        }",
      "        imageOffset += imageHeader[1];",
      "        if (imageOffset === fileLength)",
    ].join("\n")
  );
  out = out.replace(
    "            imageSize = getImageSize(imageHeader[0]);\n            imageOffset += imageHeader[1];\n            result.images.push(imageSize);",
    [
      "            imageSize = getImageSize(imageHeader[0]);",
      "            if (!imageHeader[1] || imageHeader[1] < 8) {",
      "                break;",
      "            }",
      "            imageOffset += imageHeader[1];",
      "            result.images.push(imageSize);",
    ].join("\n")
  );
  return out;
});

patchFile("jxl.js", (src) => {
  if (src.includes("jxlpBox.size <= 8")) return src;
  return src.replace(
    [
      "        if (!jxlpBox)",
      "            break;",
      "        partialStreams.push(input.slice(jxlpBox.offset + 12, jxlpBox.offset + jxlpBox.size));",
      "        offset = jxlpBox.offset + jxlpBox.size;",
    ].join("\n"),
    [
      "        if (!jxlpBox)",
      "            break;",
      "        // size <= 8 cannot contain a jxlp payload and would not advance (DoS)",
      "        if (!jxlpBox.size || jxlpBox.size <= 8)",
      "            break;",
      "        partialStreams.push(input.slice(jxlpBox.offset + 12, jxlpBox.offset + jxlpBox.size));",
      "        offset = jxlpBox.offset + jxlpBox.size;",
    ].join("\n")
  );
});

patchFile("utils.js", (src) => {
  if (src.includes("boxSize < 8")) return src;
  return src.replace(
    [
      "function readBox(input, offset) {",
      "    if (input.length - offset < 4)",
      "        return;",
      "    const boxSize = (0, exports.readUInt32BE)(input, offset);",
      "    if (input.length - offset < boxSize)",
      "        return;",
    ].join("\n"),
    [
      "function readBox(input, offset) {",
      "    if (input.length - offset < 8)",
      "        return;",
      "    const boxSize = (0, exports.readUInt32BE)(input, offset);",
      "    // size 0 / undersized boxes must not be returned (DoS / invalid containers)",
      "    if (!boxSize || boxSize < 8)",
      "        return;",
      "    if (input.length - offset < boxSize)",
      "        return;",
    ].join("\n")
  );
});
