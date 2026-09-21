const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const { CAPTURES_DIR } = require('./screenshot');

function readPngDimensions(filePath) {
  const fd = fs.openSync(filePath, 'r');
  const header = Buffer.alloc(24);
  try {
    fs.readSync(fd, header, 0, 24, 0);
  } finally {
    fs.closeSync(fd);
  }
  if (header.toString('ascii', 1, 4) !== 'PNG') {
    throw new Error(`Pas un PNG: ${filePath}`);
  }
  return {
    width: header.readUInt32BE(16),
    height: header.readUInt32BE(20),
  };
}

function inspectCapture(filename) {
  const filePath = path.join(CAPTURES_DIR, path.basename(filename));
  const stat = fs.statSync(filePath);
  const { width, height } = readPngDimensions(filePath);
  const sha256 = crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
  return {
    filename: path.basename(filename),
    path: filePath,
    width,
    height,
    bytes: stat.size,
    sha256,
  };
}

module.exports = {
  inspectCapture,
  readPngDimensions,
};
