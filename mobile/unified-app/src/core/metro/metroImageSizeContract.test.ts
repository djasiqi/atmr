/**
 * Contrat Metro 0.83.3 + image-size-next : imageSize reçoit les octets du fichier.
 * L'asset expo-router/assets/pkg.png a fait échouer le bundle EAS release.
 */
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";

const nodeRequire = createRequire(__filename);

describe("contrat Metro image-size v2", () => {
  it("mesure pkg.png en 48x48 via getAssetData", async () => {
    const assetsPath = path.join(process.cwd(), "node_modules/metro/src/Assets.js");
    const source = fs.readFileSync(assetsPath, "utf8");
    expect(source).toContain(
      "(0, _imageSize.default)(_fs.default.readFileSync(assetInfo.files[0]))",
    );
    expect(source).not.toContain('assetInfo.files[0].includes(".zip/")');

    const { getAssetData } = nodeRequire(assetsPath) as {
      getAssetData: (
        assetPath: string,
        localPath: string,
        plugins: string[],
        platform: string | null,
        publicPath: string,
      ) => Promise<{ width?: number; height?: number; type?: string }>;
    };
    const png = path.join(process.cwd(), "node_modules/expo-router/assets/pkg.png");
    const data = await getAssetData(png, "pkg.png", [], null, "/assets");

    expect(data.width).toBe(48);
    expect(data.height).toBe(48);
    expect(data.type).toBe("png");
  });
});
