/* Genera icon.png e icon.ico desde el favicon SVG de la web. */
const fs = require("fs");
const path = require("path");
const sharp = require("sharp");
const pngToIco = require("png-to-ico").default;

const desktop = path.resolve(__dirname, "..");
const build = path.join(desktop, "build");
const svg = path.resolve(desktop, "..", "web", "assets", "favicon.svg");

async function main() {
  fs.mkdirSync(build, { recursive: true });
  const pngPath = path.join(build, "icon.png");
  const icoPath = path.join(build, "icon.ico");

  await sharp(svg, { density: 384 })
    .resize(256, 256, { fit: "contain", background: { r: 6, g: 10, b: 18, alpha: 0 } })
    .png()
    .toFile(pngPath);

  const ico = await pngToIco(pngPath);
  fs.writeFileSync(icoPath, ico);
  console.log(`Iconos generados:\n  ${pngPath}\n  ${icoPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
