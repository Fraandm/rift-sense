/* Guarda la raiz del proyecto para que el .exe portable la encuentre. */
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..", "..");
const output = path.resolve(__dirname, "..", "root.json");
fs.writeFileSync(output, JSON.stringify({ root }, null, 2), "utf8");
console.log(`Raiz del proyecto: ${root}`);
console.log(`Config: ${output}`);
