const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const typescript = require(path.join(__dirname, "../frontend/node_modules/typescript"));

const sourcePath = path.join(__dirname, "../frontend/lib/formatting.ts");
const source = fs.readFileSync(sourcePath, "utf8");
const { outputText } = typescript.transpileModule(source, {
  compilerOptions: { module: typescript.ModuleKind.CommonJS, target: typescript.ScriptTarget.ES2020 },
});
const compiledModule = { exports: {} };
vm.runInNewContext(outputText, { exports: compiledModule.exports, module: compiledModule });
const { formatCompactValue, formatValue, missingValue } = compiledModule.exports;
const brl = (amount) => `R$\u00a0${amount}`;

assert.equal(formatCompactValue(7_490_000_000, "money", 1), `${brl("7,5")} bi`);
assert.equal(formatCompactValue(-1_490_000, "money", 1), `−${brl("1,5")} mi`);
assert.equal(formatCompactValue(8_500, "integer"), "8,5 mil");
assert.equal(formatCompactValue(1_000_000, "number"), "1 mi");
assert.equal(formatCompactValue(0.04, "money"), brl("0,04"));
assert.equal(formatCompactValue(0, "money"), brl("0,00"));
assert.equal(formatCompactValue(null, "money"), missingValue);
assert.equal(formatCompactValue(Number.NaN, "number"), missingValue);
assert.equal(formatValue(7.49, "money", 2), brl("7,49"));
assert.equal(formatCompactValue(1.2345, "ratio"), "1,23");

process.stdout.write("compact formatting checks passed\n");
