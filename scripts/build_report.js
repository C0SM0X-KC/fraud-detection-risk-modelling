/**
 * Builds the Phase-3 report as a .docx:  node scripts/build_report.js
 */

const fs = require("fs");
const path = require("path");
const {
  AlignmentType, Document, Footer, LevelFormat, PageNumber, Packer, Paragraph, TextRun,
} = require("docx");

const R = require("./report_frontmatter.js");
const { chapter1, chapter2 } = require("./report_chapters.js");
const { chapter3, chapter4, chapter5, backMatter } = require("./report_results.js");

const body = [
  ...R.titlePage,
  ...R.certificate,
  ...R.acknowledgement,
  ...R.abstract,
  ...R.frontTables,
  ...chapter1,
  ...chapter2,
  ...chapter3,
  ...chapter4,
  ...chapter5,
  ...backMatter,
];

const doc = new Document({
  creator: "Nandurkar Shreyas Vivek",
  title: "Financial Transaction Fraud Detection and Risk Modelling",
  description: "Phase-3 minor project report",
  // Word recalculates the table of contents on open rather than showing an
  // empty placeholder.
  features: { updateFields: true },
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{
        level: 0,
        format: LevelFormat.BULLET,
        text: "•",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 520, hanging: 260 } } },
      }],
    }],
  },
  styles: {
    default: {
      document: { run: { font: "Times New Roman", size: 22 } },
    },
  },
  sections: [{
    properties: {
      page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } },
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({
            children: [PageNumber.CURRENT], size: 18, font: "Times New Roman",
          })],
        })],
      }),
    },
    children: body,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.mkdirSync(path.dirname(R.OUT), { recursive: true });
  fs.writeFileSync(R.OUT, buf);
  console.log(`\nwrote ${path.relative(R.ROOT, R.OUT)} (${(buf.length / 1024).toFixed(0)} KB, ${body.length} blocks)`);
}).catch((err) => {
  console.error("failed to build report:", err);
  process.exit(1);
});
