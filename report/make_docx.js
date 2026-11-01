// Render report/report_blocks.json (written by report/make_report.py) to a right-to-left Persian Word file.
// Run from the repository root:  node report/make_docx.js
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, HeadingLevel, LevelFormat, TabStopType, BorderStyle, Footer, PageNumber, TableLayoutType, VerticalAlign,
} = require('docx');

const blocks = JSON.parse(fs.readFileSync('report/report_blocks.json', 'utf8'));
const OUT = process.argv[2] || 'report/MobileNetV4_UIB_CIFAR10_Report.docx';

const FA = 'Tahoma';          // Persian (complex-script) font, present on every Windows / Office install
const LAT = 'Calibri';
const MONO = 'Consolas';
const BLUE = '1F4E79';
const PAGE_W = 11906, MARGIN = 1300;          // A4, DXA
const CONTENT_W = PAGE_W - 2 * MARGIN;          // 9306 DXA = 6.46 in
const font = (cs) => ({ ascii: LAT, hAnsi: LAT, cs: cs || FA, eastAsia: LAT });

// ---------------------------------------------------------------- inline markup: **bold**, `code`
function runs(text, opts = {}) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/).filter((s) => s.length);
  return parts.map((p) => {
    if (p.startsWith('**')) {
      return new TextRun({ text: p.slice(2, -2), bold: true, boldComplexScript: true, rightToLeft: true, font: font(), ...opts });
    }
    if (p.startsWith('`')) {
      return new TextRun({ text: p.slice(1, -1), font: { ascii: MONO, hAnsi: MONO, cs: MONO }, size: 18, sizeComplexScript: 18, rightToLeft: false });
    }
    return new TextRun({ text: p, rightToLeft: true, font: font(), ...opts });
  });
}

// justified text stretches badly around long inline code, so those paragraphs are aligned to the start (right) instead
const align = (text) => (text.includes('`') ? AlignmentType.START : AlignmentType.BOTH);
const rtlPara = (text, extra = {}) => new Paragraph({
  bidirectional: true, alignment: align(text), spacing: { after: 120, line: 300 }, children: runs(text), ...extra,
});

// ---------------------------------------------------------------- images (read PNG size from the header)
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), data: b };
}

function image(b) {
  const { w, h, data } = pngSize(b.path);
  const maxIn = Math.min(b.width || 6.3, CONTENT_W / 1440);
  const wPx = Math.round(maxIn * 96);
  const hPx = Math.round(wPx * h / w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, keepNext: true,
      children: [new ImageRun({ type: 'png', data, transformation: { width: wPx, height: hPx },
        altText: { title: b.caption, description: b.caption, name: path.basename(b.path) } })] }),
    new Paragraph({ bidirectional: true, alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: b.caption, rightToLeft: true, italics: true, italicsComplexScript: true, size: 18, sizeComplexScript: 18, color: '404040', font: font() })] }),
  ];
}

// ---------------------------------------------------------------- tables
const border = { style: BorderStyle.SINGLE, size: 4, color: 'A6A6A6' };
const borders = { top: border, bottom: border, left: border, right: border };

function table(b) {
  const n = b.header.length;
  const rel = b.widths || Array(n).fill(1);
  const sum = rel.reduce((a, c) => a + c, 0);
  const widths = rel.map((x) => Math.floor(CONTENT_W * x / sum));
  widths[widths.length - 1] += CONTENT_W - widths.reduce((a, c) => a + c, 0);
  const ltr = !!b.ltr;
  const small = n > 6 ? 16 : 18;
  const cell = (text, i, header) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    borders,
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 50, bottom: 50, left: 80, right: 80 },
    shading: header ? { fill: 'DCE6F1', type: ShadingType.CLEAR, color: 'auto' } : undefined,
    children: [new Paragraph({
      bidirectional: !ltr,
      alignment: AlignmentType.CENTER,
      spacing: { after: 0, line: 260 },
      children: runs(String(text), { size: small, sizeComplexScript: small, ...(header ? { bold: true, boldComplexScript: true } : {}) })
        .map((r) => r),
    })],
  });
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: widths,
    layout: TableLayoutType.FIXED,
    visuallyRightToLeft: !ltr,
    rows: [
      new TableRow({ tableHeader: true, children: b.header.map((h, i) => cell(h, i, true)) }),
      ...b.rows.map((r) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, i, false)) })),
    ],
  });
}

// ---------------------------------------------------------------- build
const children = [];
let figNo = 0;
for (const b of blocks) {
  switch (b.t) {
    case 'title':
      children.push(new Paragraph({ bidirectional: true, alignment: AlignmentType.CENTER, spacing: { before: 600, after: 120 },
        children: [new TextRun({ text: b.text, rightToLeft: true, bold: true, boldComplexScript: true, size: 34, sizeComplexScript: 34, color: BLUE, font: font() })] }));
      break;
    case 'subtitle':
      children.push(new Paragraph({ bidirectional: true, alignment: AlignmentType.CENTER, spacing: { after: 120 },
        children: [new TextRun({ text: b.text, rightToLeft: true, size: 26, sizeComplexScript: 26, color: '2E75B6', font: font() })] }));
      break;
    case 'meta':
      children.push(new Paragraph({ bidirectional: true, alignment: AlignmentType.CENTER, spacing: { after: 360 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: '2E75B6', space: 8 } },
        children: [new TextRun({ text: b.text, rightToLeft: true, size: 18, sizeComplexScript: 18, color: '595959', font: font() })] }));
      break;
    case 'h1':
      children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, bidirectional: true, children: [new TextRun({ text: b.text, rightToLeft: true })] }));
      break;
    case 'h2':
      children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, bidirectional: true, children: [new TextRun({ text: b.text, rightToLeft: true })] }));
      break;
    case 'p':
      children.push(rtlPara(b.text));
      break;
    case 'bullets':
      for (const it of b.items) {
        // Bullet drawn in the paragraph itself: Word numbering labels stay on the left in LibreOffice for RTL paragraphs.
        children.push(new Paragraph({ bidirectional: true, alignment: align(it),
          indent: { right: 420, hanging: 260 }, tabStops: [{ type: TabStopType.RIGHT, position: 420 }],
          spacing: { after: 80, line: 300 },
          children: [new TextRun({ text: '\u2022', rightToLeft: true, color: '2E75B6', font: font() }), new TextRun({ text: '\t', rightToLeft: true }), ...runs(it)] }));
      }
      break;
    case 'refs':
      for (const it of b.items) {
        children.push(new Paragraph({ numbering: { reference: 'refs', level: 0 }, spacing: { after: 60 },
          children: [new TextRun({ text: it, size: 19, font: font(LAT) })] }));
      }
      break;
    case 'table':
      children.push(table(b));
      children.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
      break;
    case 'img':
      figNo += 1;
      children.push(...image(b));
      break;
    case 'code':
    case 'eq': {
      const lines = b.t === 'code' ? b.lines : [b.text];
      lines.forEach((ln, i) => children.push(new Paragraph({
        alignment: b.t === 'eq' ? AlignmentType.CENTER : AlignmentType.LEFT,
        shading: b.t === 'code' ? { fill: 'F2F2F2', type: ShadingType.CLEAR, color: 'auto' } : undefined,
        spacing: { after: i === lines.length - 1 ? 160 : 0, before: i === 0 ? 80 : 0 },
        children: [new TextRun({ text: ln, font: b.t === 'code' ? { ascii: MONO, hAnsi: MONO, cs: MONO } : { ascii: 'Cambria Math', hAnsi: 'Cambria Math', cs: 'Cambria Math' },
          size: b.t === 'code' ? 17 : 24 })],
      })));
      break;
    }
    default:
      throw new Error('unknown block ' + b.t);
  }
}

const heading = (id, name, size, color, before) => ({
  id, name, basedOn: 'Normal', next: 'Normal', quickFormat: true,
  run: { size, sizeComplexScript: size, bold: true, boldComplexScript: true, color, font: font() },
  paragraph: { spacing: { before, after: 120 }, keepNext: true, outlineLevel: id === 'Heading1' ? 0 : 1 },
});

const doc = new Document({
  creator: 'MobileNetV4 UIB report',
  title: 'MobileNetV4 UIB (ConvNeXt) on CIFAR-10: BatchNorm vs LayerNorm',
  styles: {
    default: { document: { run: { font: font(), size: 21, sizeComplexScript: 21 } } },
    paragraphStyles: [
      heading('Heading1', 'Heading 1', 28, BLUE, 320),
      heading('Heading2', 'Heading 2', 24, '2E75B6', 200),
    ],
  },
  numbering: {
    config: [
      { reference: 'refs', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '[%1]', alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 560, hanging: 560 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: 1200, bottom: 1200, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: '808080' })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log('wrote', OUT, `(${figNo} figures)`);
});
