#!/usr/bin/env python3
"""
增强版PDF解析器 - 借鉴RAGFlow核心思路

特性：
1. 双通道融合（PyMuPDF + pdfplumber）+ 乱码检测降级
2. KMeans自适应分栏 + 轮廓系数评估（静默警告）
3. 页眉页脚过滤
4. 表格提取增强：
   - 使用 pdfplumber.find_tables() 做整页强表格检测
   - 不在整页范围使用 text/text 策略，避免目录、释义、列表、段落被误判成表格
   - 表格区域独立输出为 [TABLE_START] ... [TABLE_END]
   - 文本块与表格区域重叠时跳过，避免重复
   - 过滤单列碎片表、目录表、释义表、弱结构伪表格
5. 批量处理：支持处理整个目录的 PDF 文件
6. 断点续传：自动跳过已处理的文件
"""

import fitz  # PyMuPDF
import pdfplumber
import re
import unicodedata
import warnings
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import List, Dict, Tuple, Optional
import sys
import time


class EnhancedPDFParser:
    """借鉴RAGFlow核心思路的增强PDF解析器"""

    def __init__(
        self,
        header_margin: float = 80,
        footer_margin: float = 80,
        use_ocr_fallback: bool = False,
        table_overlap_threshold: float = 0.3,
    ):
        """
        Args:
            header_margin: 页眉区域高度（PDF点数）
            footer_margin: 页脚区域高度（PDF点数）
            use_ocr_fallback: 是否启用OCR降级（需要额外安装paddleocr）
            table_overlap_threshold: 文本块与表格区域重叠超过该比例时跳过文本块
        """
        self.header_margin = header_margin
        self.footer_margin = footer_margin
        self.use_ocr_fallback = use_ocr_fallback
        self.table_overlap_threshold = table_overlap_threshold
        self.mean_height = 0

    # ============================================
    # 主入口
    # ============================================

    def parse(self, file_path: Path) -> str:
        """主入口：双通道解析PDF"""
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        print(f"📄 开始解析: {file_path.name}")

        mupdf_doc = fitz.open(str(file_path))
        plumber_pdf = pdfplumber.open(str(file_path))

        all_markdown = []
        total_pages = len(mupdf_doc)

        for page_num in range(total_pages):
            print(f"  处理第 {page_num + 1}/{total_pages} 页...", end="")

            mupdf_page = mupdf_doc[page_num]
            plumber_page = plumber_pdf.pages[page_num]

            chars = self._get_chars(plumber_page)
            is_garbled = self._detect_garbled_text(chars)

            if is_garbled:
                print(" ⚠️ 乱码，降级处理")
                if self.use_ocr_fallback:
                    page_md = self._ocr_fallback(mupdf_page)
                else:
                    page_md = self._extract_text_only(mupdf_page)
            else:
                page_md = self._process_page_normal(mupdf_page, plumber_page)
                print("")

            if page_md.strip():
                all_markdown.append(page_md)

        mupdf_doc.close()
        plumber_pdf.close()

        result = "\n\n---\n\n".join(all_markdown)
        print(f"✅ 解析完成，共提取 {len(result)} 个字符")
        return result

    # ============================================
    # 一、双通道融合 + 乱码检测降级
    # ============================================

    def _get_chars(self, plumber_page) -> List[Dict]:
        """获取字符级数据"""
        try:
            return plumber_page.chars
        except Exception:
            return []

    def _detect_garbled_text(self, chars: List[Dict]) -> bool:
        """
        借鉴RAGFlow的乱码检测
        检测策略：
        1. PUA字符（Private Use Area）
        2. CID映射失败
        3. 字体编码乱码（CJK映射到ASCII）
        """
        if not chars:
            return False

        sample = chars[:200]
        text = "".join([c.get("text", "") for c in sample])

        if not text.strip():
            return False

        garbled_count = 0
        total_count = 0

        for ch in text:
            if ch.isspace():
                continue
            total_count += 1
            if self._is_garbled_char(ch):
                garbled_count += 1

        if total_count > 0 and garbled_count / total_count >= 0.3:
            return True

        if re.search(r"\(cid\s*:\s*\d+\s*\)", text):
            return True

        if self._detect_font_encoding_garbled(chars):
            return True

        return False

    def _is_garbled_char(self, ch: str) -> bool:
        """判断单个字符是否为乱码"""
        cp = ord(ch)

        if 0xE000 <= cp <= 0xF8FF:
            return True
        if 0xF0000 <= cp <= 0xFFFFF:
            return True
        if 0x100000 <= cp <= 0x10FFFF:
            return True

        if cp == 0xFFFD:
            return True

        if cp < 0x20 and ch not in ("\t", "\n", "\r"):
            return True

        if 0x80 <= cp <= 0x9F:
            return True

        cat = unicodedata.category(ch)
        if cat in ("Cn", "Cs"):
            return True

        return False

    def _detect_font_encoding_garbled(self, chars: List[Dict], min_chars: int = 20) -> bool:
        """检测字体编码导致的乱码"""
        if len(chars) < min_chars:
            return False

        subset_font_count = 0
        total_non_space = 0
        ascii_punct_sym = 0
        cjk_like = 0

        for c in chars:
            text = c.get("text", "")
            fontname = c.get("fontname", "")

            if not text or text.isspace():
                continue

            total_non_space += 1

            if re.match(r"^[A-Z0-9]{2,6}\+", fontname or ""):
                subset_font_count += 1

            cp = ord(text[0])

            if (
                0x2E80 <= cp <= 0x9FFF
                or 0xF900 <= cp <= 0xFAFF
                or 0x20000 <= cp <= 0x2FA1F
                or 0xAC00 <= cp <= 0xD7AF
                or 0x3040 <= cp <= 0x30FF
            ):
                cjk_like += 1
            elif (
                0x21 <= cp <= 0x2F
                or 0x3A <= cp <= 0x40
                or 0x5B <= cp <= 0x60
                or 0x7B <= cp <= 0x7E
            ):
                ascii_punct_sym += 1

        if total_non_space < min_chars:
            return False

        subset_ratio = subset_font_count / total_non_space
        if subset_ratio < 0.3:
            return False

        cjk_ratio = cjk_like / total_non_space
        punct_ratio = ascii_punct_sym / total_non_space

        if cjk_ratio < 0.05 and punct_ratio > 0.4:
            return True

        return False

    # ============================================
    # 二、KMeans自适应分栏（静默警告版）
    # ============================================

    def _detect_columns(self, boxes: List[Dict]) -> int:
        """
        借鉴RAGFlow的自适应分栏
        使用KMeans聚类 + 轮廓系数自动确定最优栏数
        """
        if len(boxes) < 2:
            return 1

        x0s = np.array([b["x0"] for b in boxes]).reshape(-1, 1)

        min_x0 = np.min(x0s)
        max_x1 = np.max([b["x1"] for b in boxes])
        page_width = max_x1 - min_x0

        indent_tolerance = page_width * 0.12

        adjusted_x0s = []
        for x in x0s.flatten():
            if abs(x - min_x0) < indent_tolerance:
                adjusted_x0s.append([min_x0])
            else:
                adjusted_x0s.append([x])

        adjusted_x0s = np.array(adjusted_x0s, dtype=float)
        unique_x = len(set(round(x[0], 1) for x in adjusted_x0s))

        if unique_x < 2:
            return 1

        max_try = min(4, unique_x, len(boxes))
        best_k = 1
        best_score = -1

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

            for k in range(1, max_try + 1):
                if k == 1:
                    best_score = 0
                    continue

                if unique_x < k:
                    continue

                try:
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = km.fit_predict(adjusted_x0s)

                    if len(set(labels)) > 1:
                        score = silhouette_score(adjusted_x0s, labels)
                        if score > best_score:
                            best_score = score
                            best_k = k
                except Exception:
                    continue

        if best_k > 1:
            print(f" [{best_k}栏, 轮廓系数:{best_score:.3f}]", end="")

        return best_k

    def _sort_by_reading_order(self, boxes: List[Dict], page_width: float) -> List[Dict]:
        """按阅读顺序排序（处理多栏）"""
        if len(boxes) < 2:
            return boxes

        n_cols = self._detect_columns(boxes)

        if n_cols == 1:
            return sorted(boxes, key=lambda b: (b["top"], b["x0"]))

        x0s = np.array([b["x0"] for b in boxes]).reshape(-1, 1)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            km = KMeans(n_clusters=n_cols, random_state=42, n_init=10)
            labels = km.fit_predict(x0s)

        cluster_centers = km.cluster_centers_.flatten()
        col_order = np.argsort(cluster_centers)

        sorted_boxes = []
        for col_idx in col_order:
            col_boxes = [b for i, b in enumerate(boxes) if labels[i] == col_idx]
            col_boxes.sort(key=lambda b: b["top"])
            sorted_boxes.extend(col_boxes)

        return sorted_boxes

    # ============================================
    # 三、页面处理主流程
    # ============================================

    def _process_page_normal(self, mupdf_page, plumber_page) -> str:
        """正常流程：提取文本、表格、处理布局"""
        page_height = mupdf_page.rect.height
        page_width = mupdf_page.rect.width

        blocks = self._extract_blocks(mupdf_page, plumber_page)

        if not blocks:
            return ""

        heights = [b["bottom"] - b["top"] for b in blocks]
        self.mean_height = np.median(heights) if heights else 12

        blocks = [b for b in blocks if not self._is_header_footer(b, page_height)]
        blocks = self._sort_by_reading_order(blocks, page_width)

        return self._blocks_to_markdown(blocks)

    def _extract_blocks(self, mupdf_page, plumber_page) -> List[Dict]:
        """
        提取文本块和表格块。

        逻辑：
        1. 先用 pdfplumber 在整页找强表格
        2. 表格作为独立 block
        3. 与表格区域重叠较大的 PyMuPDF 文本块跳过，避免重复
        4. 对没有被 find_tables 捕获的明显疑似表格，保留 fallback
        """
        blocks = []

        table_blocks = self._extract_tables_from_page(plumber_page)
        table_bboxes = [b["bbox"] for b in table_blocks]

        blocks.extend(table_blocks)

        mupdf_blocks = mupdf_page.get_text("dict")["blocks"]

        for block in mupdf_blocks:
            if block["type"] != 0:
                continue

            bbox = block["bbox"]
            text = self._extract_block_text(block)

            if not text.strip():
                continue

            if self._overlaps_any_table(bbox, table_bboxes):
                continue

            is_table = self._is_table_block(block, plumber_page)

            if is_table:
                table_md = self._extract_table_from_bbox(plumber_page, bbox)
                if table_md:
                    blocks.append({
                        "type": "table",
                        "x0": bbox[0],
                        "y0": bbox[1],
                        "x1": bbox[2],
                        "y1": bbox[3],
                        "top": bbox[1],
                        "bottom": bbox[3],
                        "bbox": bbox,
                        "text": table_md,
                    })
                    continue

            blocks.append({
                "type": "text",
                "x0": bbox[0],
                "y0": bbox[1],
                "x1": bbox[2],
                "y1": bbox[3],
                "top": bbox[1],
                "bottom": bbox[3],
                "bbox": bbox,
                "text": text,
            })

        return blocks

    def _extract_block_text(self, block: Dict) -> str:
        """从PyMuPDF块中提取文本"""
        text_parts = []
        for line in block.get("lines", []):
            line_parts = []
            for span in line.get("spans", []):
                text = span.get("text", "")
                if text:
                    line_parts.append(text)
            if line_parts:
                text_parts.append("".join(line_parts))
        return "\n".join(text_parts)

    # ============================================
    # 四、表格提取：保守强表格识别
    # ============================================

    def _extract_tables_from_page(self, plumber_page) -> List[Dict]:
        """
        使用 pdfplumber 在整页范围提取强表格。

        注意：
        这里故意不使用 vertical_strategy='text' + horizontal_strategy='text' 做全页表格检测，
        因为它很容易把目录、释义、列表、段落误判成表格。

        只保留 lines/lines 和默认策略，走保守路线。
        """
        table_blocks = []

        strategies = [
            {
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "intersection_tolerance": 5,
                "snap_tolerance": 3,
                "join_tolerance": 3,
            },
            {},
        ]

        seen_bboxes = []

        for strategy in strategies:
            try:
                tables = plumber_page.find_tables(table_settings=strategy)
            except Exception:
                continue

            for table_obj in tables:
                try:
                    bbox = tuple(table_obj.bbox)

                    if self._is_duplicate_bbox(bbox, seen_bboxes):
                        continue

                    table = table_obj.extract()

                    if not self._is_valid_table(table):
                        continue

                    table_md = self._table_to_markdown(table)
                    if not table_md.strip():
                        continue

                    seen_bboxes.append(bbox)

                    table_blocks.append({
                        "type": "table",
                        "x0": bbox[0],
                        "y0": bbox[1],
                        "x1": bbox[2],
                        "y1": bbox[3],
                        "top": bbox[1],
                        "bottom": bbox[3],
                        "bbox": bbox,
                        "text": table_md,
                    })

                except Exception:
                    continue

        return table_blocks

    def _is_table_block(self, block: Dict, plumber_page) -> bool:
        """
        判断 PyMuPDF 文本块是否为疑似表格。
        这是 fallback，不是主表格提取路径。

        为避免误判，只有满足明显表格特征才返回 True。
        """
        text = self._extract_block_text(block)
        bbox = block["bbox"]

        if not text.strip():
            return False

        if self._looks_like_toc(text):
            return False

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        line_count = len(lines)

        if line_count < 3:
            return False

        digit_count = sum(1 for c in text if c.isdigit())
        digit_ratio = digit_count / max(len(text), 1)
        multi_space_count = len(re.findall(r"\s{2,}", text))

        # 至少多行 + 多数字 + 多空格，才认为像表格
        if line_count >= 3 and digit_count >= 20 and digit_ratio >= 0.18 and multi_space_count >= 3:
            return True

        try:
            cropped = plumber_page.within_bbox(bbox)
            tables = cropped.extract_tables()
            if tables:
                for table in tables:
                    if self._is_valid_table(table):
                        return True
        except Exception:
            pass

        return False

    def _extract_table_from_bbox(self, plumber_page, bbox: Tuple) -> Optional[str]:
        """用 pdfplumber 在指定 bbox 内提取表格并转为 Markdown"""
        x0, y0, x1, y1 = bbox

        strategies = [
            {},
            {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
        ]

        for strategy in strategies:
            try:
                cropped = plumber_page.within_bbox((x0, y0, x1, y1))
                tables = cropped.extract_tables(table_settings=strategy)

                if tables:
                    for table in tables:
                        if self._is_valid_table(table):
                            return self._table_to_markdown(table)
            except Exception:
                continue

        return None

    def _is_valid_table(self, table: List) -> bool:
        """
        验证表格是否有效。

        目标：
        1. 保留财务指标表、收入成本表、季度指标表等强表格
        2. 过滤目录、释义、普通段落、单列碎片、短文本列表
        """
        if not table or len(table) < 3:
            return False

        cleaned_rows = []
        for row in table:
            if row is None:
                continue

            cleaned = [self._clean_cell(cell) for cell in row]

            if any(cell for cell in cleaned):
                cleaned_rows.append(cleaned)

        if len(cleaned_rows) < 3:
            return False

        max_cols = max(len(row) for row in cleaned_rows)

        # 单列表格基本都不要，很多误识别都来自这里
        if max_cols < 2:
            return False

        non_empty_cells = [
            cell
            for row in cleaned_rows
            for cell in row
            if cell
        ]

        if len(non_empty_cells) < 6:
            return False

        total_text = " ".join(non_empty_cells)

        if self._looks_like_toc(total_text):
            return False

        if self._looks_like_definition_list(cleaned_rows):
            return False

        # 表格应当有一定结构密度：至少 2 行有 2 个以上非空单元格
        dense_rows = 0
        for row in cleaned_rows:
            non_empty_in_row = sum(1 for cell in row if cell)
            if non_empty_in_row >= 2:
                dense_rows += 1

        if dense_rows < 2:
            return False

        # 金融表格通常包含数字、百分比、金额单位、GWh等
        numeric_cells = 0
        metric_cells = 0

        for cell in non_empty_cells:
            if re.search(r"\d", cell):
                numeric_cells += 1

            if re.search(
                r"[%％]|元|千元|万元|亿元|GWh|MWh|Wh/kg|次|股|吨|辆|座",
                cell,
                flags=re.IGNORECASE,
            ):
                metric_cells += 1

        # 强表格：有足够数字或指标单位
        if numeric_cells >= 3 or metric_cells >= 2:
            return True

        return False

    def _looks_like_toc(self, text: str) -> bool:
        """判断是否像目录页内容"""
        if not text:
            return False

        dot_count = text.count(".") + text.count("·") + text.count("…")
        if dot_count >= 10:
            return True

        section_hits = len(re.findall(r"第[一二三四五六七八九十\d]+节", text))

        if section_hits >= 2:
            return True

        if "目录" in text and section_hits >= 1:
            return True

        return False

    def _looks_like_definition_list(self, rows: List[List[str]]) -> bool:
        """
        判断是否像释义表/定义列表。

        这类内容虽然有表格形态，但第一版 RAG 可以先保守处理为 text，
        避免把大量普通解释性内容误判成 table。
        """
        flat_text = " ".join(
            cell
            for row in rows
            for cell in row
            if cell
        )

        if not flat_text:
            return False

        if "释义项" in flat_text and "释义内容" in flat_text:
            return True

        zhi_count = flat_text.count(" 指 ")
        if zhi_count >= 3:
            return True

        return False

    def _clean_cell(self, cell) -> str:
        """清洗表格单元格"""
        if cell is None:
            return ""
        text = str(cell)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _table_to_markdown(self, table: List) -> str:
        """表格转Markdown"""
        if not table:
            return ""

        cleaned_table = []

        for row in table:
            if row is None:
                continue
            cleaned_row = [self._clean_cell(cell) for cell in row]
            if any(cell for cell in cleaned_row):
                cleaned_table.append(cleaned_row)

        if not cleaned_table:
            return ""

        max_cols = max(len(row) for row in cleaned_table)

        normalized_table = []
        for row in cleaned_table:
            normalized_row = list(row)
            while len(normalized_row) < max_cols:
                normalized_row.append("")
            normalized_table.append(normalized_row)

        lines = []

        header = normalized_table[0]
        lines.append("| " + " | ".join(self._escape_markdown_cell(cell) for cell in header) + " |")
        lines.append("|" + "|".join(["---"] * max_cols) + "|")

        for row in normalized_table[1:]:
            lines.append("| " + " | ".join(self._escape_markdown_cell(cell) for cell in row) + " |")

        return "\n".join(lines)

    def _escape_markdown_cell(self, cell: str) -> str:
        """避免单元格中的 | 破坏 Markdown 表格"""
        if cell is None:
            return ""
        return str(cell).replace("|", "\\|").strip()

    # ============================================
    # 五、bbox 工具函数
    # ============================================

    def _is_duplicate_bbox(self, bbox: Tuple, seen_bboxes: List[Tuple], iou_threshold: float = 0.7) -> bool:
        """判断表格 bbox 是否重复"""
        for seen in seen_bboxes:
            if self._bbox_iou(bbox, seen) >= iou_threshold:
                return True
        return False

    def _bbox_iou(self, a: Tuple, b: Tuple) -> float:
        """计算两个 bbox 的 IoU"""
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b

        inter_x0 = max(ax0, bx0)
        inter_y0 = max(ay0, by0)
        inter_x1 = min(ax1, bx1)
        inter_y1 = min(ay1, by1)

        inter_w = max(0, inter_x1 - inter_x0)
        inter_h = max(0, inter_y1 - inter_y0)
        inter_area = inter_w * inter_h

        area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
        area_b = max(0, bx1 - bx0) * max(0, by1 - by0)

        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0

        return inter_area / union

    def _overlaps_any_table(self, bbox: Tuple, table_bboxes: List[Tuple]) -> bool:
        """
        判断文本块是否与任意表格 bbox 重叠。
        重叠面积 / 文本块面积 超过阈值时认为该文本块属于表格区域。
        """
        if not table_bboxes:
            return False

        for table_bbox in table_bboxes:
            ratio = self._bbox_overlap_ratio(bbox, table_bbox)
            if ratio >= self.table_overlap_threshold:
                return True

        return False

    def _bbox_overlap_ratio(self, block_bbox: Tuple, table_bbox: Tuple) -> float:
        """计算 block 被 table 覆盖的比例"""
        bx0, by0, bx1, by1 = block_bbox
        tx0, ty0, tx1, ty1 = table_bbox

        inter_x0 = max(bx0, tx0)
        inter_y0 = max(by0, ty0)
        inter_x1 = min(bx1, tx1)
        inter_y1 = min(by1, ty1)

        inter_w = max(0, inter_x1 - inter_x0)
        inter_h = max(0, inter_y1 - inter_y0)
        inter_area = inter_w * inter_h

        block_area = max(0, bx1 - bx0) * max(0, by1 - by0)
        if block_area <= 0:
            return 0.0

        return inter_area / block_area

    # ============================================
    # 六、页眉页脚与 Markdown 输出
    # ============================================

    def _is_header_footer(self, block: Dict, page_height: float) -> bool:
        """判断是否为页眉页脚"""
        y0 = block["y0"]
        y1 = block["y1"]
        height = y1 - y0

        if y1 <= self.header_margin:
            return True

        if y0 >= page_height - self.footer_margin:
            return True

        if height < 20 and (y0 < self.header_margin or y0 > page_height - self.footer_margin):
            return True

        return False

    def _blocks_to_markdown(self, blocks: List[Dict]) -> str:
        """将块列表转为Markdown"""
        lines = []

        for block in blocks:
            text = block["text"].strip()
            if not text:
                continue

            if block["type"] == "table":
                lines.append("[TABLE_START]")
                lines.append(text)
                lines.append("[TABLE_END]")
                lines.append("")
            else:
                lines.append(text)
                lines.append("")

        return "\n".join(lines)

    # ============================================
    # 七、降级方案
    # ============================================

    def _extract_text_only(self, mupdf_page) -> str:
        """纯文本提取（降级方案）"""
        return mupdf_page.get_text("text")

    def _ocr_fallback(self, mupdf_page) -> str:
        """
        OCR降级（需要安装paddleocr）
        pip install paddlepaddle paddleocr
        """
        try:
            from paddleocr import PaddleOCR

            pix = mupdf_page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
            result = ocr.ocr(np.array(img), cls=True)

            lines = []
            if result and result[0]:
                for line in result[0]:
                    lines.append(line[1][0])

            return "\n".join(lines)

        except ImportError:
            return self._extract_text_only(mupdf_page)

    def supported_extensions(self) -> List[str]:
        return [".pdf"]


# ============================================
# 批量处理函数
# ============================================

def batch_parse_pdfs(
    input_dir: Path,
    output_dir: Path,
    parser: EnhancedPDFParser,
    force_reprocess: bool = False,
    skip_errors: bool = True,
) -> Dict:
    """
    批量处理目录下的所有 PDF 文件
    
    Args:
        input_dir: 输入 PDF 文件目录
        output_dir: 输出 Markdown 文件目录
        parser: PDF 解析器实例
        force_reprocess: 是否强制重新处理（忽略已存在的文件）
        skip_errors: 是否跳过出错的 PDF 继续处理其他文件
    
    Returns:
        处理结果统计
    """
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有 PDF 文件
    pdf_files = list(input_dir.glob("*.pdf"))
    pdf_files.sort()
    
    print("=" * 80)
    print(f"📂 PDF 批量解析工具")
    print(f"   输入目录: {input_dir}")
    print(f"   输出目录: {output_dir}")
    print(f"   发现 PDF 文件: {len(pdf_files)} 个")
    print(f"   强制重新处理: {force_reprocess}")
    print(f"   跳过错误: {skip_errors}")
    print("=" * 80)
    
    # 统计信息
    stats = {
        "total": len(pdf_files),
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "files": [],
    }
    
    start_time = time.time()
    
    for idx, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{idx}/{len(pdf_files)}] 处理文件: {pdf_file.name}")
        
        # 确定输出文件路径
        output_file = output_dir / f"{pdf_file.stem}.md"
        
        # 检查是否已处理
        if not force_reprocess and output_file.exists():
            # 检查文件是否为空
            if output_file.stat().st_size > 0:
                print(f"   ⏭️ 跳过（已存在）: {output_file.name}")
                stats["skipped"] += 1
                stats["files"].append({
                    "name": pdf_file.name,
                    "status": "skipped",
                    "output": str(output_file),
                })
                continue
            else:
                print(f"   ⚠️ 发现空文件，重新处理: {output_file.name}")
        
        # 解析 PDF
        try:
            file_start_time = time.time()
            result = parser.parse(pdf_file)
            elapsed = time.time() - file_start_time
            
            # 保存结果
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result)
            
            # 统计字符数
            char_count = len(result)
            print(f"   ✅ 完成! 耗时: {elapsed:.2f}s, 输出: {char_count} 字符")
            print(f"   💾 保存到: {output_file}")
            
            stats["processed"] += 1
            stats["files"].append({
                "name": pdf_file.name,
                "status": "success",
                "output": str(output_file),
                "char_count": char_count,
                "elapsed": elapsed,
            })
            
        except Exception as e:
            print(f"   ❌ 解析失败: {type(e).__name__}: {e}")
            stats["failed"] += 1
            stats["files"].append({
                "name": pdf_file.name,
                "status": "failed",
                "error": str(e),
            })
            
            if not skip_errors:
                print("   检测到错误，停止处理")
                break
    
    # 打印总结
    elapsed_total = time.time() - start_time
    print("\n" + "=" * 80)
    print("📊 批量处理完成!")
    print(f"   总计: {stats['total']} 个文件")
    print(f"   ✅ 成功: {stats['processed']}")
    print(f"   ⏭️ 跳过: {stats['skipped']}")
    print(f"   ❌ 失败: {stats['failed']}")
    print(f"   ⏱️ 总耗时: {elapsed_total:.2f}秒")
    print("=" * 80)
    
    # 保存统计信息
    stats_file = output_dir / "_processing_stats.json"
    import json
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n📈 统计信息已保存到: {stats_file}")
    
    return stats


# ============================================
# 主程序
# ============================================

if __name__ == "__main__":
    import argparse

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    
    parser = argparse.ArgumentParser(description="增强版PDF解析器 - 批量处理")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(PROJECT_ROOT / "data/pdfs"),
        help="PDF 文件输入目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "data/parsed"),
        help="Markdown 输出目录"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新处理所有文件（忽略已存在的输出）"
    )
    parser.add_argument(
        "--no-skip-errors",
        action="store_true",
        help="遇到错误时停止处理（默认跳过错误继续）"
    )
    parser.add_argument(
        "--header-margin",
        type=float,
        default=80,
        help="页眉区域高度"
    )
    parser.add_argument(
        "--footer-margin",
        type=float,
        default=80,
        help="页脚区域高度"
    )
    parser.add_argument(
        "--use-ocr",
        action="store_true",
        help="启用 OCR 降级（需要安装 paddleocr）"
    )
    parser.add_argument(
        "--single-file",
        type=str,
        default=None,
        help="处理单个 PDF 文件（指定文件路径）"
    )
    
    args = parser.parse_args()
    
    # 创建解析器实例
    pdf_parser = EnhancedPDFParser(
        header_margin=args.header_margin,
        footer_margin=args.footer_margin,
        use_ocr_fallback=args.use_ocr,
        table_overlap_threshold=0.3,
    )
    
    if args.single_file:
        # 处理单个文件
        pdf_file = Path(args.single_file)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{pdf_file.stem}.md"
        
        result = pdf_parser.parse(pdf_file)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"✅ 结果已保存到: {output_file}")
    else:
        # 批量处理
        stats = batch_parse_pdfs(
            input_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            parser=pdf_parser,
            force_reprocess=args.force,
            skip_errors=not args.no_skip_errors,
        )
