#!/usr/bin/env python3
"""
表格保护切块器 - Table-aware Chunker

功能：
1. 读取 data/parsed/ 下的 Markdown 文件
2. 默认 chunk_size=1024, overlap=50
3. 支持 [TABLE_START] ... [TABLE_END] 表格保护
4. 同时生成：
   - data/chunks/by_doc/{doc_id}_cs{chunk_size}_ov{overlap}.jsonl
   - data/chunks/all_cs{chunk_size}_ov{overlap}.jsonl
5. 支持 --run_experiments 执行 3x3 切块实验
6. 新增精简 document metadata：
   - doc_id
   - doc_title
   - doc_type
   - source_type
   - company
   - company_short
   - year
   - period
   - source_file
7. metadata 只保存，不参与 embedding / BM25 文本构造

默认运行：
python src/chunking/table_aware_chunker.py

开启切块实验：
python src/chunking/table_aware_chunker.py --run_experiments

使用人工 metadata 覆盖：
python src/chunking/table_aware_chunker.py \
  --metadata_file data/doc_metadata.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================
# 公司简称映射
# ============================================

COMPANY_ALIAS=({
    "晶门半导体": "晶门半导体",
    "荣阳实业": "荣阳实业",
    "彩虹新能源": "彩虹新能源",
    "昆仑万维": "昆仑万维",
    "芯海科技": "芯海科技",
    "三只松鼠": "三只松鼠",
    "深圳市三旺通信股份有限公司": "三旺通信",
    "三旺通信": "三旺通信",
    "君亭酒店": "君亭酒店",
    "浙江帕瓦新能源股份有限公司": "帕瓦股份",
    "帕瓦新能源": "帕瓦新能源",
    "帕瓦股份": "帕瓦股份",
    "依波路": "依波路",
    "富盈环球集团": "富盈环球",
    "富盈环球": "富盈环球",
    "科蓝软件": "科蓝软件",
    "重庆千里科技股份有限公司": "重庆千里",
    "重庆千里科技": "重庆千里",
    "重庆千里": "重庆千里",
    "汉商集团": "汉商集团",
    "箭牌家居": "箭牌家居",
    "竞业达": "竞业达",
    "中国智能交通": "中国智能交通",
    "皖通科技": "皖通科技",
    "奇安信": "奇安信",
    "乐鑫科技": "乐鑫科技",
    "大唐新能源": "大唐新能源",
    "瀛晟科学": "瀛晟科学",
    "FUTURE BRIGHT": "FUTURE BRIGHT",
    "威腾电气集团股份有限公司": "威腾电气",
    "威腾电气": "威腾电气",
    "慧博云通": "慧博云通",
    "朸濬国际": "朸濬国际",
    "中建富通": "中建富通",
    "权识国际": "权识国际",
    "供销大集": "供销大集",
    "立昂技术": "立昂技术",
    "凯撒旅业": "凯撒旅业",
    "熵基科技": "熵基科技",
    "嘉士利集团": "嘉士利集团",
    "北京海天瑞声科技股份有限公司": "海天瑞声",
    "海天瑞声": "海天瑞声",
    "华控赛格": "华控赛格",
    "合兴汽车电子股份有限公司": "合兴股份",
    "合兴汽车电子": "合兴汽车电子",
    "合兴股份": "合兴股份",
    "味千（中国）": "味千中国",
    "味千中国": "味千中国",
    "绿岛风": "绿岛风",
    "英集芯": "英集芯",
    "瀛海集团": "瀛海集团",
    "交大慧谷": "交大慧谷",
    "金石资源集团股份有限公司": "金石资源",
    "金石资源": "金石资源",
    "海信家电": "海信家电",
    "创识科技": "创识科技",
    "盛达资源": "盛达资源",
    "爱德新能源": "爱德新能源",
    "东华能源": "东华能源",
    "龙源电力": "龙源电力",
    "四方光电": "四方光电",
    "中国圣牧": "中国圣牧",
    "五矿新能源材料（湖南）股份有限公司": "五矿新能",
    "五矿新能源材料": "五矿新能源材料",
    "宏和科技": "宏和科技",
    "松景科技": "松景科技",
    "世纪瑞尔": "世纪瑞尔",
    "安联锐视": "安联锐视",
    "杰创智能": "杰创智能",
    "叙福楼集团": "叙福楼集团",
    "东华软件": "东华软件",
    "扬州金泉": "扬州金泉",
    "协鑫新能源": "协鑫新能源",
    "亿道信息": "亿道信息",
    "皇马科技": "皇马科技",
    "中国能源建设股份有限公司": "中国能建",
    "中国能源建设": "中国能建",
    "中国能建": "中国能建",
    "龙芯中科": "龙芯中科",
    "蒙牛乳业": "蒙牛乳业",
    "网誉科技": "网誉科技",
    "中油工程": "中油工程",
    "佐丹奴国际": "佐丹奴国际",
    "植华集团": "植华集团",
    "辽宁时代万恒股份有限公司": "时代万恒",
    "时代万恒": "时代万恒",
    "中国食品": "中国食品",
    "华联综超": "华联综超",
    "能辉科技": "能辉科技",
    "明阳电气": "明阳电气",
    "新集能源": "新集能源",
    "山水比德": "山水比德",
    "旋极信息": "旋极信息",
    "澳优": "澳优",
    "仁智股份": "仁智股份",
    "中原内配": "中原内配",
    "罗普特科技集团股份有限公司": "罗普特",
    "罗普特科技": "罗普特",
    "罗普特": "罗普特",
    "晶雪节能": "晶雪节能",
    "浙江德宏汽车电子电器股份有限公司": "德宏股份",
    "德宏汽车电子": "德宏汽车电子",
    "德宏股份": "德宏股份",
    "数码通电讯": "数码通电讯",
    "山西华阳集团新能股份有限公司": "华阳股份",
    "华阳股份": "华阳股份",
    "宏光半导体": "宏光半导体",
    "北京龙软科技股份有限公司": "龙软科技",
    "龙软科技": "龙软科技",
    "中科微至": "中科微至",
    "来伊份": "来伊份",
    "中信博": "中信博",
    "原生态牧业": "原生态牧业",
    "浙江正特": "浙江正特",
    "浙江新能": "浙江新能",
    "米格国际控股": "米格国际控股",
    "漱玉平民": "漱玉平民",
    "天润工业": "天润工业",
    "菱电电控": "菱电电控",
    "猫屎咖啡控股": "猫屎咖啡控股",
    "华致酒行": "华致酒行",
    "卡姆丹克太阳能": "卡姆丹克太阳能",
    "中广核新能源": "中广核新能源",
    "南京新百": "南京新百",
    "佳都科技": "佳都科技",
    "捷安高科": "捷安高科",
    "日清食品": "日清食品",
    "细叶榕科技": "细叶榕科技",
    "同程旅行": "同程旅行",
    "阳光乳业": "阳光乳业",
    "宝通科技": "宝通科技",
    "交控科技股份有限公司": "交控科技",
    "交控科技": "交控科技",
    "陕西能源": "陕西能源",
    "众信旅游": "众信旅游",
    "嘉曼服饰": "嘉曼服饰",
    "新秀丽": "新秀丽",
    "王朝酒业": "王朝酒业",
    "ST岭南": "ST岭南",
    "万集科技": "万集科技",
    "数字政通": "数字政通",
    "哈尔滨新光光电科技股份有限公司": "新光光电",
    "新光光电": "新光光电",
    "新开普": "新开普",
    "欣锐科技": "欣锐科技",
    "格灵深瞳": "格灵深瞳",
    "力宝华润": "力宝华润",
    "汉诺佳池": "汉诺佳池",
    "信濠光电": "信濠光电",
    "天山电子": "天山电子",
    "旷世芳香": "旷世芳香",
    "华住集团－Ｓ": "华住集团",
    "华住集团": "华住集团",
    "华锋股份": "华锋股份",
    "瑞风新能源": "瑞风新能源",
    "四川省新能源动力股份有限公司": "川能动力",
    "川能动力": "川能动力",
    "中稀有色金属股份有限公司": "中稀有色",
    "中稀有色": "中稀有色",
    "亚香股份": "亚香股份",
    "天元医疗": "天元医疗",
    "苏州上声电子股份有限公司": "上声电子",
    "上声电子": "上声电子",
    "居然智家": "居然智家",
    "科大讯飞": "科大讯飞",
    "统一企业中国": "统一企业中国",
    "盛诺集团": "盛诺集团",
    "湖北广电": "湖北广电",
    "亚太股份": "亚太股份",
    "致欧科技": "致欧科技",
    "皇庭智家": "皇庭智家",
    "南网能源": "南网能源",
    "ST雪发": "ST雪发",
    "莱斯信息": "莱斯信息",
    "第一太平": "第一太平",
    "大全能源": "大全能源",
    "鸥玛软件": "鸥玛软件",
    "恒锋信息": "恒锋信息",
    "金现代": "金现代",
    "金禄电子": "金禄电子",
    "申昊科技": "申昊科技",
    "中远海科": "中远海科",
})


REPORT_TYPE_MAP = {
    "年度报告": ("annual_report", "annual"),
    "年报": ("annual_report", "annual"),
    "半年度报告": ("semi_annual_report", "semi_annual"),
    "中期报告": ("semi_annual_report", "semi_annual"),
    "第一季度报告": ("quarterly_report", "Q1"),
    "第二季度报告": ("quarterly_report", "Q2"),
    "第三季度报告": ("quarterly_report", "Q3"),
    "第四季度报告": ("quarterly_report", "Q4"),
    "季度报告": ("quarterly_report", "quarterly"),
}


POLICY_KEYWORDS = [
    "规划",
    "通知",
    "意见",
    "指导意见",
    "办法",
    "管理办法",
    "实施方案",
    "方案",
    "政策",
    "条例",
    "细则",
    "决定",
    "公告",
    "蓝皮书",
    "白皮书"
]


RESEARCH_KEYWORDS = [
    "行业研究",
    "深度报告",
    "证券研究",
    "研报",
    "研究报告",
    "点评",
    "策略",
]


# ============================================
# Metadata 提取
# ============================================

def load_metadata_overrides(metadata_file: Optional[Path]) -> Dict[str, Dict]:
    """
    读取人工 metadata 覆盖文件。

    格式示例：
    {
      "11987738": {
        "doc_title": "宁德时代2025年年度报告",
        "doc_type": "annual_report",
        "source_type": "company_report",
        "company": "宁德时代",
        "company_short": "宁德时代",
        "year": "2025",
        "period": "annual"
      }
    }
    """
    if metadata_file is None:
        return {}

    if not metadata_file.exists():
        print(f"metadata_file 不存在，跳过人工覆盖: {metadata_file}")
        return {}

    with metadata_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("doc_metadata.json 顶层必须是 dict")

    print(f"已加载人工 metadata 覆盖: {metadata_file}, 条数={len(data)}")
    return data


def extract_doc_metadata(input_file: Path, text: str, overrides: Dict[str, Dict]) -> Dict:
    """
    提取文档级 metadata。

    优先级：
    1. md 正文首页规则
    2. 文件名规则
    3. doc_metadata.json 人工覆盖
    4. 默认兜底

    注意：
    - 这里文件名规则会覆盖正文规则，因为用户已清洗文件名。
    - 人工覆盖优先级最高。
    """
    doc_id = input_file.stem
    source_file = input_file.name

    base = default_metadata(doc_id=doc_id, source_file=source_file)

    text_meta = extract_metadata_from_text(text[:5000])
    filename_meta = extract_metadata_from_filename(input_file.name)

    merged = base.copy()
    merged.update(non_empty_items(text_meta))
    merged.update(non_empty_items(filename_meta))

    override = overrides.get(doc_id) or overrides.get(source_file) or {}
    merged.update(non_empty_items(override))

    if merged.get("company") and not merged.get("company_short"):
        merged["company_short"] = shorten_company_name(merged["company"])

    if not merged.get("doc_title"):
        merged["doc_title"] = build_doc_title(merged)

    return merged


def default_metadata(doc_id: str, source_file: str) -> Dict:
    return {
        "doc_id": doc_id,
        "doc_title": "",
        "doc_type": "unknown",
        "source_type": "unknown",
        "company": "",
        "company_short": "",
        "year": "",
        "period": "",
        "source_file": source_file,
    }


def non_empty_items(d: Dict) -> Dict:
    allowed_keys = {
        "doc_id",
        "doc_title",
        "doc_type",
        "source_type",
        "company",
        "company_short",
        "year",
        "period",
        "source_file",
    }

    return {
        k: v
        for k, v in d.items()
        if k in allowed_keys and v is not None and v != ""
    }


def extract_metadata_from_filename(filename: str) -> Dict:
    """
    从文件名提取 metadata。

    支持示例：
    - 宁德时代_2025年年度报告.md
    - 比亚迪2025年年度报告.md
    - 2025年比亚迪年度报告.md
    - 比亚迪年度报告2025.md
    - 新能源汽车产业发展规划_2021-2035.md
    - 关于促进新型储能并网和调度运用的通知_2024.md

    年份识别规则：
    - 在文件名任意位置查找第一个 20xx
    - 不要求出现在文件名开头
    """
    stem = Path(filename).stem
    name = clean_filename_stem(stem)

    meta = empty_auto_metadata()
    meta["source_file"] = filename

    year = extract_year(name)

    # 1. 公司报告：公司名 + 年份 + 报告类型
    report_type_cn = detect_report_type(name)
    if report_type_cn:
        company = extract_company_from_report_filename(name, year, report_type_cn)
        doc_type, period = REPORT_TYPE_MAP.get(report_type_cn, ("company_report", ""))

        if company:
            meta.update({
                "doc_title": f"{company}{year}年{report_type_cn}" if year else f"{company}{report_type_cn}",
                "doc_type": doc_type,
                "source_type": "company_report",
                "company": company,
                "company_short": shorten_company_name(company),
                "year": year,
                "period": period,
            })
            return meta

    # 2. 政策文件
    if looks_like_policy_name(name):
        meta.update({
            "doc_title": name,
            "doc_type": "policy",
            "source_type": "policy_file",
            "year": year,
            "period": "",
        })
        return meta

    # 3. 行业研报 / 研究报告
    if looks_like_research_name(name):
        meta.update({
            "doc_title": name,
            "doc_type": "industry_research",
            "source_type": "research_report",
            "year": year,
            "period": "",
        })
        return meta

    # 4. 未识别，保留标题和年份
    meta.update({
        "doc_title": name,
        "year": year,
    })

    return meta


def empty_auto_metadata() -> Dict:
    return {
        "doc_title": "",
        "doc_type": "",
        "source_type": "",
        "company": "",
        "company_short": "",
        "year": "",
        "period": "",
        "source_file": "",
    }


def extract_metadata_from_text(head_text: str) -> Dict:
    """
    从 md 正文前部提取 metadata。

    主要用于类似 11987738.md 这种文件名无法提供信息的情况。
    """
    lines = [
        line.strip()
        for line in head_text.splitlines()
        if line.strip() and line.strip() != "---"
    ]

    meta = empty_auto_metadata()

    if not lines:
        return meta

    report_pattern = re.compile(
        r"(?P<year>20\d{2})\s*年\s*"
        r"(?P<report_type>年度报告|半年度报告|第一季度报告|第三季度报告|季度报告)"
    )

    report_line_idx = -1
    report_type_cn = ""

    for i, line in enumerate(lines[:50]):
        m = report_pattern.search(line)
        if m:
            year = m.group("year")
            report_type_cn = m.group("report_type")
            doc_type, period = REPORT_TYPE_MAP.get(report_type_cn, ("company_report", ""))

            meta["year"] = year
            meta["doc_type"] = doc_type
            meta["source_type"] = "company_report"
            meta["period"] = period
            report_line_idx = i
            break

    if report_line_idx > 0:
        company = lines[report_line_idx - 1].strip()
        if is_probable_company_name(company):
            meta["company"] = company
            meta["company_short"] = shorten_company_name(company)

    if meta["company"] and meta["year"] and report_type_cn:
        meta["doc_title"] = f"{meta['company']}{meta['year']}年{report_type_cn}"

    # 政策文件正文兜底
    first_part = " ".join(lines[:30])
    if not meta["doc_type"] and looks_like_policy_name(first_part):
        meta["doc_type"] = "policy"
        meta["source_type"] = "policy_file"
        meta["year"] = extract_year(first_part)
        meta["doc_title"] = lines[0] if lines else ""
        meta["period"] = ""

    return meta


def clean_filename_stem(stem: str) -> str:
    """
    清洗历史文件名中的交易所、研报网站等噪声。
    对已经清洗好的文件名也安全。
    """
    name = stem

    name = re.sub(r"【洞见研报.*?】", "", name)
    name = re.sub(r"【上交所科创板】", "", name)
    name = re.sub(r"【上交所】", "", name)
    name = re.sub(r"【深交所】", "", name)
    name = re.sub(r"【[^】]+】", "", name)

    name = name.strip(" _-：:")
    name = re.sub(r"\s+", "", name)

    return name


def extract_year(text: str) -> str:
    """
    从文本任意位置提取年份。

    规则：
    - 匹配第一个 20xx
    - 不要求在开头
    - 支持 2021-2035 这类政策规划，返回 2021
    """
    m = re.search(r"(20\d{2})", text)
    return m.group(1) if m else ""


def detect_report_type(text: str) -> str:
    """
    检测报告类型。

    注意顺序：
    - 先匹配更长的报告类型，避免 “第三季度报告” 被 “季度报告” 抢先匹配。
    """
    report_types = [
        "第一季度报告",
        "第三季度报告",
        "半年度报告",
        "年度报告",
        "季度报告",
    ]

    for rt in report_types:
        if rt in text:
            return rt

    return ""


def extract_company_from_report_filename(name: str, year: str, report_type_cn: str) -> str:
    """
    从公司报告文件名中提取公司名。

    支持：
    - 比亚迪_2025年年度报告
    - 比亚迪2025年年度报告
    - 2025年比亚迪年度报告
    - 比亚迪年度报告2025
    """
    company = name

    # 去掉报告类型
    company = company.replace(report_type_cn, "")

    # 去掉年份和 “年”
    if year:
        company = company.replace(f"{year}年", "")
        company = company.replace(year, "")

    company = company.strip(" _-：:")

    # 清理常见残留词
    company = company.replace("报告", "")
    company = company.strip(" _-：:")

    # 如果还有冒号前缀/后缀，进一步清洗
    company = re.sub(r"^[：:_\-\s]+", "", company)
    company = re.sub(r"[：:_\-\s]+$", "", company)

    # 防止空
    if not company:
        return ""

    return company


def looks_like_policy_name(text: str) -> bool:
    return any(kw in text for kw in POLICY_KEYWORDS)


def looks_like_research_name(text: str) -> bool:
    return any(kw in text for kw in RESEARCH_KEYWORDS)


def is_probable_company_name(text: str) -> bool:
    if not text:
        return False

    company_suffixes = [
        "股份有限公司",
        "有限责任公司",
        "有限公司",
        "集团股份有限公司",
        "集团有限公司",
        "公司",
    ]

    return any(text.endswith(suffix) for suffix in company_suffixes)


def shorten_company_name(company: str) -> str:
    if not company:
        return ""

    if company in COMPANY_ALIAS:
        return COMPANY_ALIAS[company]

    short = company

    suffixes = [
        "新能源科技股份有限公司",
        "汽车电子股份有限公司",
        "集团股份有限公司",
        "股份有限公司",
        "有限责任公司",
        "集团有限公司",
        "有限公司",
        "公司",
    ]

    for suffix in suffixes:
        if short.endswith(suffix):
            short = short[: -len(suffix)]
            break

    return short.strip()


def report_type_cn_from_doc_type(doc_type: str, period: str) -> str:
    if doc_type == "annual_report" or period == "annual":
        return "年度报告"
    if doc_type == "semi_annual_report" or period == "semi_annual":
        return "半年度报告"
    if period == "q1":
        return "第一季度报告"
    if period == "q3":
        return "第三季度报告"
    if doc_type == "quarterly_report":
        return "季度报告"
    return "报告"


def build_doc_title(meta: Dict) -> str:
    if meta.get("company") and meta.get("year") and meta.get("period"):
        report_type_cn = report_type_cn_from_doc_type(
            meta.get("doc_type", ""),
            meta.get("period", ""),
        )
        return f"{meta['company']}{meta['year']}年{report_type_cn}"

    if meta.get("doc_title"):
        return meta["doc_title"]

    if meta.get("source_file"):
        return Path(meta["source_file"]).stem

    return meta.get("doc_id", "")


# ============================================
# Chunker
# ============================================

class TableAwareChunker:
    """表格保护切块器"""

    def __init__(
        self,
        chunk_size: int = 256,
        overlap: int = 50,
        min_chunk_size: int = 30,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")

        if overlap < 0:
            raise ValueError("overlap 不能小于 0")

        if overlap >= chunk_size:
            raise ValueError("overlap 必须小于 chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

    # ============================================
    # 主入口
    # ============================================

    def chunk_file(
        self,
        input_file: Path,
        doc_id: Optional[str] = None,
        doc_metadata: Optional[Dict] = None,
    ) -> List[Dict]:
        """读取单个 Markdown 文件并切块"""
        if not input_file.exists():
            raise FileNotFoundError(f"文件不存在: {input_file}")

        if doc_id is None:
            doc_id = input_file.stem

        text = input_file.read_text(encoding="utf-8")

        if doc_metadata is None:
            doc_metadata = default_metadata(doc_id=doc_id, source_file=input_file.name)

        return self.chunk_text(text=text, doc_id=doc_id, doc_metadata=doc_metadata)

    def chunk_text(self, text: str, doc_id: str, doc_metadata: Dict) -> List[Dict]:
        """对 Markdown 文本进行 table-aware 切块"""
        segments = self._split_table_and_text_segments(text)

        chunks = []
        text_chunk_count = 0
        table_chunk_count = 0

        for seg in segments:
            seg_type = seg["type"]
            content = seg["content"].strip()

            if not content:
                continue

            if seg_type == "table":
                table_chunk_count += 1
                chunk = self._build_chunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}_table_{table_chunk_count:04d}",
                    chunk_type="table",
                    content=content,
                    doc_metadata=doc_metadata,
                    extra={
                        "source_segment_index": seg["segment_index"],
                        "chunk_size": self.chunk_size,
                        "overlap": self.overlap,
                    },
                )
                chunks.append(chunk)

            else:
                text_chunks = self._split_text_with_overlap(content)

                for text_chunk in text_chunks:
                    text_chunk = text_chunk.strip()
                    if len(text_chunk) < self.min_chunk_size:
                        continue

                    text_chunk_count += 1
                    chunk = self._build_chunk(
                        doc_id=doc_id,
                        chunk_id=f"{doc_id}_text_{text_chunk_count:04d}",
                        chunk_type="text",
                        content=text_chunk,
                        doc_metadata=doc_metadata,
                        extra={
                            "source_segment_index": seg["segment_index"],
                            "chunk_size": self.chunk_size,
                            "overlap": self.overlap,
                        },
                    )
                    chunks.append(chunk)

        return chunks

    # ============================================
    # 表格保护
    # ============================================

    def _split_table_and_text_segments(self, text: str) -> List[Dict]:
        """
        将全文拆成 text/table segment。

        表格格式：
        [TABLE_START]
        ...
        [TABLE_END]
        """
        pattern = re.compile(
            r"\[TABLE_START\](.*?)\[TABLE_END\]",
            flags=re.DOTALL | re.IGNORECASE,
        )

        segments = []
        cursor = 0
        segment_index = 0

        for match in pattern.finditer(text):
            start, end = match.span()

            before_text = text[cursor:start].strip()
            if before_text:
                segment_index += 1
                segments.append({
                    "segment_index": segment_index,
                    "type": "text",
                    "content": before_text,
                })

            table_content = text[start:end].strip()
            if table_content:
                segment_index += 1
                segments.append({
                    "segment_index": segment_index,
                    "type": "table",
                    "content": table_content,
                })

            cursor = end

        tail_text = text[cursor:].strip()
        if tail_text:
            segment_index += 1
            segments.append({
                "segment_index": segment_index,
                "type": "text",
                "content": tail_text,
            })

        return segments

    # ============================================
    # 普通文本切块
    # ============================================

    def _split_text_with_overlap(self, text: str) -> List[str]:
        """普通文本 sliding window 切块"""
        text = self._normalize_text(text)

        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            hard_end = min(start + self.chunk_size, text_len)
            window = text[start:hard_end]

            if hard_end >= text_len:
                chunk = text[start:text_len]
                if chunk.strip():
                    chunks.append(chunk)
                break

            split_pos = self._find_best_split_position(window)

            if split_pos <= 0:
                split_pos = len(window)

            end = start + split_pos
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            next_start = end - self.overlap

            if next_start <= start:
                next_start = end

            start = max(0, next_start)

        return chunks

    def _find_best_split_position(self, window: str) -> int:
        """在当前窗口内寻找最合适的切分点"""
        if not window:
            return 0

        min_pos = int(len(window) * 0.6)

        paragraph_candidates = [
            m.end()
            for m in re.finditer(r"\n\s*\n", window)
            if m.end() >= min_pos
        ]
        if paragraph_candidates:
            return paragraph_candidates[-1]

        zh_sentence_candidates = [
            m.end()
            for m in re.finditer(r"[。！？；]", window)
            if m.end() >= min_pos
        ]
        if zh_sentence_candidates:
            return zh_sentence_candidates[-1]

        en_sentence_candidates = [
            m.end()
            for m in re.finditer(r"[.!?;]", window)
            if m.end() >= min_pos
        ]
        if en_sentence_candidates:
            return en_sentence_candidates[-1]

        newline_candidates = [
            m.end()
            for m in re.finditer(r"\n", window)
            if m.end() >= min_pos
        ]
        if newline_candidates:
            return newline_candidates[-1]

        comma_candidates = [
            m.end()
            for m in re.finditer(r"[，,]", window)
            if m.end() >= min_pos
        ]
        if comma_candidates:
            return comma_candidates[-1]

        return len(window)

    def _normalize_text(self, text: str) -> str:
        """基础文本清洗，避免破坏金融数字和表格"""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)
        return text.strip()

    # ============================================
    # chunk 构造
    # ============================================

    def _build_chunk(
        self,
        doc_id: str,
        chunk_id: str,
        chunk_type: str,
        content: str,
        doc_metadata: Dict,
        extra: Optional[Dict] = None,
    ) -> Dict:
        """构造标准 chunk 结构"""
        chunk = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "doc_title": doc_metadata.get("doc_title", ""),
            "doc_type": doc_metadata.get("doc_type", "unknown"),
            "source_type": doc_metadata.get("source_type", "unknown"),
            "company": doc_metadata.get("company", ""),
            "company_short": doc_metadata.get("company_short", ""),
            "year": doc_metadata.get("year", ""),
            "period": doc_metadata.get("period", ""),
            "source_file": doc_metadata.get("source_file", ""),
            "chunk_type": chunk_type,
            "content": content,
            "char_len": len(content),
        }

        if extra:
            chunk.update(extra)

        return chunk


# ============================================
# 文件读写
# ============================================

def find_markdown_files(input_dir: Path) -> List[Path]:
    """查找 parsed 目录下所有 md 文件"""
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    md_files = sorted(input_dir.glob("*.md"))

    if not md_files:
        raise FileNotFoundError(f"目录下没有找到 .md 文件: {input_dir}")

    return md_files


def write_jsonl(chunks: List[Dict], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def read_jsonl(input_file: Path) -> List[Dict]:
    rows = []
    with input_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ============================================
# 统计与报告
# ============================================

def summarize_chunks(chunks: List[Dict]) -> Dict:
    if not chunks:
        return {
            "total_chunks": 0,
            "text_chunks": 0,
            "table_chunks": 0,
            "avg_len": 0,
            "min_len": 0,
            "max_len": 0,
        }

    lengths = [c["char_len"] for c in chunks]
    text_chunks = [c for c in chunks if c["chunk_type"] == "text"]
    table_chunks = [c for c in chunks if c["chunk_type"] == "table"]

    return {
        "total_chunks": len(chunks),
        "text_chunks": len(text_chunks),
        "table_chunks": len(table_chunks),
        "avg_len": round(sum(lengths) / len(lengths), 2),
        "min_len": min(lengths),
        "max_len": max(lengths),
    }


def write_experiment_report(results: List[Dict], report_file: Path) -> None:
    lines = []

    lines.append("# 切块策略对比实验")
    lines.append("")
    lines.append("## 实验设置")
    lines.append("")
    lines.append("- 表格保护策略：`[TABLE_START] ... [TABLE_END]` 整体保留，不参与普通文本切分")
    lines.append("- 每篇文档单独输出 jsonl 到 `data/chunks/by_doc/`")
    lines.append("- 同一组 chunk 参数下，所有文档合并输出到 `data/chunks/all_cs*_ov*.jsonl`")
    lines.append("- 新增精简 document metadata，但 metadata 不参与向量化")
    lines.append("")

    lines.append("## 切块结果统计")
    lines.append("")
    lines.append("| chunk_size | overlap | 文档数 | 总chunk数 | 文本chunk数 | 表格chunk数 | 平均长度 | 最短长度 | 最长长度 | 合并文件 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    for r in results:
        lines.append(
            f"| {r['chunk_size']} "
            f"| {r['overlap']} "
            f"| {r['doc_count']} "
            f"| {r['total_chunks']} "
            f"| {r['text_chunks']} "
            f"| {r['table_chunks']} "
            f"| {r['avg_len']} "
            f"| {r['min_len']} "
            f"| {r['max_len']} "
            f"| `{r['merged_output_file']}` |"
        )

    lines.append("")
    lines.append("## Metadata 说明")
    lines.append("")
    lines.append("每个 chunk 包含以下文档级 metadata：")
    lines.append("")
    lines.append("- `doc_title`")
    lines.append("- `doc_type`")
    lines.append("- `source_type`")
    lines.append("- `company`")
    lines.append("- `company_short`")
    lines.append("- `year`")
    lines.append("- `period`")
    lines.append("- `source_file`")
    lines.append("")
    lines.append("metadata 当前只用于过滤、评测和展示，不拼接进 embedding 文本。")
    lines.append("")

    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text("\n".join(lines), encoding="utf-8")


# ============================================
# 批量处理
# ============================================

def run_chunking_once(
    input_dir: Path,
    output_dir: Path,
    chunk_size: int,
    overlap: int,
    save_by_doc: bool = True,
    metadata_overrides: Optional[Dict[str, Dict]] = None,
) -> Dict:
    md_files = find_markdown_files(input_dir)
    metadata_overrides = metadata_overrides or {}

    chunker = TableAwareChunker(
        chunk_size=chunk_size,
        overlap=overlap,
        min_chunk_size=30,
    )

    by_doc_dir = output_dir / "by_doc"
    all_chunks = []

    print("=" * 80)
    print(f"切块配置: chunk_size={chunk_size}, overlap={overlap}")
    print(f"输入目录: {input_dir}")
    print(f"文档数量: {len(md_files)}")
    print("=" * 80)

    for md_file in md_files:
        doc_id = md_file.stem
        text = md_file.read_text(encoding="utf-8")
        doc_metadata = extract_doc_metadata(md_file, text, metadata_overrides)

        chunks = chunker.chunk_text(
            text=text,
            doc_id=doc_id,
            doc_metadata=doc_metadata,
        )
        all_chunks.extend(chunks)

        if save_by_doc:
            by_doc_file = by_doc_dir / f"{doc_id}_cs{chunk_size}_ov{overlap}.jsonl"
            write_jsonl(chunks, by_doc_file)

        stat = summarize_chunks(chunks)
        print(
            f"{doc_id}: "
            f"doc_type={doc_metadata.get('doc_type')} | "
            f"source_type={doc_metadata.get('source_type')} | "
            f"company={doc_metadata.get('company_short') or doc_metadata.get('company')} | "
            f"year={doc_metadata.get('year')} | "
            f"period={doc_metadata.get('period')} | "
            f"chunks={stat['total_chunks']} | "
            f"text={stat['text_chunks']} | "
            f"table={stat['table_chunks']} | "
            f"avg_len={stat['avg_len']}"
        )

    merged_output_file = output_dir / f"all_cs{chunk_size}_ov{overlap}.jsonl"
    write_jsonl(all_chunks, merged_output_file)

    total_stat = summarize_chunks(all_chunks)
    total_stat.update({
        "chunk_size": chunk_size,
        "overlap": overlap,
        "doc_count": len(md_files),
        "merged_output_file": str(merged_output_file),
    })

    print("-" * 80)
    print(f"合并文件已保存: {merged_output_file}")
    print(
        f"总 chunks={total_stat['total_chunks']}, "
        f"text={total_stat['text_chunks']}, "
        f"table={total_stat['table_chunks']}, "
        f"avg_len={total_stat['avg_len']}"
    )
    print("")

    return total_stat


def run_default_chunking(
    input_dir: Path,
    output_dir: Path,
    experiment_dir: Path,
    chunk_size: int,
    overlap: int,
    save_by_doc: bool,
    metadata_overrides: Dict[str, Dict],
) -> None:
    result = run_chunking_once(
        input_dir=input_dir,
        output_dir=output_dir,
        chunk_size=chunk_size,
        overlap=overlap,
        save_by_doc=save_by_doc,
        metadata_overrides=metadata_overrides,
    )

    report_file = experiment_dir / "chunking_results.md"
    write_experiment_report([result], report_file)

    print(f"切块统计报告已保存: {report_file}")


def run_chunking_experiments(
    input_dir: Path,
    output_dir: Path,
    experiment_dir: Path,
    chunk_sizes: List[int],
    overlaps: List[int],
    save_by_doc: bool,
    metadata_overrides: Dict[str, Dict],
) -> None:
    results = []

    for chunk_size in chunk_sizes:
        for overlap in overlaps:
            if overlap >= chunk_size:
                print(f"跳过非法组合: chunk_size={chunk_size}, overlap={overlap}")
                continue

            result = run_chunking_once(
                input_dir=input_dir,
                output_dir=output_dir,
                chunk_size=chunk_size,
                overlap=overlap,
                save_by_doc=save_by_doc,
                metadata_overrides=metadata_overrides,
            )
            results.append(result)

    report_file = experiment_dir / "chunking_results.md"
    write_experiment_report(results, report_file)

    print(f"切块对比实验完成，报告已保存: {report_file}")


# ============================================
# CLI
# ============================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Table-aware chunker for FinRAG")

    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(PROJECT_ROOT / "data/parsed"),
        help="输入 Markdown 目录",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "data/chunks"),
        help="chunks 输出目录",
    )

    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=str(PROJECT_ROOT / "experiments"),
        help="实验报告输出目录",
    )

    parser.add_argument(
        "--metadata_file",
        type=str,
        default=str(PROJECT_ROOT / "data/doc_metadata.json"),
        help="可选人工 metadata 覆盖文件",
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1024,
        help="默认模式使用的 chunk_size",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="默认模式使用的 overlap",
    )

    parser.add_argument(
        "--run_experiments",
        action="store_true",
        help="开启后执行 3x3 切块实验",
    )

    parser.add_argument(
        "--chunk_sizes",
        type=int,
        nargs="+",
        default=[256, 512, 1024],
        help="实验模式 chunk_size 列表",
    )

    parser.add_argument(
        "--overlaps",
        type=int,
        nargs="+",
        default=[50, 100, 200],
        help="实验模式 overlap 列表",
    )

    parser.add_argument(
        "--no_save_by_doc",
        action="store_true",
        help="只保存合并后的 all jsonl，不保存每篇文档 jsonl",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    experiment_dir = Path(args.experiment_dir)

    metadata_file = Path(args.metadata_file) if args.metadata_file else None
    metadata_overrides = load_metadata_overrides(metadata_file)

    save_by_doc = not args.no_save_by_doc

    if args.run_experiments:
        run_chunking_experiments(
            input_dir=input_dir,
            output_dir=output_dir,
            experiment_dir=experiment_dir,
            chunk_sizes=args.chunk_sizes,
            overlaps=args.overlaps,
            save_by_doc=save_by_doc,
            metadata_overrides=metadata_overrides,
        )
    else:
        run_default_chunking(
            input_dir=input_dir,
            output_dir=output_dir,
            experiment_dir=experiment_dir,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            save_by_doc=save_by_doc,
            metadata_overrides=metadata_overrides,
        )


if __name__ == "__main__":
    main()
