from __future__ import annotations

import json
import re
import urllib.request
from typing import Any, Dict, Tuple

DEFAULT_CHAT_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL_NAME = "GLM-4-Flash"
KEYWORD_MODEL_NAME = "glm-4.7"

_QUOTE_NORMALIZE_BASE = str.maketrans(
    {
        ord("“"): '"',
        ord("”"): '"',
        ord("„"): '"',
        ord("‟"): '"',
        ord("«"): '"',
        ord("»"): '"',
        ord("‹"): '"',
        ord("›"): '"',
        ord("‘"): "'",
        ord("’"): "'",
        ord("‚"): "'",
        ord("‛"): "'",
        ord("「"): '"',
        ord("」"): '"',
        ord("『"): '"',
        ord("』"): '"',
        ord("《"): '"',
        ord("》"): '"',
        ord("〈"): '"',
        ord("〉"): '"',
        ord("＂"): '"',
        ord("＇"): "'",
    }
)
_QUOTE_NORMALIZE_ALL = str.maketrans(
    {
        ord('"'): '"',
        ord("'"): '"',
        ord("“"): '"',
        ord("”"): '"',
        ord("„"): '"',
        ord("‟"): '"',
        ord("«"): '"',
        ord("»"): '"',
        ord("‹"): '"',
        ord("›"): '"',
        ord("‘"): '"',
        ord("’"): '"',
        ord("‚"): '"',
        ord("‛"): '"',
        ord("「"): '"',
        ord("」"): '"',
        ord("『"): '"',
        ord("』"): '"',
        ord("《"): '"',
        ord("》"): '"',
        ord("〈"): '"',
        ord("〉"): '"',
        ord("＂"): '"',
        ord("＇"): '"',
    }
)


def post_json(url: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_review_prompt(doc: Dict[str, Any], query: str) -> str:
    title = doc.get("title", "")
    content = doc.get("content", "") or ""
    return (
        "You are a reviewer. Only quote exact sentences from the article. "
        "If the article does not answer the question, return an empty quote.\n\n"
        f"Question: {query}\n\n"
        f"Title: {title}\n"
        f"Article: {content}\n\n"
        'Return JSON only: {"quote": "...", "score": 0-10}'
    )


def build_keyword_prompt(query: str, max_keywords: int) -> str:
    return (
        "你是一个研究助理，需要把问题扩展成用于检索的概念集合。\n"
        "- 提取核心本质、隐含约束、外延相关概念。\n"
        "- 可包含：基本概念、相关领域、对照/反义概念（如有帮助）。\n"
        "- 若是关系/情感类问题，优先包含 1-2 个核心情感/价值概念（如：爱、信任、尊重、安全感、亲密、边界），选择与问题最相关者。\n"
        f"- 正常返回 3-5 个关键词，最多 {max_keywords} 个。\n"
        "- 优先具体概念，避免空泛词；关键词以中文为主。\n"
        "- 优先从下方“候选关键词库”中选取最相关项；必要时可补充少量新词。\n"
        "- 只输出 JSON。\n\n"
        "候选关键词库：\n"
        "A: AI, 阿拉伯, 爱, 爱不生债, 爱不思恶, 爱国, 爱情, 安全, 安全感, 安慰, 傲慢\n"
        "B: 帮派, 榜样, 帮助, 报恩, 暴力, 报应, 抱怨, 悲观/乐观, 悲剧, 备灾, 崩溃, 比较, 表白, 表达, 表达能力, 标签化, 表演, 博士, 剥削, 博弈, 不卑不亢, 不合理, 不计较人之恶, 不可预测性, 补强, 不求回报, 补弱/补强, 不朽, 不以恶为业\n"
        "C: CARITAS, 财富, 财富观, 材料, 财务, 财务管理, 残酷, 策略, 忏悔, 产品, 产业, 常识, 吵架, 潮流, 超视觉, 撤退, 惩罚, 成功, 成功税, 惩戒, 承诺, 诚实, 城市规划, 成熟, 成长, 耻, 宠爱, 崇拜, 冲动, 冲突, 仇恨, 出行, 揣测, 传承, 传道, 传染病, 传统, 创新, 创业, 创造力, 创作, 慈悲, 慈善, 聪明, 从众, 脆弱, 存在感, 错得对\n"
        "D: 大方, 大过滤器, 打鼾, 代际差异, 代价, 道, 道德, 道德主义, 道理, 道歉, 等待, 地方菜, 帝国主义, 地理, 底气, 敌人, 地狱, 电影, 调查, 调研, 钓鱼, 顶层设计, 定罪, 洞察力, 动画, 懂事, 动物, 度, 毒打, 独立, 读书, 独特, 短视频, 对事不对人\n"
        "E: 俄罗斯, 俄乌战争, 恶意\n"
        "F: 法国, 法律, 发明家, 反驳, 反脆弱, 反对, 反社会, 反思, 繁文缛节, 翻译, 反抑郁, 反自欺, 房地产, 方法论, 放下, 放心, 放纵, 废话, 奋斗, 分类, 愤怒, 分手, 讽刺, 风流, 风险管理, 服从, 复盘, 福气, 服饰, 服务, 负债, 服装\n"
        "G: 改变, 概念, 感谢, 高贵, 告密, 高手, 高铁, 格局, 个人成长, 功夫, 公关, 攻击, 攻击性, 弓箭, 工具, 公平, 共情, 共情力, 公权力, 共识, 公司, 公私, 公务员, 工业, 工业化, 工艺, 公正, 公众人物, 工作, 沟通, 孤独, 辜负, 孤立, 股票, 骨气, 股市, 观点, 管理, 广东, 规划, 归因, 贵族, 果断, 国际关系, 国际贸易, 国际政治, 国家战略, 过失观\n"
        "H: 韩国, 汉语, 汉字, 豪华, 好奇心, 好书, 合法性, 合规, 合伙人, 合理, 和平, 合作, 黑色幽默, 黑社会, 狠人, 宏大叙事, 后悔, 呼吸, 怀疑, 怀疑的艺术, 换电, 幻灭, 绘画, 会议纪要, 婚恋观, 婚姻, 火锅, 活死人\n"
        "J: 羁绊, 基本功, 基本技能, 基层, 基础科学, 嫉妒, 基督教, 饥饿, 及格主义, 计划, 计划经济, 计划生育, 机会, 基建, 疾控, 记录, 纪律, 纪念, 机器人, 技术, 技术解决, 祭祀, 集体, 集体主义, 加班, 家规, 驾驶, 家庭, 家庭主妇, 价值, 价值观, 家族, 家族构建, 坚持, 坚定, 简洁, 健康, 建模, 坚强, 坚韧, 健身, 见义勇为, 见义智为, 建筑, 将错就错, 讲道理, 焦虑, 交通, 交往准则, 教训, 教养, 交易, 教育, 教育家, 阶层, 戒律, 洁癖, 介意, 节奏, 矜持, 进攻, 进攻精神, 进化, 尽力, 金融, 紧张, 经济, 经济独立, 精神病, 净收益, 净输出, 静心, 经验, 精英, 竞争, 精致, 酒店, 就事论事, 拒绝, 决策, 决斗, 绝望, 决心, 军事, 君主\n"
        "K: 开心, 开源, 考验, 可持续性, 科技, 科普, 科学, 科研, 可预测, 克制, 恐惧, 控制欲, 苦难, 夸奖, 会计, 快乐, 宽容, 宽恕, 狂妄, 旷野, 愧疚, 困难\n"
        "L: 浪漫, 劳动, 老实, 乐观, 冷启动, 立场, 立法, 理工科, 利己, 理解, 礼貌, 历史, 历史观, 历史遗留, 利他, 礼物, 理想, 理性, 礼仪, 利用, 吏治, 恋爱, 怜悯, 练习, 粮食, 两性差异, 两性关系, 聊天, 疗愈, 领导, 领袖, 留学, 伦理, 逻辑, 旅游\n"
        "M: 买卖, 猫, 矛盾, 冒犯, 冒险, 美, 美德, 美国, 魅力, 美貌, 美人, 美学, 迷信, 面试, 面子, 敏感, 民主, 民族, 名分, 命运, 目标, 慕强\n"
        "N: 内务, 内务班子, 内向, 能力, 能源, 年轻人, 农村, 农业, 努力, 奴性, 女权, 女权主义, 女性, 女性独立\n"
        "O: 偶像, 偶像崇拜\n"
        "P: POETIC IRONY, 攀比, 判断力, 叛逆, 叛徒, 陪伴, 培养, 朋友, 批评, 偏见, 偏执, 票选, 品牌, 贫穷, 平等, 评价权, 平庸, 普通人\n"
        "Q: 汽车, 期待, 祈祷, 企管, 欺骗, 祈求, 歧视, 骑士精神, 企业家, 企业文化, 契约, 气质, 谦卑, 欠债, 强迫, 强势, 强者/弱者, 侵犯, 勤俭, 侵略, 亲密关系, 亲子关系, 亲子教育, 情报, 情报分析, 情商, 青少年, 轻视, 倾听, 情绪, 情绪管理, 情绪价值, 求教, 求救, 求职, 求助, 驱动力, 祛魅, 取名, 趋势, 趋同, 权柄, 拳击, 权利, 权力, 全球化, 权威\n"
        "R: 热爱, 人, 人才, 仁慈, 认错, 仁德, 人格, 人工智能, 人际关系, 人际交往, 人己权界, 人口, 人类, 忍耐, 人权, 人设, 人神分野, 人生观, 人生规划, 人生价值, 人性, 人形机器人, 认知, 认知战, 日本, 软弱, 弱者\n"
        "S: SOP, 沙漠, 善恶, 善良, 赏罚分明, 伤害, 上进, 尚武, 商业, 商业道德, 商业伦理, 商业逻辑, 商业思维, 上瘾, 奢侈品, 社会发展, 社会化, 社会伦理, 社会评价, 社会性, 社会责任, 社会治理, 社会主义, 设计, 社交, 社交策略, 社群, 神话, 审美, 审美观, 审判, 审判权, 绅士, 神性, 神性享乐, 神学, 生产工具, 生产关系, 生存, 生活方式, 生活水平, 生命, 生死观, 生态环境, 生物, 生育, 市场, 释怀, 实践, 时间, 时间管理, 世界观, 世界史, 视觉, 失恋, 使命感, 识人, 师生关系, 事实, 实事求是, 视死如归, 师徒, 失望, 食物, 实验, 失业, 事业, 石油, 书法, 书籍, 熟能生巧, 舒适区, 庶务管理, 数学, 输赢, 摔跤, 睡眠, 顺势而为, 思辨, 私德, 司法, 思考, 死士, 死亡, 思维, 思维导图, 思想, 松弛感, 俗, 诉求, 俗事, 素问, 碎片化学习\n"
        "T: 踏实, 态度, 台湾, 贪婪, 谈判, 逃避, 讨论, 特权, 提问, 提意见, 体育, 天赐的代价, 天赋, 天人合一, 天意, 条理, 挑剔, 挑战, 统计学, 痛苦, 同情, 同志, 投降, 投资, 团队精神, 颓废, 推理, 退休, 脱口秀, 脱敏, 脱贫, 妥协\n"
        "W: 外号, 外卖, 外星文明, 玩具, 完美, 王天下, 未成年人, 维权, 为人处世, 威慑, 危险, 威胁, 文笔, 文化, 文化自信, 文明, 温柔, 文学, 文艺作品, 稳重, 文字, 我, 物化, 武力, 物理, 无聊, 武器, 侮辱, 诬陷, 悟性, 物质, 无知, 物质条件\n"
        "X: 习惯, 喜欢, 细节, 喜剧, 牺牲, 系统化, 希望, 狎, 侠义, 现代性, 闲鱼, 香火情, 享乐, 享乐主义, 向上管理, 相声, 向死而生, 乡愿, 孝, 消费, 消费观, 效率, 销售, 孝顺, 小说, 校园霸凌, 写作, 新冠, 心理, 心理学, 心理咨询, 心流, 新能源, 芯片, 信任, 欣赏, 新闻, 信息, 信心, 信仰, 信用, 信誉, 信源管理, 性, 性别认知, 行动, 幸福, 性感, 性教育, 性命相托, 性侵, 性骚扰, 形象管理, 刑讯, 刑侦, 雄心, 羞耻, 休息, 需求, 虚荣, 选举, 选择, 学历, 学术, 学习, 训练, 殉情\n"
        "Y: 目光, 严厉, 颜色革命, 严肃, 厌学, 演员, 养老, 阳谋, 谣言, 野外生存, 野心, 义, 义愤, 意见, 依赖, 伊朗, 移民, 意识, 意识形态, 艺术, 艺术创作, 艺术批评, 艺术评论, 艺术修养, 义务, 义乌, 意义, 抑郁, 抑郁症, 瘾, 因材施教, 印度, 饮食, 隐私, 阴阳怪气, 音乐, 英国, 应急, 应试教育, 影响力, 营销, 英雄, 英勇, 英语, 勇敢, 勇气, 永生, 友好, 幽默, 有趣, 游戏, 优先级, 优越感, 幼稚, 愚蠢, 预告, 娱乐, 舆论, 舆论战, 欲望, 语文, 语言, 预言, 宇宙, 愿赌服输, 元规则, 原画, 元技能, 原谅, 原生家庭, 原则, 圆周率, 阅读, 约会, 阅历, 运动, 运气\n"
        "Z: 赞美, 早恋, 择偶, 责任, 择善固执, 赠礼, 战斗, 展会, 战略, 战略规划, 战争, 张力, 哲学, 真诚, 真理, 震慑, 珍惜, 政策, 正常, 正反馈, 正念, 整齐, 政权, 正义, 政治, 政治正确, 职场, 支持, 知错能改, 智慧, 直觉, 智能驾驶, 智能制造, 知识, 志向, 执行力, 秩序, 职业, 职业规划, 职业伦理, 质疑, 制造业, 执着, 忠诚, 中国, 种族矛盾, 种族歧视, 种族主义, 祝福, 诛心, 专业, 自卑, 资本, 资本家, 资本主义, 资产, 自律, 自媒体, 子女教育, 自欺欺人, 自治, 自强, 自然法, 自杀, 自私, 字体, 自卫, 自我, 自我辩护, 自我认知, 自信, 自省, 自由, 自愿, 自证预言, 自尊, 宗教, 宗族, 租房, 组织管理, 组织伦理, 诅咒, 罪, 尊敬, 尊师重道, 尊严, 尊重, 作息\n\n"
        "示例：\n"
        "问题：恋人在争吵的冷静期间，适合做些什么？\n"
        "只输出 JSON：{\"keywords\": [\"爱\", \"冷静\", \"情绪管理\", \"沟通\", \"亲密关系\", \"边界\"]}\n\n"
        f"问题：{query}\n\n"
        '只输出 JSON：{"keywords": ["...", "..."]}'
    )


def build_review_payload(
    doc: Dict[str, Any], query: str, model: str = MODEL_NAME
) -> Dict[str, Any]:
    prompt = build_review_prompt(doc, query)
    return {
        "model": model,
        "agentic": False,
        "temperature": 0.2,
        "max_tokens": 512,
        "messages": [
            {
                "role": "system",
                "content": "You are a reviewer. Use only exact quotes from the article.",
            },
            {"role": "user", "content": prompt},
        ],
    }


def build_keyword_payload(
    query: str, max_keywords: int, model: str = MODEL_NAME
) -> Dict[str, Any]:
    prompt = build_keyword_prompt(query, max_keywords)
    return {
        "model": model,
        "agentic": False,
        "thinking": {"type": "disabled"},
        "temperature": 0.2,
        "max_tokens": 256,
        "messages": [
            {
                "role": "system",
                "content": "Extract search keywords and return JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
    }


def extract_message_content(response: Dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    if isinstance(message, dict):
        return message.get("content", "") or ""
    return choices[0].get("text", "") or ""


def extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def parse_keywords_response(
    response: Dict[str, Any], max_keywords: int
) -> Dict[str, Any]:
    content_text = extract_message_content(response)
    parsed = extract_json(content_text)
    keywords = []
    if isinstance(parsed, dict):
        raw_list = parsed.get("keywords") or parsed.get("key_terms") or []
        if isinstance(raw_list, list):
            keywords = [str(item).strip() for item in raw_list if str(item).strip()]

    seen = set()
    deduped = []
    for item in keywords:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
        if len(deduped) >= max_keywords:
            break

    return {
        "keywords": deduped,
        "raw_text": content_text,
        "parsed": parsed,
    }


def _find_with_map(content: str, quote: str, trans: Dict[int, str]) -> int:
    if not quote:
        return -1
    norm_content = content.translate(trans)
    norm_quote = quote.translate(trans)
    return norm_content.find(norm_quote)


def _normalize_wo_space(text: str, trans: Dict[int, str]) -> Tuple[str, List[int]]:
    normalized = []
    index_map: List[int] = []
    for idx, ch in enumerate(text):
        if ch.isspace():
            continue
        norm_ch = ch.translate(trans)
        if norm_ch.isspace():
            continue
        normalized.append(norm_ch)
        index_map.append(idx)
    return "".join(normalized), index_map


def locate_quote(content: str, quote: str) -> Tuple[int, int, str, str]:
    if not quote:
        return -1, -1, "", "empty"
    start = content.find(quote)
    if start != -1:
        return start, start + len(quote), quote, "exact"

    trimmed = quote.strip()
    if trimmed and trimmed != quote:
        start = content.find(trimmed)
        if start != -1:
            return start, start + len(trimmed), trimmed, "trimmed"

    start = _find_with_map(content, quote, _QUOTE_NORMALIZE_BASE)
    if start != -1:
        return start, start + len(quote), content[start : start + len(quote)], "normalize_curly"

    start = _find_with_map(content, quote, _QUOTE_NORMALIZE_ALL)
    if start != -1:
        return start, start + len(quote), content[start : start + len(quote)], "normalize_all"

    norm_content, index_map = _normalize_wo_space(content, _QUOTE_NORMALIZE_BASE)
    norm_quote, _ = _normalize_wo_space(quote, _QUOTE_NORMALIZE_BASE)
    if norm_quote:
        pos = norm_content.find(norm_quote)
        if pos != -1:
            start = index_map[pos]
            end = index_map[pos + len(norm_quote) - 1] + 1
            return start, end, content[start:end], "normalize_ws"

    norm_content, index_map = _normalize_wo_space(content, _QUOTE_NORMALIZE_ALL)
    norm_quote, _ = _normalize_wo_space(quote, _QUOTE_NORMALIZE_ALL)
    if norm_quote:
        pos = norm_content.find(norm_quote)
        if pos != -1:
            start = index_map[pos]
            end = index_map[pos + len(norm_quote) - 1] + 1
            return start, end, content[start:end], "normalize_ws_all"

    return -1, -1, "", "not_found"


def parse_review_response(doc: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
    content_text = extract_message_content(response)
    parsed = extract_json(content_text)

    quote_raw = str(parsed.get("quote", "")) if parsed else ""
    score = parsed.get("score", 0) if parsed else 0
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        score_value = 0.0

    content = doc.get("content", "") or ""
    start, end, quote, strategy = locate_quote(content, quote_raw)
    if start == -1:
        quote = ""
        start = 0
        end = 0
        score_value = 0.0

    return {
        "quote_raw": quote_raw,
        "quote": quote,
        "quote_start": start,
        "quote_end": end,
        "score": score_value,
        "match_strategy": strategy,
        "raw_text": content_text,
        "parsed": parsed,
    }


def review_doc(doc: Dict[str, Any], query: str, chat_url: str) -> Dict[str, Any]:
    payload = build_review_payload(doc, query, model=MODEL_NAME)
    error = ""
    try:
        response = post_json(chat_url, payload)
    except Exception as exc:
        error = str(exc)
        response = {}

    parsed = parse_review_response(doc, response)

    return {
        "id": doc.get("id", ""),
        "title": doc.get("title", ""),
        "url": doc.get("url", ""),
        "quote": parsed.get("quote", ""),
        "quote_start": parsed.get("quote_start", 0),
        "quote_end": parsed.get("quote_end", 0),
        "score": parsed.get("score", 0.0),
        "error": error,
    }


def extract_keywords(query: str, chat_url: str, max_keywords: int = 10) -> Dict[str, Any]:
    payload = build_keyword_payload(query, max_keywords, model=KEYWORD_MODEL_NAME)
    error = ""
    primary_response: Dict[str, Any] = {}
    fallback_response: Dict[str, Any] = {}
    try:
        response = post_json(chat_url, payload)
    except Exception as exc:
        error = str(exc)
        response = {}

    primary_response = response
    parsed = parse_keywords_response(response, max_keywords=max_keywords)
    if not parsed.get("raw_text"):
        fallback_payload = build_keyword_payload(query, max_keywords, model=MODEL_NAME)
        try:
            response = post_json(chat_url, fallback_payload)
        except Exception as exc:
            if not error:
                error = str(exc)
        else:
            parsed = parse_keywords_response(response, max_keywords=max_keywords)
            payload = fallback_payload
            fallback_response = response
    return {
        "query": query,
        "keywords": parsed.get("keywords", []),
        "raw_text": parsed.get("raw_text", ""),
        "parsed": parsed.get("parsed", {}),
        "payload": payload,
        "response": response,
        "response_primary": primary_response,
        "response_fallback": fallback_response,
        "keyword_model": payload.get("model", ""),
        "error": error,
    }
