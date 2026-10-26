from __future__ import annotations

import re

LETTERS = "ABCDEFGHIJ"


def find_boxed(text: str, last: bool = False) -> str | None:
    if not text:
        return None
    starts = [m.end() for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text)]
    if not starts:
        return None
    s = starts[-1] if last else starts[0]
    depth, i = 1, s
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[s:i]
        i += 1
    return text[s:]


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    for substr in substrs[1:]:
        new_str += "\\frac"
        if len(substr) > 0 and substr[0] == "{":
            new_str += substr
        else:
            if len(substr) < 2:
                return string
            a, b = substr[0], substr[1]
            if b != "{":
                new_str += "{" + a + "}{" + b + "}" + substr[2:]
            else:
                new_str += "{" + a + "}" + b + substr[2:]
    return new_str


def _fix_a_slash_b(string: str) -> str:
    parts = string.split("/")
    if len(parts) != 2:
        return string
    try:
        a, b = int(parts[0]), int(parts[1])
    except ValueError:
        return string
    if string != f"{a}/{b}":
        return string
    return "\\frac{" + str(a) + "}{" + str(b) + "}"


def _remove_right_units(string: str) -> str:
    if "\\text{ " in string:
        return string.split("\\text{ ")[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if len(split) > 0 and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def _unwrap_text(string: str) -> str:
    return re.sub(r"\\(?:text|mbox|textbf|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}", r"\1", string)


def strip_math(string: str) -> str:
    s = str(string).strip()
    s = s.replace("\n", "").replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", "")
    s = s.replace("\\\\", "\\")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("°", "")
    s = s.replace("\\$", "").replace("$", "")
    s = _remove_right_units(s)
    s = _unwrap_text(s)
    s = s.replace("\\%", "").replace("%", "")
    s = re.sub(r"(\d),(\d\d\d)", r"\1\2", s)
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if not s:
        return s
    if s[0] == ".":
        s = "0" + s
    if len(s.split("=")) == 2 and len(s.split("=")[0]) <= 2:
        s = s.split("=")[1]
    s = re.sub(r"^[a-zA-Z]\\in", "", s)
    s = _fix_sqrt(s)
    s = s.replace(" ", "")
    s = _fix_fracs(s)
    if s == "0.5":
        s = "\\frac{1}{2}"
    s = _fix_a_slash_b(s)
    s = s.rstrip(".")
    return s


def _to_float(s: str):
    s = str(s).replace(",", "").replace("$", "").strip().rstrip(".")
    m = re.fullmatch(r"\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except ZeroDivisionError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def math_equiv(pred: str, gold: str) -> bool:
    if pred is None or gold is None:
        return False
    a, b = strip_math(pred), strip_math(gold)
    if a == b and a != "":
        return True
    fa, fb = _to_float(a), _to_float(b)
    if fa is not None and fb is not None:
        return abs(fa - fb) <= 1e-6 * max(1.0, abs(fb))
    la, lb = re.sub(r"[()\s]", "", a).upper(), re.sub(r"[()\s]", "", b).upper()
    return len(lb) == 1 and la == lb


_NUM = r"-?\d[\d,]*(?:\.\d+)?"


def _first_line(text: str) -> str:
    for line in str(text).split("\n"):
        if line.strip():
            return line.strip()
    return ""


def _clean_head(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^(?:\*\*)?\s*(?:final\s+answer|answer)\s*(?:\*\*)?\s*[:：]\s*(?:\*\*)?", "", s, flags=re.I)
    s = s.strip().strip("*").strip()
    return s


def _mc_natural(text: str, n_options: int) -> str:
    allowed = LETTERS[:max(2, min(n_options or 10, 10))]
    pats = [rf"answer\s*(?:is|:|：)?\s*[:：]?\s*\**\s*\(?([{allowed}])\)?(?![A-Za-z])",
            rf"\*\*\s*\(?([{allowed}])\)?\s*[.:)]?\s*\*\*",
            rf"(?:option|choice)\s*\(?([{allowed}])\)?(?![A-Za-z])"]
    best = None
    for p in pats:
        for mt in re.finditer(p, text, flags=re.I):
            if best is None or mt.start() > best[0]:
                best = (mt.start(), mt.group(1).upper())
    if best:
        return best[1]
    lines = [l for l in text.split("\n") if l.strip()]
    if lines:
        hits = re.findall(rf"(?<![A-Za-z])\(?([{allowed}])\)?(?![A-Za-z])", lines[-1])
        if hits:
            return hits[-1]
    return ""


def _norm_text(s: str) -> str:
    s = str(s).lower().replace(",", " ")
    s = re.sub(r"\s+", " ", s).strip().strip(".").strip("`").strip()
    if s and re.fullmatch(r"[\[\]\(\)\{\}<>\s]+", s):
        s = re.sub(r"\s+", "", s)
    return s


def extract(text: str, mode: str, n_options: int = 10, where: str = "probe") -> str:
    if text is None:
        return ""
    text = str(text).replace("Ġ", " ").replace("Ċ", "\n")
    if where == "natural":
        box = find_boxed(text, last=True)
        if box is not None:
            seg = box
        else:
            parts = re.split(r"final\s+answer\s*\**\s*[:：]", text, flags=re.I)
            if len(parts) > 1:
                seg = _first_line(parts[-1].lstrip("*: \t\n"))
            elif mode == "numeric":
                nums = re.findall(_NUM, text)
                seg = nums[-1] if nums else ""
            elif mode == "mc":
                seg = _mc_natural(text, n_options)
            else:
                lines = [l.strip() for l in text.split("\n") if l.strip()]
                seg = lines[-1] if lines else ""
    else:
        box = find_boxed(text, last=False)
        seg = box if box is not None else _clean_head(_first_line(text))

    if mode == "numeric":
        seg = re.sub(r"\\[!,;: ]|\{,\}", "", seg)
        m = re.search(_NUM, seg.replace("\\$", "").replace("$", ""))
        return m.group(0).replace(",", "").rstrip(".") if m else ""
    if mode == "mc":
        allowed = LETTERS[:max(2, min(n_options, 10))]
        s = re.sub(r"\\text\{([^{}]*)\}", r"\1", seg)
        m = re.search(rf"(?<![A-Za-z])\(?([{allowed}])\)?(?![A-Za-z])", s)
        return m.group(1) if m else ""
    if mode == "text":
        return _norm_text(seg)
    s = seg.strip()
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    s = re.sub(r"^\\\(|\\\)$", "", s).strip()
    s = re.sub(r"^\\\[|\\\]$", "", s).strip()
    return s.rstrip(".").strip()


def is_correct(pred: str, gold: str, mode: str, alts=()) -> bool:
    if pred is None or pred == "":
        return False
    if alts:
        return any(is_correct(pred, g, mode) for g in (gold, *alts))
    if mode == "numeric":
        fa, fb = _to_float(pred), _to_float(gold)
        return fa is not None and fb is not None and abs(fa - fb) <= 1e-6 * max(1.0, abs(fb))
    if mode == "mc":
        return pred.strip().upper() == str(gold).strip().upper()
    if mode == "text":
        return _norm_text(pred) == _norm_text(gold)
    return math_equiv(pred, gold)


def answer_key(pred: str, mode: str) -> str:
    if not pred:
        return ""
    if mode == "math":
        return strip_math(pred)
    if mode == "numeric":
        f = _to_float(pred)
        return "" if f is None else repr(round(f, 6))
    return pred.strip().lower()


def grade(text: str, gold: str, mode: str, n_options: int = 10, where: str = "probe", alts=()):
    pred = extract(text, mode, n_options=n_options, where=where)
    return answer_key(pred, mode), int(is_correct(pred, gold, mode, alts))


if __name__ == "__main__":
    assert find_boxed("so \\boxed{\\frac{1}{4}} done") == "\\frac{1}{4}"
    assert math_equiv("\\dfrac{3}{2}", "\\frac{3}{2}")
    assert math_equiv("1/4", "\\frac14")
    assert math_equiv("3\\sqrt{13}", "3\\sqrt{13}")
    assert math_equiv("\\left(3\\frac{\\pi}{2}\\right)", "\\left(3\\frac{\\pi}{2}\\right)")
    assert grade(" \\boxed{\\frac{1}{8}}\n\nBecause", "\\frac{1}{8}", "math")[1] == 1
    assert grade(" 18\n\nExplanation: 3*6=18, minus 2", "18", "numeric")[1] == 1
    assert grade(" $1,200.", "1200", "numeric")[1] == 1
    assert grade(" B", "B", "mc", 10)[1] == 1
    assert grade(" (C) because", "C", "mc", 4)[1] == 1
    assert grade("reasoning...\nFinal answer: 42", "42", "numeric", where="natural")[1] == 1
    assert grade(" ) ] }", ") ] }", "text")[1] == 1
    assert grade(" gloucester, raytheon, slurp", "gloucester raytheon slurp", "text")[1] == 1
    assert grade(" [ [ ] ]", "] ]", "text", alts=("[ [ ] ]",))[1] == 1
    assert grade(" [[ ]]", "] ]", "text", alts=("[ [ ] ]",))[1] == 1
    assert grade(" [ ] ]", "] ]", "text", alts=("[ [ ] ]",))[1] == 0
    assert grade("A car goes... so the answer is (C).", "C", "mc", 10, where="natural")[1] == 1
    assert grade("because of X. \n\n**Final answer**: G", "G", "mc", 10, where="natural")[1] == 1
    assert grade("---\n\n**Final answer:**  \nG. No chance", "G", "mc", 10, where="natural")[1] == 1
    assert grade("so = \\boxed{\\$1,\\!430}", "1430", "numeric", where="natural")[1] == 1
    assert grade("A. first\nB. second\n**Answer: D**", "D", "mc", 10, where="natural")[1] == 1
    print("grading_v2 self-test OK")
