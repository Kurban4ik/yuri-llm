from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

ACT_TYPES = {"ФЗ", "ГК", "НК", "иное"}


def detect_act_type(filename: str) -> str:
    name = filename.upper()
    if "ФЗ" in name or "FZ" in name or "ФЕДЕРАЛЬНЫЙ ЗАКОН" in name or "ФЕДЕРАЛЬНЫЙ" in name:
        return "ФЗ"
    if "ГК" in name or "GK" in name or "ГРАЖДАНСКИЙ КОДЕКС" in name.upper() or "ГРАЖДАНСКИЙ" in name.upper():
        return "ГК"
    if "НК" in name or "NK" in name or 'НАЛОГОВЫЙ КОДЕКС' in name.upper() or "НАЛОГОВЫЙ" in name.upper():
        return "НК"
    return "иное"


@dataclass
class Clause:
    number: str          # "1)", "а)", "1." …
    full_text: str
    id: str              # будет заполнен в parser


@dataclass
class Part:
    number: str          # "1", "2" …
    clauses: List[Clause] = field(default_factory=list)


@dataclass
class Article:
    number: str
    title: str
    full_text: str = ""
    parts: List[Part] = field(default_factory=list)


@dataclass
class Section:
    name: Optional[str]          # None — если раздел не найден
    articles: List[Article] = field(default_factory=list)


@dataclass
class LegalAct:
    act_name: str
    act_type: str                # ФЗ | ГК | НК | иное
    date: str                    # строка как в тексте
    sections: List[Section] = field(default_factory=list)
    articles: List[Article] = field(default_factory=list)  # плоский список всех статей
