"""Source-only diagnostic objective; not an official accuracy metric for open VQA."""
import re

from anchor.medeval.evaluate_mixed_vqa_table import answer_token_recall
from anchor.corrected_sgta.evaluate_medheval_answers import evaluate_rows

VERSION = "tx-source-task-utility-v1"


def source_score(row, text, references):
    if str(row.get("split", row.get("role"))) not in {"source", "train", "development", "dev"}:
        raise ValueError("source utility forbids target rows")
    if str(row["id"]).startswith("pathorob-"):
        categories = (
            "tumor tissue", "muscularis propria", "oesophageal mucosa",
            "gastric mucosa", "regressed tumor tissue", "adventitia",
        )
        pattern = r"\b(?:" + "|".join(re.escape(x) for x in sorted(categories, key=len, reverse=True)) + r")\b"
        found = set(re.findall(pattern, text.casefold()))
        return float(len(found) == 1 and next(iter(found)) in {r.casefold().strip() for r in references})
    if row["answer_type"] == "closed":
        return max(float(evaluate_rows([{**row, "gt_ans": ref, "text": text}])["details"][0]["correct"]) for ref in references)
    if row["answer_type"] == "open":
        return max(answer_token_recall(text, ref) for ref in references)
    raise ValueError("unsupported source answer type")
