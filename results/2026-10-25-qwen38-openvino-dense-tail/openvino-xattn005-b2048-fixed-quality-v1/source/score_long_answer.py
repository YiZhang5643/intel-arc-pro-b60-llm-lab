"""Strict per-field grading of the first JSON answer, including fenced answers."""
import json


def score_answer(output, expected):
    decoder = json.JSONDecoder()
    answer = None
    span = None
    # Grade the first object that mentions an expected key. Never select a later,
    # improved answer after an earlier contradictory one.
    for at, char in enumerate(output):
        if char != "{":
            continue
        try:
            obj, end = decoder.raw_decode(output[at:])
        except ValueError:
            continue
        if isinstance(obj, dict) and set(obj) & set(expected):
            answer, span = obj, (at, at + end)
            break
    matched, wrong, missing = [], {}, []
    for key, value in expected.items():
        if answer is None or key not in answer:
            missing.append(key)
        elif type(answer[key]) is type(value) and answer[key] == value:
            matched.append(key)
        else:
            wrong[key] = {"expected": value, "actual": answer[key]}
    return {"matched": len(matched), "total": len(expected),
            "all_correct": bool(expected) and len(matched) == len(expected),
            "matched_keys": matched, "missing_keys": missing, "wrong": wrong,
            "answer": answer,
            "format_exact_json": span is not None and not (output[:span[0]] + output[span[1]:]).strip()}
