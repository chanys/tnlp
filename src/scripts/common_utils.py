from typing import List

class DotDict(dict):
    def __getattr__(self, attr):
        value = self.get(attr)
        if isinstance(value, dict):
            return DotDict(value)
        return value

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Span():
    def __init__(self, text, token_start, token_end, char_start, char_end, label=None):
        self.text = text
        self.token_start = token_start
        self.token_end = token_end
        self.char_start = char_start
        self.char_end = char_end
        self.label = label


def get_span_using_char_start(spans: List[Span], offset: int):
    for span in spans:
        if span.char_start == offset:
            return span
    return None


def get_span_using_char_end(spans: List[Span], offset: int):
    for span in spans:
        if span.char_end == offset:
            return span
    return None


def find_phrase_offsets(phrase, text):
    return [(i, i + len(phrase)) for i in range(len(text)) if text.startswith(phrase, i)]

