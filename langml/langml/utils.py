# -*- coding: utf-8 -*-

import functools
from typing import List, Optional, Tuple

import jieba

from langml.log import warn


def deprecated_warning(msg='this function is deprecated! it might be removed in a future version.'):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warn(msg)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def modify_boundary(target: str, content: str, expand_range: Optional[int] = 3) -> str:
    """
    分词修正目标字符串的边界
    Args:
        target:       目标字符串
        content:      目标字符串所在的文本
        expand_range: 目标字符串在文本所在位置，向前后扩展的字数，无需太大，因为一个中文词语大约1到4字
    Returns: 使用分词修正目标字符串边界后的字符串
    """

    begin = content.find(target)

    # 在content中无法找到target
    if begin == -1:
        return target

    # 目标字符串在文本所在位置，向前后扩展的上下文
    context = content[begin - expand_range if begin >= expand_range else 0: begin + len(target) + expand_range]

    # 目标字符串在上下文的起始终止位置
    context_start = expand_range if begin >= expand_range else begin
    context_end = context_start + len(target)

    seg_list = list(jieba.cut(context, cut_all=False))

    # 找出target头尾在分词列表的位置
    seg_sum = 0
    seg_start, seg_end = None, None
    for seg_id, seg in enumerate(seg_list):
        seg_range = [x + seg_sum for x in range(len(seg))]
        seg_sum += len(seg)
        if context_start in seg_range:
            seg_start = seg_id
        if context_end in seg_range:
            # 如果原始结尾在分词的区间第一个位置, 那么这个分词区间不需要；# 否则需要这个区间
            if context_end == seg_range[0]:
                seg_end = seg_id
            else:
                seg_end = seg_id + 1

    if seg_start and seg_end:
        new_target = "".join(seg_list[seg_start: seg_end])
    else:
        new_target = target

    return new_target


@deprecated_warning(msg='`rematch` is deprecated, it might be removed in a future version! '
                        'please turn to `Tokenizer.tokens_mapping`.')
def rematch(offsets: List) -> List:
    mapping = []
    for offset in offsets:
        if offset[0] == 0 and offset[1] == 0:
            mapping.append([])
        else:
            mapping.append([i for i in range(offset[0], offset[1])])
    return mapping


def bio_decode(tags: List[str]) -> List[Tuple[int, int, str]]:
    """ Decode BIO tags

    Examples:
    >>> bio_decode(['B-PER', 'I-PER', 'O', 'B-ORG', 'I-ORG', 'I-ORG'])
    >>> [(0, 1, 'PER'), (3, 5, 'ORG')]
    """
    entities = []
    start_tag = None
    for i, tag in enumerate(tags):
        tag_capital = tag.split('-')[0]
        tag_name = tag.split('-')[1] if tag != 'O' else ''
        if tag_capital in ['B', 'O']:
            if start_tag is not None:
                entities.append((start_tag[0], i - 1, start_tag[1]))
                start_tag = None
            if tag_capital == 'B':
                start_tag = (i, tag_name)
        elif tag_capital == 'I' and start_tag is not None and start_tag[1] != tag_name:
            entities.append((start_tag[0], i, start_tag[1]))
            start_tag = None
    if start_tag is not None:
        entities.append((start_tag[0], i, start_tag[1]))
    return entities
