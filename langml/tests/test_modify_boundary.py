# -*- coding: utf-8 -*-

import pytest
from langml.utils import modify_boundary


@pytest.mark.parametrize(
    "inputs,expected",
    [
        (("动隔振平台", "题 主动隔振平台|||"), "主动隔振平台"),
        (("制备型液", "|||制备型液相 1"), "制备型液相")
    ]
)
def test_modify_boundary(inputs, expected):
    target, content = inputs
    result = modify_boundary(target, content)
    assert result == expected
