# -*- coding: utf-8 -*-
"""Implementation of the LTLf parser."""
import inspect
import os
from pathlib import Path

from lark import Lark, Transformer

from flloat.helpers import ParsingError
from flloat.ltlf import (
    LTLfEquivalence,
    LTLfImplies,
    LTLfOr,
    LTLfAnd,
    LTLfNot,
    LTLfUntil,
    LTLfRelease,
    LTLfAlways,
    LTLfEventually,
    LTLfNext,
    LTLfWeakNext,
    LTLfTrue,
    LTLfAtomic,
    LTLfFalse,
    LTLfLast,
)
from flloat.parser import CUR_DIR
from flloat.parser.pl import PLTransformer


class LTLfTransformer(Transformer):
    def __init__(self):
        super().__init__()
        self._pl_transformer = PLTransformer()

    def start(self, args):
        assert len(args) == 1
        return args[0]

    def ltlf_formula(self, args):
        assert len(args) == 1
        return args[0]

    def ltlf_equivalence(self, args):
        if len(args) == 1:
            return args[0]
        elif (len(args) - 1) % 2 == 0:
            subformulas = args[::2]
            return LTLfEquivalence(subformulas)
        else:
            raise ParsingError

    def ltlf_implication(self, args):
        if len(args) == 1:
            return args[0]
        elif (len(args) - 1) % 2 == 0:
            subformulas = args[::2]
            return LTLfImplies(subformulas)
        else:
            raise ParsingError

    def ltlf_or(self, args):
        if len(args) == 1:
            return args[0]
        elif (len(args) - 1) % 2 == 0:
            subformulas = args[::2]
            return LTLfOr(subformulas)
        else:
            raise ParsingError

    def ltlf_and(self, args):
        if len(args) == 1:
            return args[0]
        elif (len(args) - 1) % 2 == 0:
            subformulas = args[::2]
            return LTLfAnd(subformulas)
        else:
            raise ParsingError

    def ltlf_until(self, args):
        if len(args) == 1:
            return args[0]
        elif (len(args) - 1) % 2 == 0:
            subformulas = args[::2]
            return LTLfUntil(subformulas)
        else:
            raise ParsingError

    def ltlf_release(self, args):
        if len(args) == 1:
            return args[0]
        elif (len(args) - 1) % 2 == 0:
            subformulas = args[::2]
            return LTLfRelease(subformulas)
        else:
            raise ParsingError

    def ltlf_always(self, args):
        if len(args) == 1:
            return args[0]
        else:
            f = args[-1]
            for _ in args[:-1]:
                f = LTLfAlways(f)
            return f

    def ltlf_eventually(self, args):
        if len(args) == 1:
            return args[0]
        else:
            f = args[-1]
            for _ in args[:-1]:
                f = LTLfEventually(f)
            return f

    def ltlf_next(self, args):
        if len(args) == 1:
            return args[0]
        else:
            f = args[-1]
            for _ in args[:-1]:
                f = LTLfNext(f)
            return f

    def ltlf_weak_next(self, args):
        if len(args) == 1:
            return args[0]
        else:
            f = args[-1]
            for _ in args[:-1]:
                f = LTLfWeakNext(f)
            return f

    def ltlf_not(self, args):
        if len(args) == 1:
            return args[0]
        else:
            f = args[-1]
            for _ in args[:-1]:
                f = LTLfNot(f)
            return f

    def ltlf_wrapped(self, args):
        if len(args) == 1:
            return args[0]
        elif len(args) == 3:
            _, formula, _ = args
            return formula
        else:
            raise ParsingError

    def ltlf_atom(self, args):
        assert len(args) == 1
        return args[0]

    def ltlf_true(self, args):
        return LTLfTrue()

    def ltlf_false(self, args):
        return LTLfFalse()

    def ltlf_last(self, args):
        return LTLfLast()

    def ltlf_end(self, args):
        raise NotImplementedError("LTLf end not supported, yet")

    def ltlf_symbol(self, args):
        assert len(args) == 1
        token = args[0]
        symbol = str(token)
        return LTLfAtomic(symbol)


class LTLfParser:
    def __init__(self):
        self._transformer = LTLfTransformer()
        self._parser = Lark(open(str(Path(CUR_DIR, "ltlf.lark"))), parser="lalr")

    def __call__(self, text):
        tree = self._parser.parse(text)
        formula = self._transformer.transform(tree)
        return formula


if __name__ == "__main__":
    parser = LTLfParser()
    while True:
        try:
            s = input("ltlf> ")
            if not s:
                continue
            result = parser(s)
            print("result:", result, type(result))
        except EOFError:
            break
        except Exception as e:
            print(str(e))
