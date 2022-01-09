from pylogics.parsers.ltl import parse_ltl
from flloat.parser.ltlf import LTLfParser as FLLOATParser
from flloat.parser.ldlf import LDLfParser
from reward_monitor import create_monitor
import random
import unittest

random.seed(1)

def generate_random_restricted_formula(layer=1, gurf=True):
    symbols = ("true", "false", "a", "b", "c", "d", "e")
    if gurf:
        unary_operators = ("!", "X[!]", "X", "G", "F")
    else:
        unary_operators = ("!", "X[!]", "X")
    binary_operators = ("&", "|")

    if layer >= 5:
        symbol = random.choice(symbols)
        return symbol, symbol

    operator_type = random.randint(0, 2)

    if operator_type == 0:
        # Symbol
        symbol = random.choice(symbols)
        return symbol, symbol
    if operator_type == 1:
        # Unary operator
        operator = random.choice(unary_operators)
        flloat_operator = operator
        if operator in ("G", "F"):
            # No more GURF operators allowed in lower layers
            gurf = False
        elif operator == "X[!]":
            flloat_operator = "X"
        elif operator == "X":
            flloat_operator = "WX"
        subformula, flloat_subformula = generate_random_restricted_formula(layer + 1, gurf)
        return f"{operator}({subformula})", f"{flloat_operator}({flloat_subformula})"
    else:
        # Binary operator
        operator = random.choice(binary_operators)
        if operator in ("G", "F"):
            # No more GURF operators allowed in lower layers
            gurf = False
        left_formula, flloat_left_formula = generate_random_restricted_formula(layer + 1, gurf)
        right_formula, flloat_right_formula = generate_random_restricted_formula(layer + 1, gurf)
        return f"({left_formula}){operator}({right_formula})", \
               f"({flloat_left_formula}){operator}({flloat_right_formula})"


def generate_random_ltlf_formula(layer=1):
    symbols = ("true", "false", "a", "b", "c", "d", "e")
    unary_operators = ("!", "X", "WX", "G", "F")
    binary_operators = ("&", "|", "R", "U")

    if layer >= 5:
        symbol = random.choice(symbols)
        return symbol

    operator_type = random.randint(0, 2)

    if operator_type == 0:
        # Symbol
        symbol = random.choice(symbols)
        return symbol
    if operator_type == 1:
        # Unary operator
        operator = random.choice(unary_operators)
        subformula = generate_random_ltlf_formula(layer + 1)
        return f"{operator}({subformula})"
    else:
        # Binary operator
        operator = random.choice(binary_operators)
        left_formula = generate_random_ltlf_formula(layer + 1)
        right_formula = generate_random_ltlf_formula(layer + 1)
        return f"({left_formula}){operator}({right_formula})"


def generate_random_regular_formula(layer=1):
    symbols = ("true", "false", "a", "b", "c", "d", "e")
    unary_operators = ("?", "*")
    binary_operators = ("+", ";")

    if layer >= 5:
        symbol = random.choice(symbols)
        return symbol

    operator_type = random.randint(0, 2)

    if operator_type == 0:
        # Symbol
        symbol = random.choice(symbols)
        return symbol
    if operator_type == 1:
        # Unary operator
        operator = random.choice(unary_operators)
        if operator == "?":
            subformula = generate_random_ldlf_formula(layer + 1)
            return f"{operator}({subformula})"
        else:
            subformula = generate_random_regular_formula(layer + 1)
            return f"({subformula}){operator}"
    else:
        # Binary operator
        operator = random.choice(binary_operators)
        left_formula = generate_random_regular_formula(layer + 1)
        right_formula = generate_random_regular_formula(layer + 1)
        return f"({left_formula}){operator}({right_formula})"


def generate_random_ldlf_formula(layer=1):
    symbols = ("tt", "ff")
    unary_operators = ("!",)
    binary_operators = ("&", "|")
    regular_operators = ("<>", "[]")

    if layer >= 5:
        symbol = random.choice(symbols)
        return symbol

    operator_type = random.randint(0, 3)

    if operator_type == 0:
        # Symbol
        symbol = random.choice(symbols)
        return symbol
    if operator_type == 1:
        # Unary operator
        operator = random.choice(unary_operators)
        subformula = generate_random_ldlf_formula(layer + 1)
        return f"{operator}({subformula})"
    elif operator_type == 2:
        # Binary operator
        operator = random.choice(binary_operators)
        left_formula = generate_random_ldlf_formula(layer + 1)
        right_formula = generate_random_ldlf_formula(layer + 1)
        return f"({left_formula}){operator}({right_formula})"
    else:
        # Regular operator
        operator = random.choice(regular_operators)
        regular_formula = generate_random_regular_formula(layer + 1)
        ldlf_formula = generate_random_ldlf_formula(layer + 1)
        if operator == "<>":
            return f"<{regular_formula}>({ldlf_formula})"
        else:
            return f"[{regular_formula}]({ldlf_formula})"


def generate_random_trace():
    trace_length = random.randint(1, 10)
    symbols = ("a", "b", "c", "d", "e")
    trace = []

    for _ in range(trace_length):
        trace.append({symbol: random.uniform(0, 1) for symbol in symbols})

    return trace


def generate_random_boolean_trace():
    trace_length = random.randint(1, 10)
    symbols = ("a", "b", "c", "d", "e")
    trace_num = []
    trace_bool = []

    for _ in range(trace_length):
        random_assignment = {symbol: random.randint(0, 1) for symbol in symbols}
        trace_num.append(random_assignment)
        trace_bool.append({k: bool(v) for k, v in random_assignment.items()})

    return trace_num, trace_bool


class TestQuant(unittest.TestCase):

    def test_quant_ltlf(self):
        print("Testing quantitative LTLf")
        for run in range(100):
            formula = generate_random_ltlf_formula()
            tg = FLLOATParser()(formula)
            print(f"Formula {run + 1}: {tg}")
            trace_num, trace_bool = generate_random_boolean_trace()

            # On a boolean trace, quantitative and boolean semantics should evaluate to the same value
            assert tg.truth(trace_bool) == bool(tg.value(trace_num))

    def test_quant_ldlf(self):
        print("Testing quantitative LDLf")
        for run in range(100):
            formula = generate_random_ldlf_formula()
            tg = LDLfParser()(formula)
            print(f"Formula {run + 1}: {tg}")
            trace_num, trace_bool = generate_random_boolean_trace()

            # On a boolean trace, quantitative and boolean semantics should evaluate to the same value
            assert tg.truth(trace_bool) == bool(tg.value(trace_num))

    def test_reward_monitor(self):
        print("Testing reward monitor construction")
        for run in range(100):
            formula, flloat_formula = generate_random_restricted_formula()
            trace = generate_random_trace()

            tg = parse_ltl(formula)
            print(f"Formula {run + 1}: {tg}")

            tg_flloat = FLLOATParser()(flloat_formula)
            try:
                rm = create_monitor(tg)
            except ValueError:
                # Formula is malformed or incorrectly reduced by parser for quantitative case
                continue

            rm_rewards = []
            actual_rewards = []
            for i, step in enumerate(trace):
                rm.step(step)
                rm_reward = rm.reward()
                actual_reward = tg_flloat.value(trace[:i + 1])
                rm_rewards.append(rm_reward)
                actual_rewards.append(actual_reward)

            assert rm_rewards == actual_rewards


