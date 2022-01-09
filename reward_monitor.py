from pylogics.syntax.base import (
    And,
    Not,
    Or
)
from pylogics.syntax.ltl import (
    Always,
    Atomic,
    Eventually,
    Next,
    PropositionalFalse,
    PropositionalTrue,
    Release,
    Until,
    WeakNext,
    FalseFormula,
    TrueFormula
)


class RewardMonitor:
    def __init__(self):
        self.S = []  # list of RM states
        self.s1 = None  # initial state
        self.V = {}  # dict of register values
        self.V_initial = {}  # dict of register values before reading in a trace
        self.delta_s = {}  # state-transition function
        self.delta_v = {}  # monitor register update function
        self.current_state = 1

    def step(self, fluents):
        # Updating all monitor registers in order, if an update function exists
        if self.current_state in self.delta_v.keys():
            for v in self.V:
                if v in self.delta_v[self.current_state].keys():
                    self.V[v] = self.delta_v[self.current_state][v](self.V, fluents)
        # Doing state transition
        self.current_state = self.delta_s[self.current_state]

    def reset(self):
        self.V = self.V_initial.copy()
        self.current_state = self.s1

    def reward(self):
        return self.V[list(self.V)[-1]]


def create_monitor(formula, id=""):
    """
    :param id: Unique identifier for differentiating different subformulas with the same name
    :return: Reward monitor that represents the formula
    """
    if isinstance(formula, PropositionalTrue):
        formula_key = f"true{id}"
        monitor = RewardMonitor()
        monitor.S = [1, 2]
        monitor.s1 = 1
        monitor.V = {formula_key: 0}  # Any propositional formula is false on the empty trace
        monitor.delta_s = {1: 2, 2: 2}
        monitor.delta_v = {1: {formula_key: (lambda V, g: 1)}, 2: {}}  # V[true] updated to 1
        monitor.V_initial = monitor.V.copy()
    elif isinstance(formula, PropositionalFalse):
        formula_key = f"false{id}"
        monitor = RewardMonitor()
        monitor.S = [1, 2]
        monitor.s1 = 1
        monitor.V = {formula_key: 0}
        monitor.delta_s = {1: 2, 2: 2}
        monitor.delta_v = {1: {formula_key: (lambda V, g: 0)}, 2: {}}  # V[false] is always 0
        monitor.V_initial = monitor.V.copy()
    elif isinstance(formula, (TrueFormula, FalseFormula)):
        # E.g. a&!a is reduced to False, which is only correct in boolean semantics
        # In quantitative semantics, a&!a is not equal to 0, but to the min between a and 1-a
        raise ValueError('Reduction to True/False only works for boolean semantics.')
    elif isinstance(formula, Atomic):
        formula_key = formula.name + id
        monitor = RewardMonitor()
        monitor.S = [1, 2]
        monitor.s1 = 1
        monitor.V = {formula_key: 0}
        monitor.delta_s = {1: 2, 2: 2}
        # V[formula] updated to g value at state 1, then no more updates
        monitor.delta_v = {1: {formula_key: (lambda V, g: g[formula.name])}, 2: {}}
        monitor.V_initial = monitor.V.copy()
    elif isinstance(formula, Not):
        # We add a /0 symbol to the id, signifying going down a layer
        monitor = create_monitor(formula.argument, id + "/0")
        subformula_key = list(monitor.V.keys())[-1]  # The last register represents the value of the subformula
        formula_key = f"!({subformula_key})"  # The new formula register key has an ! prepended
        if isinstance(formula.argument, (Next, Until, Eventually)):
            # !Xvarphi and !(varphi1Uvarphi2) are true for empty traces
            monitor.V.update({formula_key: 1})
        else:
            monitor.V.update({formula_key: 0})

        for u in monitor.S:
            # The value of the formula is 1 subtracted by the value of the subformula
            monitor.delta_v[u][formula_key] = lambda V, g: 1 - V[subformula_key]
        monitor.V_initial = monitor.V.copy()
    elif isinstance(formula, (And, Or)):
        monitor = RewardMonitor()
        # The id is extended by a unique number for each subformula
        submonitors = [create_monitor(operand, f"{id}/{i}") for i, operand in enumerate(formula.operands)]
        # Last register of each submonitor, which represents final value of that submonitor
        subformula_keys = [list(submonitor.V.keys())[-1] for submonitor in submonitors]
        if isinstance(formula, And):
            formula_key = f"&({','.join(subformula_keys)})"
        else:
            formula_key = f"|({','.join(subformula_keys)})"

        monitor.s1 = 1
        monitor.delta_v = {}

        # All registers are merged into one
        monitor.V = {k: v for submonitor in submonitors for k, v in submonitor.V.items()}
        # The initial formula register value is min/max of subformula register values
        if isinstance(formula, And):
            monitor.V.update({formula_key: min([monitor.V[key] for key in subformula_keys])})
        else:
            monitor.V.update({formula_key: max([monitor.V[key] for key in subformula_keys])})

        # A superstate represents the states you are in for each submonitor
        super_states = {}
        super_state_index = 1
        submonitor_state_indexes = tuple([submonitor.s1 for submonitor in submonitors])
        super_state = {submonitor_state_indexes: super_state_index}
        while submonitor_state_indexes not in super_states:
            super_states.update(super_state)
            monitor.S.append(super_state_index)
            monitor.delta_v[super_state_index] = {}
            if super_state_index != 1:
                monitor.delta_s[super_state_index - 1] = super_state_index
            # Merging the delta_ms for the state super_state_index from all submonitors
            for i, submonitor in enumerate(submonitors):
                u = submonitor_state_indexes[i]
                if u in submonitor.delta_v.keys():
                    monitor.delta_v[super_state_index].update(
                        {k: v for k, v in submonitor.delta_v[u].items()})
            if isinstance(formula, And):
                # The value of "and" is the minimum between the values of the subformulas
                monitor.delta_v[super_state_index][formula_key] = lambda V, g: min([V[key] for key in subformula_keys])
            else:
                # The value of "or" is the maximum between the values of the subformulas
                monitor.delta_v[super_state_index][formula_key] = lambda V, g: max([V[key] for key in subformula_keys])

            super_state_index += 1
            submonitor_state_indexes = \
                tuple([submonitor.delta_s[submonitor_state_indexes[i]] for i, submonitor in enumerate(submonitors)])
            super_state = {submonitor_state_indexes: super_state_index}

        monitor.delta_s[super_state_index - 1] = super_states[submonitor_state_indexes]

        monitor.V_initial = monitor.V.copy()
    elif isinstance(formula, (Next, WeakNext)):
        # We add a /0 symbol to the id, signifying going down a layer
        submonitor = create_monitor(formula.argument, id + "/0")
        subformula_key = list(submonitor.V.keys())[-1]  # The last register represents the value of the subformula
        if isinstance(formula, Next):
            formula_key = f"X({subformula_key})"  # The new formula register key has an X prepended
            submonitor.V.update({formula_key: 0})
        else:
            formula_key = f"WX({subformula_key})"  # The new formula register key has a WX prepended
            submonitor.V.update({formula_key: 1})

        # Creating a new monitor with a prepended state
        monitor = RewardMonitor()
        monitor.S = submonitor.S + [len(submonitor.S) + 1]
        monitor.s1 = 1
        monitor.V = submonitor.V
        # Making a new first state that transitions to the submonitor's first state
        monitor.delta_s = {1: 2}
        if isinstance(formula, Next):
            monitor.delta_v = {1: {formula_key: lambda V, g: 0}}
        else:
            monitor.delta_v = {1: {formula_key: lambda V, g: 1}}
        # Renaming each state, transition and register update as one higher number
        for u in submonitor.S:
            monitor.delta_s[u + 1] = submonitor.delta_s[u] + 1
            if u in submonitor.delta_v.keys():
                monitor.delta_v[u + 1] = {}
                monitor.delta_v[u + 1].update(submonitor.delta_v[u])
                # The value is the same as the subformula after reading the first symbol
                monitor.delta_v[u + 1][formula_key] = lambda V, g: V[subformula_key]
        monitor.V_initial = monitor.V.copy()
    elif isinstance(formula, (Eventually, Always)):
        if isinstance(formula.argument, (Eventually, Always, Until, Release)):
            raise ValueError('Nesting of GURF operators is not allowed.')
        # We add a /0 symbol to the id, signifying going down a layer
        submonitors = [create_monitor(formula.argument, id + "/0")]
        # Creating a submonitor for each additional state
        for u in range(1, len(submonitors[0].S) - 1):
            submonitors.append(create_monitor(formula.argument, id + f"/{u}"))
        # The last register represents the value of the subformula
        subformula_keys = [list(submonitor.V.keys())[-1] for submonitor in submonitors]
        if isinstance(formula, Eventually):
            formula_key = f"F({subformula_keys[0]})"  # The new formula register key has an F prepended
        else:
            formula_key = f"G({subformula_keys[0]})"  # The new formula register key has a G prepended

        monitor = RewardMonitor()
        monitor.V = {}
        monitor.S = [*range(1, 2 * len(submonitors[0].S) - 2)]
        monitor.s1 = 1
        monitor.delta_s = {u: u + 1 for u in range(1, len(monitor.S))}
        looping_index = len(submonitors[0].S) - 1  # Where the last state of the monitor loops back
        monitor.delta_s.update({len(monitor.S): looping_index})
        monitor.delta_v = {u: {} for u in monitor.S}

        # Adding registers from submonitors at correct positions
        for i, submonitor in enumerate(submonitors):
            monitor.V.update(submonitor.V)
            for u in range(i + 1, len(monitor.S) + 1):
                sub_i = (u - i - 1) % len(submonitors) + 1
                monitor.delta_v[u].update(submonitor.delta_v[sub_i])

        # Adding register updates for finished subformula values
        formula_f_key = formula_key + "_f"
        if isinstance(formula, Eventually):
            monitor.V[formula_f_key] = 0
            monitor.V[formula_key] = 0
            for subformula_key in subformula_keys:
                monitor.V[subformula_key] = 0
        else:
            monitor.V[formula_f_key] = 1
            monitor.V[formula_key] = 1
            for subformula_key in subformula_keys:
                monitor.V[subformula_key] = 1
        for u in range(looping_index, len(monitor.S) + 1):
            finished_v = u - looping_index
            if isinstance(formula, Eventually):
                # Maximum between value of previously finished subformulas and currently finished one
                monitor.delta_v[u][formula_f_key] = \
                    lambda V, g, v=finished_v: max(V[formula_f_key], V[subformula_keys[v]])
            else:
                # Minimum between value of previously finished subformulas and currently finished one
                monitor.delta_v[u][formula_f_key] = \
                    lambda V, g, v=finished_v: min(V[formula_f_key], V[subformula_keys[v]])

        # Adding register updates for formula value
        for u in monitor.S:
            if isinstance(formula, Eventually):
                # Maximum between value of previously finished subformulas and current subformula values
                monitor.delta_v[u][formula_key] = \
                    lambda V, g: max(V[formula_f_key], max([V[subformula_key] for subformula_key in subformula_keys]))
            else:
                # Minimum between value of previously finished subformulas and current subformula values
                monitor.delta_v[u][formula_key] = \
                    lambda V, g: min(V[formula_f_key], min([V[subformula_key] for subformula_key in subformula_keys]))

        monitor.V_initial = monitor.V.copy()
    return monitor
