from copy import deepcopy

import lark

rules=r"""
start: break1
break1: [/BREAK /] (text|break3)?
text: (OTHER|/\s(?!BREAK )/|escaped|emphasis|op|cp|ob|cb|break2) (text|break3)?
OTHER.-1: /[^\(\)\[\]\:\s\\]+/
escaped: "\\" /./ (text|break3)?
emphasis: "(" [text] ":" multiplier1 ([/\s+/] multiplier2)* ")" (text|break3)?
multiplier1: [sign] float [key [float [option*]]]
multiplier2: [sign [/\s+/]] sign float [key [float [option*]]]
sign: /[+-]/
key: /[a-zA-Z]+/
float: /\d+/|/\d+\.\d*/|/\d*\.\d+/
option: key [float]
op: "(" (text|break3)?
cp: ")" (text|break3)?
ob: "[" (text|break3)?
cb: "]" (text|break3)?
break2: " BREAK " (text|break3)?
break3: " BREAK"
"""
lark_rules = lark.lark.Lark(rules, parser="lalr")

class Multiplier():
    def __init__(self, weight: float = 1.0, key: str = "c", threshold: float = 0.0, options: list[tuple[str, float|None]] = []) -> None:
        self.weight = weight
        self.key = key
        self.threshold = threshold
        self.options = options

    def __repr__(self) -> str:
        return str((self.weight, self.key, self.threshold, self.options))

class EmphasisPair():
    def __init__(self, text: str = "", multipliers: list[Multiplier] = []) -> None:
        self.text = deepcopy(text)
        self.multipliers = deepcopy(multipliers)

    def __iter__(self):
        return EmphasisPairIterator(self)
    
    def __repr__(self) -> str:
        return '"' + self.text + '" ' + str(self.multipliers)
    
class EmphasisPairIterator():
    def __init__(self, ref: EmphasisPair) -> None:
        self.i = 0
        self.ref = ref
        pass

    def __iter__(self):
        return self
    
    def __next__(self):
        match self.i:
            case 0:
                self.i += 1
                return self.ref.text
            case 1:
                self.i += 1
                return self.ref.multipliers
            case _:
                raise StopIteration()
            
class BREAK_Object():
    def __init__(self) -> None:
        self.text = "BREAK"
        self.multipliers = []

    def __iter__(self):
        return BREAK_ObjectIterator(self)
    
    def __repr__(self) -> str:
        return '"' + self.text + '" ' + str(self.multipliers)
    
class BREAK_ObjectIterator():
    def __init__(self, ref: BREAK_Object) -> None:
        self.i = 0
        self.ref = ref
        pass

    def __iter__(self):
        return self
    
    def __next__(self):
        match self.i:
            case 0:
                self.i += 1
                return self.ref.text
            case 1:
                self.i += 1
                return self.ref.multipliers
            case _:
                raise StopIteration()
            
def apply_multiplier(input: list, multipliers: list[Multiplier]):
    if isinstance(input[0], str):
        input[0] = EmphasisPair(input[0], multipliers)
    elif isinstance(input[0], lark.tree.Tree):
        raise Exception(f'Unexpected tree given.', str(input[0]))
    elif isinstance(input[0], (EmphasisPair, BREAK_Object)):
        input[0].multipliers = multipliers + input[0].multipliers
    elif isinstance(input[0], list):
        for i in range(len(input[0])):
            apply_multiplier([input[0][i]], multipliers)
    else:
        raise Exception('Unexpected type given.', str(input[0]))

def parse_prompt_attention(text) -> list[EmphasisPair]:
    root = lark_rules.parse(text)
    print(root.pretty())

    class Preprocess(lark.visitors.Transformer):
        def start(self, children: list[EmphasisPair|BREAK_Object|list]):
            if isinstance(children[0], (EmphasisPair, BREAK_Object)):
                return [children[0]]
            elif isinstance(children[0], list):
                return children[0]
            else:
                raise Exception('Unexpected type given.', str(children[0]))
        def break1(self, children: list[EmphasisPair|BREAK_Object|list]):
            if len(children) == 1:
                if children[0] is None:
                    return lark.visitors.Discard
                else:
                    return BREAK_Object()
            else:
                if children[0] is None:
                    return children[1]
                elif isinstance(children[1], (EmphasisPair, BREAK_Object)):
                    return [BREAK_Object(), children[1]]
                elif isinstance(children[1], list):
                    children[1].insert(0, BREAK_Object())
                    return children[1]
                else:
                    raise Exception('Unexpected type given.', str(children[0]))
        def text(self, children: list[lark.lexer.Token|lark.tree.Tree|list[EmphasisPair|BREAK_Object]]) -> list[EmphasisPair|BREAK_Object]:
            if len(children) == 1:
                if children[0] is None:
                    return lark.visitors.Discard
                elif isinstance(children[0], str):
                    return [EmphasisPair(children[0])]
                elif isinstance(children[0], list):
                    return children[0]
                else:
                    raise Exception('Unexpected type given.', str(children[0]))
            elif isinstance(children[1][0], BREAK_Object):
                if children[0] is None:
                    return children[1]
                elif isinstance(children[0], str):
                    return [EmphasisPair(children[0])] + children[1]
                elif isinstance(children[0], list):
                    return children[0] + children[1]
                else:
                    raise Exception('Unexpected type given.', str(children[0]))
            elif isinstance(children[1][0], EmphasisPair):
                if children[0] is None:
                    return children[1]
                elif isinstance(children[0], str):
                    tmp = EmphasisPair(children[0])
                    if tmp.multipliers == children[1][0].multipliers:
                        children[1][0].text = children[0] + children[1][0].text
                        return children[1]
                    else:
                        return [tmp] + children[1]
                elif isinstance(children[0], list):
                    return children[0] + children[1]
                else:
                    raise Exception('Unexpected type given.', str(children[0]))
            else:
                raise Exception('Unexpected type given.', str(children[1][0]))
        def escaped(self, children: list[EmphasisPair|BREAK_Object|list]):
            if len(children) == 1:
                return EmphasisPair(children[0])
            else:
                if isinstance(children[1], (EmphasisPair, BREAK_Object)):
                    return [EmphasisPair(children[0]), children[1]]
                elif isinstance(children[1], list):
                    children[1].insert(0, EmphasisPair(children[0]))
                    return children[1]
                else:
                    raise Exception('Unexpected type given.', str(children[1]))
        def emphasis(self, children: list[EmphasisPair|BREAK_Object|list|Multiplier]):
            if isinstance(children[0], (EmphasisPair, BREAK_Object)):
                children[0].multipliers = list(children[1::2]) + children[0].multipliers
            elif isinstance(children[0], list):
                for i in range(len(children[0])):
                    apply_multiplier([children[0][i]], list(children[1::2]))
            else:
                raise Exception('Unexpected type given.', str(children[0]))
            if len(children) % 2 == 0:
                return children[0]
            else:
                if isinstance(children[0], (EmphasisPair, BREAK_Object)):
                    if isinstance(children[-1], (EmphasisPair, BREAK_Object)):
                        return [children[0], children[-1]]
                    elif isinstance(children[-1], list):
                        children[-1].insert(0, children[0])
                        return children[-1]
                    else:
                        raise Exception('Unexpected type given.', str(children[-1]))
                elif isinstance(children[0], list):
                    if isinstance(children[-1], (EmphasisPair, BREAK_Object)):
                        children[0].append(children[-1])
                        return children[0]
                    elif isinstance(children[-1], list):
                        return children[0] + children[-1]
                    else:
                        raise Exception('Unexpected type given.', str(children[-1]))
                else:
                    raise Exception('Unexpected type given.', str(children[0]))
        def multiplier1(self, children: list[lark.lexer.Token]):
            if children[0] is None:
                children[0] = 1.0
            if children[2] is None:
                children[2] = "c"
            if children[3] is None:
                children[3] = 1.0
            return Multiplier(children[0]*children[1], children[2], children[3], children[4:])
        def multiplier2(self, children: list[lark.lexer.Token]):
            if children[0] is None:
                children[0] = 1.0
            if children[4] is None:
                children[4] = "c"
            if children[5] is None:
                children[5] = 1.0
            return Multiplier(children[0]*children[2]*children[3], children[4], children[5], children[6:])
        def sign(self, children: list[lark.lexer.Token]):
            match children[0]:
                case "+":
                    return 1.0
                case "-":
                    return -1.0
                case _:
                    raise Exception(f'Unknown sign symbol "{children[0]}" encountered.')
        def key(self, children: list[lark.lexer.Token]):
            return children[0]
        def float(self, children: list[lark.lexer.Token]):
            return float(children[0])
        def option(self, children: list[lark.lexer.Token]):
            return (children[0], children[1])
        def op(self, children: list[lark.lexer.Token|lark.tree.Tree|EmphasisPair]):
            if len(children) == 0:
                return EmphasisPair("", [Multiplier(1.1)])
            else:
                if isinstance(children[0], str):
                    return EmphasisPair(children[0], [Multiplier(1.1)])
                elif isinstance(children[0], lark.tree.Tree):
                    raise Exception(f'Unexpected tree given.', str(children[0]))
                elif isinstance(children[0], (EmphasisPair, BREAK_Object)):
                    children[0].multipliers.insert(0, Multiplier(1.1))
                    return children[0]
                elif isinstance(children[0], list):
                    for i in range(len(children[0])):
                        apply_multiplier([children[0][i]], [Multiplier(1.1)])
                    return children[0]
                else:
                    raise Exception('Unexpected type given.', str(children[0]))
        def cp(self, children: list[lark.lexer.Token|lark.tree.Tree|EmphasisPair]):
            if len(children) == 0:
                return EmphasisPair("", [Multiplier(1.0/1.1)])
            else:
                if isinstance(children[0], str):
                    return EmphasisPair(children[0], [Multiplier(1.0/1.1)])
                elif isinstance(children[0], lark.tree.Tree):
                    raise Exception(f'Unexpected tree given.', str(children[0]))
                elif isinstance(children[0], (EmphasisPair, BREAK_Object)):
                    children[0].multipliers.insert(0, Multiplier(1.0/1.1))
                    return children[0]
                elif isinstance(children[0], list):
                    for i in range(len(children[0])):
                        apply_multiplier([children[0][i]], [Multiplier(1.0/1.1)])
                    return children[0]
                else:
                    raise Exception('Unexpected type given.', str(children[0]))
        def ob(self, children: list[lark.lexer.Token|lark.tree.Tree|EmphasisPair]):
            if len(children) == 0:
                return EmphasisPair("", [Multiplier(1.0/1.1)])
            else:
                if isinstance(children[0], str):
                    return EmphasisPair(children[0], [Multiplier(1.0/1.1)])
                elif isinstance(children[0], lark.tree.Tree):
                    raise Exception(f'Unexpected tree given.', str(children[0]))
                elif isinstance(children[0], (EmphasisPair, BREAK_Object)):
                    children[0].multipliers.insert(0, Multiplier(1.0/1.1))
                    return children[0]
                elif isinstance(children[0], list):
                    for i in range(len(children[0])):
                        apply_multiplier([children[0][i]], [Multiplier(1.0/1.1)])
                    return children[0]
                else:
                    raise Exception('Unexpected type given.', str(children[0]))
        def cb(self, children: list[lark.lexer.Token|lark.tree.Tree|EmphasisPair|BREAK_Object|list]):
            if len(children) == 0:
                return EmphasisPair("", [Multiplier(1.1)])
            else:
                if isinstance(children[0], str):
                    return EmphasisPair(children[0], [Multiplier(1.1)])
                elif isinstance(children[0], lark.tree.Tree):
                    raise Exception(f'Unexpected tree given.', str(children[0]))
                elif isinstance(children[0], (EmphasisPair, BREAK_Object)):
                    children[0].multipliers.insert(0, Multiplier(1.1))
                    return children[0]
                elif isinstance(children[0], list):
                    for i in range(len(children[0])):
                        apply_multiplier([children[0][i]], [Multiplier(1.1)])
                    return children[0]
                else:
                    raise Exception('Unexpected type given.', str(children[0]))
        def break2(self, children: list[list[EmphasisPair|BREAK_Object]]):
            if len(children) == 0:
                return [BREAK_Object()]
            else:
                if isinstance(children[0], list):
                    children[0].insert(0, BREAK_Object())
                    return children[0]
                else:
                    raise Exception('Unexpected type given.', str(children[0]))
        def break3(self, children):
            return [BREAK_Object()]
        
    root = Preprocess().transform(root)
    return root


#a = (parse_prompt_attention(r"BREAK test\:test)(tes\t\(  aBREAK  ((abc BREAK d)e]f :1+2k3. + -.4r5.6b1b1)) BREAK"))
# print(a)