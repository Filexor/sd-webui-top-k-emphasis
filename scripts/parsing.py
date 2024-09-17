from collections import namedtuple
from copy import deepcopy
from re import A

import lark
from sympy import Mul

rules=r"""
start: text1*
text1: (other|emphasis|op|cp|ob|cb|break1)*
text2: (other|emphasis|break2)*
emphasis: "(" text2* ":" S* weight1 (S* weight2)* S* ")"
?other: (OTHER|escaped|COLON|BACKSLASH)+
?escaped.1: "\\" (OP|CP|OB|CB)
weight1.1: multiplier1 [key [threshold]]
weight2.1: multiplier2 [key [threshold]]
multiplier1: /[+-]?(?:\d+\.?\d*|\d*\.?\d+)/
multiplier2: signs /(?:\d+\.?\d*|\d*\.?\d+)/
key: /[a-zA-Z]/
threshold: /\d+\.?\d*|\d*\.?\d+/
S: /\s/
OTHER: /[^\(\)\[\]\:\\]+/
op: OP
cp: CP
ob: OB
cb: CB
OP: "("
CP: ")"
OB: "["
CB: "]"
COLON: ":"
BACKSLASH: "\\"
signs: pp|pn|np|nn|p|n
pp: /\+\s*\+/
pn: /\+\s*\-/
np: /\-\s*\+/
nn: /\-\s*\-/
p: /\+\s*/
n: /\-\s*/
break1.1: /^BREAK\s+|\s+BREAK$/
break2.2: /\sBREAK\s/
%ignore S
"""
lark_rules = lark.lark.Lark(rules)

Multiplier = namedtuple("Multiplier", ["weight", "key", "threshold"])

class EmphasisPair():
    def __init__(self, text: str, multipliers: list[Multiplier]) -> None:
        self.text = deepcopy(text)
        self.multipliers = deepcopy(multipliers)
        pass

def parse_prompt_attention(text):
    emphasis_pairs: list[EmphasisPair] = []
    root = lark_rules.parse(text)

    class Preprocess(lark.visitors.Transformer):
        def multiplier1(self, children: list):
            return str.join("", children)
        def multiplier2(self, children: list):
            return str.join("", children)
        def signs(self, children: list):
            return str.join("", children)
        def key(self, children: list):
            return str.join("", children)
        def threshold(self, children: list):
            return str.join("", children)
        def pp(self, token:lark.lexer.Token):
            return "+"
        def pn(self, token:lark.lexer.Token):
            return "-"
        def np(self, token:lark.lexer.Token):
            return "-"
        def nn(self, token:lark.lexer.Token):
            return "+"
        def p(self, token:lark.lexer.Token):
            return "+"
        def n(self, token:lark.lexer.Token):
            return "-"
        
    root = Preprocess().transform(root)
    multiplier = [[Multiplier(1.0, "c", 0.0)]]

    class Parser(lark.visitors.Interpreter):
        def start(self, tree: lark.tree.Tree):
            for i in tree.children:
                Parser().visit(i)
        def text1(self, tree: lark.tree.Tree):
            for i in tree.children:
                if isinstance(i, lark.tree.Tree):
                    Parser().visit(i)
                elif isinstance(i, lark.lexer.Token):
                    emphasis_pairs.append(EmphasisPair(i, multiplier))
        def text2(self, tree: lark.tree.Tree):
            Parser().visit_children(tree)
        def emphasis(self, tree: lark.tree.Tree):
            multiplier_this = []
            multiplier_this.append(Multiplier(float(tree.children[1].children[0]), tree.children[1].children[1] or "c", float(tree.children[1].children[2] or 0.0)))
            for i in tree.children[2::2]:
                multiplier_this.append(Multiplier(float(i.children[0]), i.children[1] or "c", float(i.children[2]) or 0.0))
            multiplier.append(multiplier_this)
            Parser().visit(tree.children[0])
            multiplier.pop()
        def other(self, tree: lark.tree.Tree):
            tokens = []
            for i in tree.children:
                tokens.append(i)
            emphasis_pairs.append(EmphasisPair(str.join("", tokens), multiplier))
        def op(self, tree: lark.tree.Tree):
            multiplier.append([Multiplier(1.1, "c", 0.0)])
        def cp(self, tree: lark.tree.Tree):
            multiplier.append([Multiplier(1.0 / 1.1, "c", 0.0)])
        def ob(self, tree: lark.tree.Tree):
            multiplier.append([Multiplier(1.0 / 1.1, "c", 0.0)])
        def cb(self, tree: lark.tree.Tree):
            multiplier.append([Multiplier(1.1, "c", 0.0)])
        def break1(self, tree: lark.tree.Tree):
            tokens = []
            for i in tree.children:
                i = str.strip(i)
                tokens.append(i)
            emphasis_pairs.append(EmphasisPair(str.join("", tokens), multiplier))
        def break2(self, tree: lark.tree.Tree):
            tokens = []
            for i in tree.children:
                i = str.strip(i)
                tokens.append(i)
            emphasis_pairs.append(EmphasisPair(str.join("", tokens), multiplier))

    def flatten_multipliers(emphasis_pairs: list[EmphasisPair]) -> list[EmphasisPair]:
        for emphasis_pair in emphasis_pairs:
            multiplier_list: list[Multiplier] = []
            for multipliers in emphasis_pair.multipliers:
                for multiplier in multipliers:
                    multiplier_list.append(multiplier)
            emphasis_pair.multipliers = multiplier_list
        return emphasis_pairs

    Parser().visit(root)
    emphasis_pairs = flatten_multipliers(emphasis_pairs)
    return emphasis_pairs


# a = (parse_prompt_attention(r"BREAK (tes\t\(  BREAK   :1+2k3. + .4r5.6) BREAK"))
# print(a)