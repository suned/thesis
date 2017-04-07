from funcparserlib import lexer, parser


def tokenize(s):
    def choice(choices):
        return r"|".join(choices)

    specs = [
        ("Tab", (r'\t',)),
        ('Number', (r'[0-9]+',)),
        ('Text', (r'''[A-Za-z0-9 ,:;\(\)\$%'\-_=&\?!~#*\+\./"]+''',)),
        ("LineBreak", (r"\n",)),
    ]
    useless = ["Tab"]
    tokenizer = lexer.make_tokenizer(specs)
    return [x for x in tokenizer(s)
            if x.type not in useless]


def parse(s):
    def tokval(x):
        return x.value

    def toktype(t):
        return parser.some(lambda x: x.type == t) >> tokval

    number = toktype("Number")
    text = toktype("Text")
    newline = toktype("LineBreak")

    line = number + text +
