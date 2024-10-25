class StructuredSentimentGraph:
    def __init__(self):
        self.expression_head = None
        self.holder_head = None
        self.target_head = None

class HolderNode:
    def __init__(self):
        self.head = None
        self.next = None

class ExpressionNode:
    def __init__(self):
        self.head = None
        self.next = None
        self.holder_head = None
        self.holder_head = None

class TargetNode:
    def __init__(self):
        self.head = None
        self.next = None