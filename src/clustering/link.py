"""
Class which represent a Link in the graph.
"""
class Link:
    def __init__(self, id_from, id_to, t):
        self.id_from = id_from
        self.id_to = id_to
        self.type = t
        
    def __str__(self) -> str:
        return f"Link(id_from: {self.id_from}, id_to: {self.id_to}, type: {self.type})"
        
__all__ = ["Link"]
      