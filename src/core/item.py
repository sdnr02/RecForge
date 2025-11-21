from typing import List, Dict

class Item:
    """Dataclass that stores item data"""

    def __init__(
        self,
        item_id: str,
        category: str,
        tags: List = None,
        features: Dict = None
    ) -> None:
        """Initializing Item object"""
        self.item_id = item_id
        self.category = category
        self.tags = tags
        self.features = features
        self.popularity = 0 # Will be tracked later