from typing import Dict

class User:
    """Dataclass that stores user data"""

    def __init__(
        self,
        user_id: str,
        preferences: Dict = None,
        demographics: Dict = None
    ) -> None:
        """Initializing User object"""
        self.user_id = user_id
        self.preferences = preferences
        self.demographics = demographics
        self.history = []