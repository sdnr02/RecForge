from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from core.recommender import RecommendationEngine

from utils.sorting import bubble_sort_map_by_values

class ContentBasedRecommender:
    """Class for implementing content-based recommendation strategies"""

    def __init__(self, engine: RecommendationEngine) -> None:
        """Initializing the content based recommendation object"""
        self.engine = engine
        self.category_index: Dict[str, List[str]] = {}
        self.tag_index: Dict[str, List[str]] = {}

    def build_category_index(self) -> None:
        """Method to build category index from the existing Item store in the Engine object"""
        # Iterating through the elements of the item store map
        for item_id, item in self.engine.item_store.items():
            # Checking if the category already exists
            if item.category not in self.category_index:
                self.category_index[item.category] = []

            # Adding the item ID since build is always a method from scratch
            self.category_index[item.category].append(item_id)

    def build_tag_index(self) -> None:
        """Method to build tag index from the existing Item store in the Engine object"""
        # Iterating through the elements of the item store map
        for item_id, item in self.engine.item_store.items():
            # Iterating through the tags present for that particular item
            if item.tags:
                for tag in item.tags:
                    if tag not in self.tag_index:
                        # Initializing empty list for first time tags
                        self.tag_index[tag] = []

                    # adding the item ID to the tag index
                    self.tag_index[tag].append(item_id)

    def extract_user_preferences(self, user_id: str) -> Dict:
        """Method to score the user preferences"""
        # Extracting the ratings for that particular user ID
        user_rating_map = self.engine.user_rating_store.get(user_id)

        # If user rating map is present
        if user_rating_map:
            # Initializing total items and feature scores tracking
            total_items = 0
            feature_map = {}

            # Iterating through existing user ratings
            for item_id, rating in user_rating_map.items():
                # Only considering high ratings
                if rating > 4.0:
                    # Retrieving item object
                    item_object = self.engine.get_item(item_id)

                    # Checking if item category is already present in the feature map
                    if item_object.category not in feature_map:
                        feature_map[item_object.category] = 1
                    else:
                        feature_map[item_object.category] = feature_map[item_object.category] + 1

                    # Iterating through the tags of the object
                    for tag in item_object.tags:
                        # Checking if the tag already exists in the feature map
                        if tag not in feature_map:
                            feature_map[tag] = 1
                        else:
                            feature_map[tag] = feature_map[tag] + 1

                    # Incrementing total number of items
                    total_items = total_items + 1

            # Initializing hash map for calculating user preferences
            user_preference_map = {}

            # Iterating through the feature score tracking map
            for feature_id, number_of_ratings in feature_map.items():
                # Calculating weight of the feature
                feature_weight = number_of_ratings/total_items
                user_preference_map[feature_id] = feature_weight

            # Returning final preference weights
            return user_preference_map

        # In case user ratings are not present
        else:
            return None
        
    def recommend(
        self,
        user_id: str,
        n: int = 10
    ) -> List[str]:
        """Method to score, rank and recommend top N items based on content match"""
        # Extracting user preferences for that particular ID
        user_preferences_map = self.extract_user_preferences(user_id)

        # In case user had no ratings
        if not user_preferences_map:
            return []
        
        # Retrieving the items that the user has rated
        user_rated_items = self.engine.user_rating_store.get(user_id, {})

        # Initializing a map for candidate items and their scores
        candidate_score_map = {}

        # Iterating through the store of items
        for item_id in self.engine.item_store.keys():
            # Initializing the item in the candidate score map
            if item_id not in user_rated_items:
                candidate_score_map[item_id] = 0.0

        # Iterating through the candidate items
        for candidate_item_id in candidate_score_map.keys():
            # Retrieving the item object
            item = self.engine.item_store.get(candidate_item_id)

            # Check if Item category exists in the user preferences map
            if item.category in user_preferences_map.keys():
                candidate_score_map[candidate_item_id] = candidate_score_map[candidate_item_id] + (user_preferences_map[item.category] * 10)

            # Check if Item tags exist in the user preferences map
            if item.tags:
                for tag in item.tags:
                    # Adding the score in case tags were present in user preferences
                    if tag in user_preferences_map.keys():
                        candidate_score_map[candidate_item_id] = candidate_score_map[candidate_item_id] + (user_preferences_map[tag] * 5)
        
        # Sorting the candidate_score_map dictionary using bubble sort
        candidate_score_map = bubble_sort_map_by_values(candidate_score_map)

        # Initializing the final return map for the top n candidate items
        first_n_candidates = []
        count = 0
        
        # Iterating through the sorted candidate map
        for item_id, score in candidate_score_map.items():
            # Checking if the we still haven't found the first n items
            if count < n:
                first_n_candidates.append(item_id)
                count = count + 1
            
            # Exiting the loop if we've found the top N candidates
            else:
                break
        
        # Finally returning the top N recommended candidates
        return first_n_candidates
