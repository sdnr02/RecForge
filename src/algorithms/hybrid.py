from typing import Dict, List

from algorithms.content_based import ContentBasedRecommender
from utils.sorting import bubble_sort_list_with_tuples_second_element
from algorithms.item_collaborative_filtering import ItemCollaborativeFilterer
from algorithms.user_collaborative_filtering import UserCollaborativeFiltering

class HybridRecommender:
    """Class for implementing a hybrid recommendation algorithm"""

    def __init__(
        self,
        content_based_filterer: ContentBasedRecommender,
        item_based_collaborative_filterer: ItemCollaborativeFilterer,
        user_based_collaborative_filterer: UserCollaborativeFiltering
    ) -> None:
        """Initializing the object"""
        self.content_based_filterer = content_based_filterer
        self.item_based_collaborative_filterer = item_based_collaborative_filterer
        self.user_based_collaborative_filterer = user_based_collaborative_filterer

    def get_hybrid_recommendations(
        self,
        user_id: str,
        n: int = 10,
        weights: Dict[str, float] = None
    ) -> List[str]:
        """Method for retrieving hybrid recommendations based on weighted score from 3"""
        # Checking if a weights map was passed
        if not weights:
            weights = {
                "content": 0.33,
                "item_cf": 0.34,
                "user_cf": 0.33
            }

        # Getting content based recommendations
        content_based_recommendations = self.content_based_filterer.recommend(
            user_id=user_id,
            n=2*n
        )

        # Getting item based collaborative recommendations
        item_collaborative_filtering_recommendations = self.item_based_collaborative_filterer.recommend(
            user_id=user_id,
            n=2*n
        )

        # Getting user based collaborative recommendations
        user_collaborative_filtering_recommendations = self.user_based_collaborative_filterer.recommend(
            user_id=user_id,
            n=2*n
        )

        # Building a position hash map for each list of recommendations
        content_based_map = {}
        for index, item in enumerate(content_based_recommendations):
            content_based_map[item] = index

        item_cf_based_map = {}
        for index, item in enumerate(item_collaborative_filtering_recommendations):
            item_cf_based_map[item] = index

        user_cf_based_map = {}
        for index, item in enumerate(user_collaborative_filtering_recommendations):
            user_cf_based_map[item] = index

        # Get union of all items
        union_set = (
            set(content_based_recommendations) | 
            set(item_collaborative_filtering_recommendations) | 
            set(user_collaborative_filtering_recommendations)
        )

        # Converting back to list
        union_list = list(union_set)

        # Initializing list to store values
        combined_score_list = []

        # Iterating across all elements in all lists
        for item in union_list:
            # Initializing zero score
            combined_score  = 0

            # Checking in content based recommendations list
            if item in content_based_map:
                # Getting position score
                position = content_based_map[item]
                length_of_list = len(content_based_recommendations)
                position_score = (length_of_list - position) / length_of_list
                # Adding to the combined score multiplying by weight
                combined_score = combined_score + (weights["content"] * position_score)

            # Checking in the item based collaborative filtering list
            if item in item_cf_based_map:
                # Getting the position score
                position = item_cf_based_map[item]
                length_of_list = len(item_collaborative_filtering_recommendations)
                position_score = (length_of_list - position) / length_of_list
                # Adding to the combined score multiplying by weight
                combined_score = combined_score + (weights["item_cf"] * position_score)

            # Checking in the user based collaborative filtering list
            if item in user_cf_based_map:
                # Getting the position score
                position = user_cf_based_map[item]
                length_of_list = len(user_collaborative_filtering_recommendations)
                position_score = (length_of_list - position) / length_of_list
                # Adding to the combined score multiplying by weight
                combined_score = combined_score + (weights["user_cf"] * position_score)

            # Storing into the map
            combined_score_list.append((item, combined_score))

        # Sorting the list
        combined_score_list = bubble_sort_list_with_tuples_second_element(combined_score_list)

        # Extracted just the items
        final_item_list = []
        for item_score_tuple in combined_score_list:
            final_item_list.append(item_score_tuple[0])

        # Returning the top N elements
        return final_item_list[:n]