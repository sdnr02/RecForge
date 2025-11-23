from typing import TYPE_CHECKING, Dict, Tuple, List

if TYPE_CHECKING:
    from core.recommender import RecommendationEngine

from utils.sorting import bubble_sort_list_with_tuples, bubble_sort_map_by_values

class CollaborativeFilterer:
    """Class that implements collaborative filtering strategy for recommendation"""

    def __init__(self, engine: RecommendationEngine) -> None:
        """Initializing the collaborative filtering object"""
        self.engine = engine
        self.cooccurence_matrix: Dict[Tuple[str,str], int] = {}

    def build_cooccurence_matrix(self) -> Dict[Tuple[str,str], int]:
        """Method to build the co-occurence matrix based off existing user ratings"""
        # Re-initializing the coocurrence matrix
        self.cooccurence_matrix = {}

        # Iterating through the existing user ratings
        for user_id, _ in self.engine.user_rating_store.items():
            # Extracting the keys of the user rating store into a list
            list_of_items = list(self.engine.user_rating_store.get(user_id).keys())

            if not list_of_items:
                continue

            # Looping through individual items
            for i in range(0, len(list_of_items)):
                for j in range(i+1, len(list_of_items)):
                    # Storing the two items
                    item1 = list_of_items[i]
                    item2 = list_of_items[j]

                    # Initializiing an empty tuple
                    item_tuple = ()

                    # Sorting the items into alphabetic order to prevent duplicates
                    if item1 < item2:
                        item_tuple = (item1, item2)
                    else:
                        item_tuple = (item2, item1)

                    # If item tuple does not exist we initialize it with count of 1
                    if item_tuple not in self.cooccurence_matrix.keys():
                        self.cooccurence_matrix[item_tuple] = 1
                    
                    # If item tuple exists we increment its count by 1
                    else:
                        self.cooccurence_matrix[item_tuple] = self.cooccurence_matrix[item_tuple] + 1

        return self.cooccurence_matrix
    
    def get_item_coocurrence(
        self,
        item1_id: str,
        item2_id: str
    ) -> int:
        """Method to retrieve the coocurrence between two items"""
        # Initializing an empty tuple
        item_tuple = ()
        
        # Sorting the items into alphabetic order to find the right key
        if item1_id < item2_id:
            item_tuple = (item1_id, item2_id)
        else:
            item_tuple = (item2_id, item1_id)

        # Retrieving the coocurrence values
        coocurrence = self.cooccurence_matrix.get(item_tuple)

        # Retrieving the value of coocurrence if it exists
        if coocurrence:
            return coocurrence
        else:
            return 0
        
    def find_frequent_pairs(self, min_count: int = 10) -> List[Tuple]:
        """Method to retrieve pairs with atleast min_count number of coocurrences"""
        # Initializing an empty list to store the pairs
        frequent_pair_list = []

        # Iterating through the coocurrence matrix
        for item_tuple, coocurrence in self.cooccurence_matrix.items():
            # Comparing coocurrency with the minimum coocurrency required
            if coocurrence > min_count:
                # Adding the individual items and the coocurrence for subsequent sorting steps
                frequent_pair_list.append((item_tuple[0], item_tuple[1], coocurrence))

        # Sorting the list with frequent pairs
        frequent_pair_list = bubble_sort_list_with_tuples(frequent_pair_list)
        return frequent_pair_list
    
    def recommend(
        self,
        user_id: str,
        n: int = 10
    ) -> List[str]:
        """Method to generate N recommendations for a user similar to what they've rated"""
        # Retrieving the rating map for that particular user
        user_rating_map = self.engine.user_rating_store.get(user_id)

        # Returning in case map doesn't exist
        if not user_rating_map:
            return []
        
        # Initializing a map for aggregating scores
        candidate_scores = {}

        # Iterating through the items that the user has rated
        for item_id, rating in user_rating_map.items():
            # Retrieve the most similar items to that item
            similar_items = self.engine.get_top_k_similar_items(item_id, k=10)

            # Looping throught the list of similar items
            for similar_item_id in similar_items:
                # Retrieving the similarity score
                similarity = self.engine.get_item_similarity(item_id, similar_item_id)

                # Calculating the score based on simlarity and rating from the user
                score = similarity * rating

                # Checking to see if the similar item exists or not
                if similar_item_id not in candidate_scores:
                    candidate_scores[similar_item_id] = 0
                
                # Adding the score to the map since more occurences = better
                candidate_scores[similar_item_id] = candidate_scores[similar_item_id] + score

        # Initializing final map for storing candidates not rated by the user
        final_candidate_scores = {}

        # Filtering out items that are already rated
        for item_id, score in candidate_scores.items():
            if item_id not in user_rating_map.keys():
                final_candidate_scores[item_id] = score

        # Sorting the final candidate score map
        final_candidate_scores = bubble_sort_map_by_values(final_candidate_scores)

        # Returning only the first n elements of this score map
        return list(final_candidate_scores.keys())[:n]
