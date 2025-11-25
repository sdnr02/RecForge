from typing import TYPE_CHECKING, Dict, Tuple, List

if TYPE_CHECKING:
    from core.recommender import RecommendationEngine

from utils.sorting import bubble_sort_map_by_values
from utils.sorting import bubble_sort_list_with_tuples_second_element

class UserCollaborativeFiltering:
    """Class for implementing collaborative filtering for users"""

    def __init__(self, engine: RecommendationEngine) -> None:
        """Initializing the object"""
        self.engine = engine
        self.user_neighbourhoods: Dict[str, List[Tuple[str, float]]] = {}
        # We will call the build neighbourhood function here

    def build_user_neighbourhoods(self, k: int = 10) -> None:
        """Method to pre-compute and build the cache for the K-most similar users for each user"""
        # Iterating through the list of users
        for user_id in self.engine.user_store.keys():
            # Initializing the empty list of similarities
            similarities = []

            # Iterating through the list of users again to check with the remaining list of users
            for other_user_id in self.engine.user_store.keys():
                # Skipping comparison with the user itself
                if user_id == other_user_id:
                    continue
                
                # Extracting user similarity score between the two users
                similarity = self.engine.get_user_similarity(user_id, other_user_id)

                # Adding the similarity score
                similarities.append((other_user_id, similarity))

            # Sorting the list
            similarities = bubble_sort_list_with_tuples_second_element(similarities)

            # Retrieving the first K elements in the sorted list
            similarities = similarities[:k]
            self.user_neighbourhoods[user_id] = similarities
        
        # In memory modification of the user neighbourhood map so no return variable
        return
    
    def find_similar_users(
        self,
        user_id: str,
        k: int = 10
    ) -> List[str]:
        """Method to find the k-most similar users to the target user ID"""
        # First checking if the user already exists in the cache
        if user_id in self.user_neighbourhoods.keys():
            # Retrieving the pre-existing list of similar neighbours
            neighbours = self.user_neighbourhoods[user_id]

            # Iterating and appending since return format is a list
            neighbour_list = []
            for neighbour_id, _ in neighbours[:k]:
                neighbour_list.append(neighbour_id)
            
            # Returning the neighbour list
            return neighbour_list

        # If user ID not pre-existing in the cache map
        similarities = []

        # Iterating through the list of users
        for other_user_id in self.engine.user_store.keys():
            # Checking if this ID is same as the user ID
            if user_id == other_user_id:
                continue

            # Calculating similarity using the existing method
            similarity = self.engine.get_user_similarity(user_id, other_user_id)

            # Appending to the list of similarities
            similarities.append((other_user_id, similarity))

        # Sorting the similarity tuples
        similarities = bubble_sort_list_with_tuples_second_element(similarities)

        # Extracting user IDs from the list
        neighbour_list = []
        for neighbor_user_id, _ in similarities[:k]:
            neighbour_list.append(neighbor_user_id)

        # Returning top k elements from the list
        return neighbour_list
    
    def get_user_neighborhood(
        self,
        user_id: str,
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """Method to find the k-most similar users to the target user ID"""
        # First checking if the user already exists in the cache
        if user_id in self.user_neighbourhoods.keys():
            # Retrieving the pre-existing list of similar neighbours
            neighbours = self.user_neighbourhoods[user_id]

            # Iterating and appending since return format is a list
            neighbour_list = []
            for neighbour_id, similarity in neighbours[:k]:
                neighbour_list.append((neighbour_id, similarity))
            
            # Returning the neighbour list
            return neighbour_list

        # If user ID not pre-existing in the cache map
        similarities = []

        # Iterating through the list of users
        for other_user_id in self.engine.user_store.keys():
            # Checking if this ID is same as the user ID
            if user_id == other_user_id:
                continue

            # Calculating similarity using the existing method
            similarity = self.engine.get_user_similarity(user_id, other_user_id)

            # Appending to the list of similarities
            similarities.append((other_user_id, similarity))

        # Sorting the similarity tuples
        similarities = bubble_sort_list_with_tuples_second_element(similarities)

        # Returning top k elements from the list
        similarities =  similarities[:k]
        return similarities

    def predict_rating(
        self,
        user_id: str,
        item_id: str,
        k: int = 10
    ) -> float | None:
        """Method that predicts the rating score for a user-item pair"""
        # Getting the user neighbourhood for that particular user
        neighbourhood = self.get_user_neighborhood(user_id, k)

        # Initializing empty lists for later steps
        neighbor_ratings = []
        similarity_weights = []

        # Then find which of the neighbours rated this particular item
        for neighbour_id, similarity in neighbourhood:
            # Extracting the rating from the neighbour
            neighbour_rating = self.engine.get_rating(neighbour_id, item_id)

            # In case the neighbour has rated the item
            if neighbour_rating:
                # Adding the rating as well as the user similarity score
                neighbor_ratings.append(neighbour_rating)
                similarity_weights.append(similarity)

        # In case none of the similar users have rated the item, we return nothing
        if neighbor_ratings == []:
            return None
        
        # Calculating the weighted average
        weighted_sum = 0

        # Iterating through the list and aggregating the score
        for i in range(len(neighbor_ratings)):
            weighted_sum = weighted_sum + (neighbor_ratings[i] * similarity_weights[i])

        # Getting the total simlarities with the user
        total_similarity = 0
        for similarity_weight in similarity_weights:
            total_similarity = total_similarity + similarity_weight

        # Calculating the predicted rating
        predicted_rating = weighted_sum / total_similarity
        return predicted_rating
    
    def recommend(
        self,
        user_id: str,
        n: int = 10,
        k: int = 10
    ) -> List[str]:
        """Method that generates N item recommendations based on what K similar users liked"""
        # Getting the user neighbourhood for that particular user
        neighbourhood = self.get_user_neighborhood(user_id, k)
        
        # Initializing a dictionary to collect candidate items and their ratings/similarities
        candidate_items: Dict[str, List[Tuple[float, float]]] = {}
        
        # Iterating through each neighbour
        for neighbour_id, similarity in neighbourhood:
            # Getting all the ratings from this neighbour
            neighbour_ratings = self.engine.user_rating_store.get(neighbour_id, {})
            
            # Iterating through each item the neighbour rated
            for item_id, rating in neighbour_ratings.items():
                # Checking if the target user already rated this item
                user_rating = self.engine.get_rating(user_id, item_id)
                
                # If user hasn't rated it, it's a candidate
                if not user_rating:
                    # If this item is not yet in candidates, initialize it
                    if item_id not in candidate_items:
                        candidate_items[item_id] = []
                    
                    # Appending the rating and similarity as a tuple
                    candidate_items[item_id].append((rating, similarity))
        
        # Initializing a dictionary to store the final scores
        candidate_scores = {}
        
        # Scoring each candidate item
        for item_id, ratings_and_sims in candidate_items.items():
            # Initializing weighted sum and total similarity
            weighted_sum = 0
            total_similarity = 0
            
            # Iterating through all ratings and similarities for this item
            for rating, similarity in ratings_and_sims:
                weighted_sum = weighted_sum + (rating * similarity)
                total_similarity = total_similarity + similarity
            
            # Calculating the normalized score
            candidate_scores[item_id] = weighted_sum / total_similarity
        
        # Sorting the candidate scores using the existing sorting function
        candidate_scores = bubble_sort_map_by_values(candidate_scores)
        
        # Extracting the top N item IDs
        result_list = []
        count = 0
        
        for item_id, score in candidate_scores.items():
            if count < n:
                result_list.append(item_id)
                count = count + 1
            else:
                break
        
        # Returning the final list of recommendations
        return result_list
