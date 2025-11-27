import math
from typing import List, Dict

from core.user import User
from core.item import Item
from data_structures.heap import MinHeap
from algorithms.content_based import ContentBasedRecommender
from algorithms.item_collaborative_filtering import ItemCollaborativeFilterer
from algorithms.user_collaborative_filtering import UserCollaborativeFiltering

class RecommendationEngine:
    """Class that orchestrates all users and item management"""

    def __init__(self):
        """Initializing the engine object"""
        # Hash maps to store users, items and ratings
        self.user_store: Dict[str,User] = {}
        self.item_store: Dict[str,Item] = {}
        self.user_rating_store: Dict[str, Dict[str,float]] = {}
        self.item_rating_store: Dict[str, Dict[str,float]] = {}

        # Matrix that maps users to their ratings
        self.user_item_matrix: List[List[float]] = []
        
        # Bidirectional mapping of users/items to index in the matrix and vice versa
        self.user_to_index: Dict[str,int] = {}
        self.index_to_user: List[str] = []
        self.item_to_index: Dict[str,int] = {}
        self.index_to_item: List[str] = []

        # Initializing recommendation algorithms
        self.content_based_filtering = ContentBasedRecommender(self)
        self.item_based_collaborative_filtering = ItemCollaborativeFilterer(self)
        self.user_based_collaborative_filtering = UserCollaborativeFiltering(self)

    def add_user(self, user_object: User) -> None:
        """Adding the user object to the appropriate stores"""
        # Adding the user to the user storage map
        self.user_store[user_object.user_id] = user_object

        # Adding user ID to both user to index map and the index list
        length_of_index = len(self.index_to_user)
        self.user_to_index[user_object.user_id] = length_of_index
        self.index_to_user.append(user_object.user_id)

        # Getting the length of the existing list of items
        number_of_items = len(self.index_to_item)

        # Adding a new subarray into the list with the same number of 0s as there are number of items
        new_subarray = []
        for i in range(0,number_of_items):
            new_subarray.append(0)

        # Adding to the matrix
        self.user_item_matrix.append(new_subarray)

    def get_user(self, user_id: str) -> User | None:
        """Retrieving the user object from the User store"""
        # O(1) lookup from the Hash Map
        user_object = self.user_store.get(user_id)
        
        # Returning only if the user object exists in the map
        if user_object:
            return user_object
        else:
            return None
        
    def add_item(self, item_object: Item) -> None:
        """Adding the item object to the appropriate stores"""
        # Adding the item to the item storage map
        self.item_store[item_object.item_id] = item_object

        # Adding the item ID to both the item to index map and the index list
        length_of_index = len(self.index_to_item)
        self.item_to_index[item_object.item_id] = length_of_index
        self.index_to_item.append(item_object.item_id)

        # Adding 0 to each user's ratings to indicate a new item has been added
        for subarray  in self.user_item_matrix:
            subarray.append(0)

    def get_item(self, item_id: str) -> Item | None:
        """Retrieving the item object from the Item store"""
        # O(1) lookup from the Hash Map
        item_object = self.item_store.get(item_id)

        # Returning it when the item object exists
        if item_object:
            return item_object
        else:
            return None
        
    def add_rating(
        self,
        user_id: str,
        item_id: str,
        rating: float
    ) -> None:
        """Initializing a user-item rating to the Ratings store"""
        # Retrieving the existing user rating map from the store
        user_map = self.user_rating_store.get(user_id)
        
        # If user map exists we simply update it to add the new item rating
        if user_map:
            user_map[item_id] = rating
            self.user_rating_store[user_id] = user_map

        # If the user map does not exist we create a new map and add the item rating
        else:
            user_map = {}
            user_map[item_id] = rating
            self.user_rating_store[user_id] = user_map

        # Retrieving the existing item rating map from the store
        item_map = self.item_rating_store.get(item_id)

        # If the item map exists we simply update it to add the new user rating
        if item_map:
            item_map[user_id] = rating
            self.item_rating_store[item_id] = item_map

        # If the item map does not exist we create a new map and add the user rating to that
        else:
            item_map = {}
            item_map[user_id] = rating
            self.item_rating_store[item_id] = item_map

        # Working off the assumption that user-item indices exist in the matrix
        # Finding the indices of users and items from the index lookups: O(1) again!
        user_index = self.user_to_index[user_id]
        item_index = self.item_to_index[item_id]

        # Adding the rating to the user-item ratings matrix
        self.user_item_matrix[user_index][item_index] = rating

    def get_rating(
        self,
        user_id: str,
        item_id: str
    ) -> float | None:
        """Method that retrieves the rating for the particular item by a user"""
        # Retrieving the rating from the hashmap for faster lookup
        user_ratings = self.user_rating_store.get(user_id, {})
        rating = user_ratings.get(item_id)

        # Returning the rating if it exists
        if rating:
            return rating
        else:
            return None
        
    def get_user_vector(self, user_id: str) -> List:
        """Method that retrieves the entire subarray for a particular user"""
        # Retrieving the index from the user index map
        user_index = self.user_to_index.get(user_id)

        # Retrieving that specific subarray from the 2D matrix
        user_row = self.user_item_matrix[user_index]
        return user_row
    
    def get_item_vector(self, item_id: str) -> List:
        """Method that retrieves the entire column for a particular item"""
        # Retrieving the index from the item index map
        item_index = self.item_to_index.get(item_id)

        # Retrieving that specific column from the 2D matrix
        item_column = []
        for subarray in self.user_item_matrix:
            item_column.append(subarray[item_index])
        
        # Returning the array with the item ratings
        return item_column
    
    def get_item_similarity(
        self,
        item1_id: str,
        item2_id: str
    ) -> float:
        """Method to calculate the similarity between two items using cosine similarity"""
        # Extracting the ratings data for two items
        item1_ratings_map = self.item_rating_store.get(item1_id)
        item2_ratings_map = self.item_rating_store.get(item2_id)

        if not item1_ratings_map or not item2_ratings_map:
            return 0.0

        # Finding the common keys between the two hash maps
        common_keys = item1_ratings_map.keys() & item2_ratings_map.keys()
        
        # Initializing sum variable for sum = (a1*b1) + (a2*b2) + (a3*b3) + ....
        sum = 0.0

        # Looping through the common keys and finding the dot product for the two vectors
        if common_keys:
            for key in common_keys:
                sum = sum + (item1_ratings_map[key] * item2_ratings_map[key])

        # Finding the magnitude of the first item (sqrt(a1^2 + a2^2 + a3^2 + ....))
        item1_sum = 0.0
        for rating in item1_ratings_map.values():
            item1_sum = item1_sum + (rating**2)
        
        # Finding the square root
        item1_magnitude = math.sqrt(item1_sum)

        # Finding the magnitude of the second item (sqrt(b1^2 + b2^2 + b3^2 + ....))
        item2_sum = 0.0
        for rating in item2_ratings_map.values():
            item2_sum = item2_sum + (rating**2)

        # Finding the square root
        item2_magnitude = math.sqrt(item2_sum)

        if item1_magnitude == 0 or item2_magnitude == 0:
            return 0.0
        
        return sum / (item1_magnitude * item2_magnitude)
    
    def get_user_similarity(
        self,
        user1_id: str,
        user2_id: str
    ) -> float:
        """Method to calculate the similarity between two users using cosine similarity"""
        # Extracting the ratings data for two users
        user1_ratings_map = self.user_rating_store.get(user1_id)
        user2_ratings_map = self.user_rating_store.get(user2_id)

        if not user1_ratings_map or not user2_ratings_map:
            return 0.0

        # Finding the common keys (items both users rated)
        common_keys = user1_ratings_map.keys() & user2_ratings_map.keys()
        
        # Initializing sum variable for dot product
        sum = 0.0

        # Looping through the common keys and finding the dot product
        if common_keys:
            for key in common_keys:
                sum = sum + (user1_ratings_map[key] * user2_ratings_map[key])

        # Finding the magnitude of the first user
        user1_sum = 0.0
        for rating in user1_ratings_map.values():
            user1_sum = user1_sum + (rating**2)
        
        user1_magnitude = math.sqrt(user1_sum)

        # Finding the magnitude of the second user
        user2_sum = 0.0
        for rating in user2_ratings_map.values():
            user2_sum = user2_sum + (rating**2)

        user2_magnitude = math.sqrt(user2_sum)

        if user1_magnitude == 0 or user2_magnitude == 0:
            return 0.0
        
        similarity_score = sum / (user1_magnitude * user2_magnitude)
        return similarity_score
    
    def get_average_rating_for_item(self, item_id: str) -> float:
        """Method to calculate and retrieve the average rating for a particular item"""
        # Retrieving the item map for that particular item ID
        item_ratings_map = self.item_rating_store.get(item_id)

        # Ensuring that the map exists
        if not item_ratings_map:
            return 0.0
        
        # Iterating through the dictionary values and calculating sum
        sum = 0.0
        for rating in item_ratings_map.values():
            sum = sum + rating

        # Calculating the average rating and returning it
        average_rating = sum / (len(item_ratings_map))
        return average_rating
    
    def get_item_popularity(self, item_id: str) -> int:
        """Method to retrieve how many users rated this item?"""
        # Simply finding the length of the item's rating subarray
        return len(self.item_rating_store[item_id])
    
    def get_top_k_popular_items(self, k: int = 10) -> List[str]:
        """Method to retrieve the top k most popular items using a min heap"""
        # Initializing the min heap object
        min_heap = MinHeap()

        # Iterating through the rating store
        for item_id, _ in self.item_rating_store.items():
            # Getting how many people rated the item
            popularity = self.get_item_popularity(item_id)

            # Since we want the top k elements the root node in the heap will contain the smallest of the current top k
            # This root node element is what will get replaced anyways if the new node is supposed to be in the top k
            if len(min_heap) < k:
                # Simply pushing the element if the current heap is less than k elements
                min_heap.push((popularity, item_id))
            
            # In case the heap is full
            else:
                # Checking if the current item is supposed to be in the top k items
                if popularity > min_heap.peek()[0]:
                    # Remove the smallest node
                    min_heap.pop()
                    # Add the new item
                    min_heap.push((popularity, item_id))

        # While extracting results from a min heap we get the elements in ascending order
        result = []
        while len(min_heap) > 0:
            _, item_id = min_heap.pop()
            result.append(item_id)

        # Initializing the two pointers
        left = 0
        right = len(result) - 1

        # Reversing the list that is in ascending order to descending
        # To ensure the larger popularity elements appear first
        while left < right:
            # Simply need to swap since the list is already in ascending order
            temp_var = result[left]
            result[left] = result[right]
            result[right] = temp_var

            # Moving the pointers
            left = left + 1
            right = right - 1

        # Finally returning the top k element list
        return result
    
    def get_top_k_similar_items(
        self,
        item_id: str,
        k: int = 5
    ) -> List[str]:
        """Method to retrieve the top k most similar items to a particular item"""
        # Initializing the heap object
        min_heap = MinHeap()

        # Iterating through the item rating map
        for store_item_id, _ in self.item_rating_store.items():
            # Calculating item similarity between the two items if they are not the same item
            if item_id != store_item_id:
                similarity_score = self.get_item_similarity(
                    item1_id=item_id,
                    item2_id=store_item_id
                )

                # Checking if the length of heap is less than k and simply pushing if True
                if len(min_heap) < k:
                    min_heap.push((similarity_score, store_item_id))

                # In case the heap is already full
                else:
                    # Checking if similarity score is greater than the smallest similarity in the heap
                    if similarity_score > min_heap.peek()[0]:
                        # Removing the smallest node and adding the new item
                        min_heap.pop()
                        min_heap.push((similarity_score, store_item_id))

        # Extracting results for top k similar items in ascending order
        result = []
        while len(min_heap) > 0:
            _, similar_item_id = min_heap.pop()
            result.append(similar_item_id)

        # Initializing two pointers
        left = 0
        right = len(result) - 1

        # Reversing the list to descending order
        while left < right:
            temp_var = result[left]
            result[left] = result[right]
            result[right] = temp_var

            # Moving the pointers
            left = left + 1
            right = right - 1

        # Returning top k similar items
        return result
    
    def explain_recommendations(
        self,
        user_id: str,
        item_id: str
    ) -> List[str]:
        """Method to generate explanations for the recommendations to improve observability"""
        # Initializing an empty list to store the explanation
        explanations = []

        # Explanations from the content-based preferences
        user_preferences = self.content_based_filtering.extract_user_preferences(user_id)
        item = self.get_item(item_id)

        # Checking if user preferences exist
        if user_preferences:
            # Checking if the item category is in the user preferences
            if item.category in user_preferences:
                # Getting the weight for that category
                weight = user_preferences[item.category]
                # Creating the explanation string
                explanation = f"You rated {item.category} items highly (preference: {weight:.0%})"
                # Adding to the explanations list
                explanations.append(explanation)

            # Checking if the item tags exist
            if item.tags:
                # Iterating through the tags
                for tag in item.tags:
                    # Checking if the tag is in user preferences
                    if tag in user_preferences:
                        # Getting the weight for that tag
                        weight = user_preferences[tag]
                        # Creating the explanation string
                        explanation = f"Matches your interest in {tag} (preference: {weight:.0%})"
                        # Adding to the explanations list
                        explanations.append(explanation)

        # Explanations from item-based collaborative filtering
        user_ratings = self.user_rating_store.get(user_id, {})

        # Iterating through the user's ratings
        for rated_item, rating in user_ratings.items():
            # Checking if the user rated it highly
            if rating >= 4.0:
                # Calculating similarity between the rated item and the recommended item
                similarity = self.get_item_similarity(rated_item, item_id)

                # Checking if the similarity is high
                if similarity >= 0.7:
                    # Creating the explanation string
                    explanation = f"Similar to '{rated_item}' which you rated {rating}★ (similarity: {similarity:.0%})"
                    # Adding to the explanations list
                    explanations.append(explanation)

        # Explanations from user-based collaborative filtering
        similar_users = self.user_based_collaborative_filtering.find_similar_users(user_id, k=10)

        # Initializing counter for high ratings
        high_rating_count = 0

        # Iterating through similar users
        for similar_user_id in similar_users:
            # Getting the rating from the similar user
            rating = self.get_rating(similar_user_id, item_id)

            # Checking if the rating exists and is high
            if rating and rating >= 4.0:
                # Incrementing the counter
                high_rating_count = high_rating_count + 1

        # Checking if any similar users rated it highly
        if high_rating_count > 0:
            # Creating the explanation string
            explanation = f"{high_rating_count} users with similar taste to yours rated this highly"
            # Adding to the explanations list
            explanations.append(explanation)

        # Returning the list of explanations
        return explanations
    
    def batch_recommend(
        self,
        user_ids: List[str],
        n: int = 10
    ) -> Dict[str, List[str]]:
        """Method to generate recommendations for multiple users at once"""
        # Checking if user neighborhoods are built
        if not self.user_based_collaborative_filtering.user_neighbourhoods:
            # Building the user neighborhoods for faster lookups
            self.user_based_collaborative_filtering.build_user_neighbourhoods(k=10)

        # Checking if cooccurrence matrix is built
        if not self.item_based_collaborative_filtering.cooccurence_matrix:
            # Building the cooccurrence matrix for faster lookups
            self.item_based_collaborative_filtering.build_cooccurence_matrix()

        # Checking if category index is built
        if not self.content_based_filtering.category_index:
            # Building the category index for faster lookups
            self.content_based_filtering.build_category_index()

        # Checking if tag index is built
        if not self.content_based_filtering.tag_index:
            # Building the tag index for faster lookups
            self.content_based_filtering.build_tag_index()

        # Initializing the results dictionary
        results = {}

        # Iterating through all user IDs
        for user_id in user_ids:
            # Getting hybrid recommendations for that user
            from algorithms.hybrid import HybridRecommender
            
            # Initializing the hybrid recommender
            hybrid_recommender = HybridRecommender(
                content_based_filterer=self.content_based_filtering,
                item_based_collaborative_filterer=self.item_based_collaborative_filtering,
                user_based_collaborative_filterer=self.user_based_collaborative_filtering
            )
            
            # Getting the recommendations
            recommendations = hybrid_recommender.get_hybrid_recommendations(user_id, n)
            
            # Adding to the results dictionary
            results[user_id] = recommendations

        # Returning the results dictionary
        return results
    
    def recommend_with_filters(
        self,
        user_id: str,
        n: int = 10,
        exclude_categories: List[str] = None,
        min_rating: float = None
    ) -> List[str]:
        """Method to generate recommendations with user-specified filters"""
        # Getting more candidates than needed since many will be filtered
        from algorithms.hybrid import HybridRecommender
        
        # Initializing the hybrid recommender
        hybrid_recommender = HybridRecommender(
            content_based_filterer=self.content_based_filtering,
            item_based_collaborative_filterer=self.item_based_collaborative_filtering,
            user_based_collaborative_filterer=self.user_based_collaborative_filtering
        )
        
        # Getting three times as many candidates
        candidates = hybrid_recommender.get_hybrid_recommendations(user_id, n=3*n)

        # Initializing the filtered list
        filtered = []

        # Iterating through the candidates
        for item_id in candidates:
            # Getting the item object
            item = self.get_item(item_id)

            # Checking if item exists
            if not item:
                # Skip to next item
                continue

            # Filter 1: Checking exclude categories
            if exclude_categories:
                # Checking if item category is in the excluded list
                if item.category in exclude_categories:
                    # Skip this item
                    continue

            # Filter 2: Checking minimum rating
            if min_rating:
                # Getting the average rating for the item
                avg_rating = self.get_average_rating_for_item(item_id)
                # Checking if it meets the minimum
                if avg_rating < min_rating:
                    # Skip this item
                    continue

            # Filter 3: Checking if user already rated it
            user_rating = self.get_rating(user_id, item_id)
            if user_rating:
                # Skip this item
                continue

            # Item passed all filters so add it to the list
            filtered.append(item_id)

            # Checking if we have enough items
            if len(filtered) >= n:
                # Stop searching
                break

        # Returning the filtered list with top N items
        return filtered[:n]