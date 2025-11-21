import math
from typing import List, Dict

from core.user import User
from core.item import Item
from data_structures.heap import MinHeap

class RecommendationEngine:
    """Class that orchestrates all users and item management"""

    def __init__(self):
        """Initializing the engine object"""
        # Hash maps to store users, items and ratings
        self.user_store: Dict = {}
        self.item_store: Dict = {}
        self.user_rating_store: Dict[Dict] = {}
        self.item_rating_store: Dict[Dict] = {}

        # Matrix that maps users to their ratings
        self.user_item_matrix: List[List] = []
        
        # Bidirectional mapping of users/items to index in the matrix and vice versa
        self.user_to_index: Dict = {}
        self.index_to_user: List = []
        self.item_to_index: Dict = {}
        self.index_to_item: List = []

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