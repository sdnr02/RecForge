from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from core.recommender import RecommendationEngine

class RecommenderMetrics:
    """Class that implements evaluation methods for the recommender system"""

    @staticmethod
    def calculate_precision_at_k(
        recommendations: List[str],
        relevant_items: List[str],
        k: int
    ) -> float:
        """Method to calculate out of the top K items we recommended, how many were actually good"""
        # Checking that the k value is greater than 0
        if k <= 0:
            return 0

        # Retrieve the top K recommendations
        top_k_recommendations = recommendations[:k]

        # Initalizing a counter to get the total number of elements that were relevant
        relevant_item_count = 0

        # Counting the number of recommendations in relevant items
        for item in top_k_recommendations:
            if item in relevant_items:
                relevant_item_count = relevant_item_count + 1

        # Calculating precision
        precision = relevant_item_count / k
        return precision
    
    @staticmethod
    def calculate_recall_at_k(
        recommendations: List[str],
        relevant_items: List[str],
        k: int
    ) -> float:
        """Method to calculate of all the good items, how many did we recommend in top K"""
        # Checking that the k value is greater than 0
        if k <= 0:
            return 0
        
        # Retrieve the top K recommendations
        top_k_recommendations = recommendations[:k]

        # Initializing a counter to get the total number of elements that were relevant
        relevant_item_count = 0

        # Counting the number of recommendations in relevant items
        for item in top_k_recommendations:
            if item in relevant_items:
                relevant_item_count = relevant_item_count + 1

        # Calculating recall
        recall = relevant_item_count / len(relevant_items)
        return recall
    
    @staticmethod
    def calculate_coverage(
        all_recommendations: List[List[str]],
        total_items: int
    ) -> float:
        """What % of our catalog was actually recommended"""
        # Ensuring that there are items in the store
        if total_items <= 0:
            return 0

        # Initializing a set to store all unique items
        unique_items_recommended = set()

        # Iterating through all the items recommended
        for user_recommendations in all_recommendations:
            for item in user_recommendations:
                unique_items_recommended.add(item)

        # Counting the unique items
        unique_item_count = len(unique_items_recommended)

        # Calculating coverage
        coverage = unique_item_count / total_items
        return coverage
    
    @staticmethod
    def calculate_diversity(
        recommendations: List[str],
        engine: RecommendationEngine
    ) -> float:
        """Method to measure how different recommended items are too each other"""
        # Checking to see that there are any items
        if len(recommendations) <= 0:
            return 0.0
        
        # If there is only one item then the similarity is max
        if len(recommendations) == 1:
            return 1.0
        
        # Initializing a list to store all the pair-wise similarities
        similarity_list = []

        # Iterating through the list of recommendations
        for i in range(0, len(recommendations)):
            for j in range(i+1, len(recommendations)):
                item1 = recommendations[i]
                item2 = recommendations[j]

                # Estimating the item similarity between the two
                similarity = engine.get_item_similarity(item1, item2)
                similarity_list.append(similarity)

        # Checking to see that similarities exist
        if len(similarity) <= 0:
            return 0.0
        
        # Calculating the average similarity
        average_similarity = sum(similarity_list) / len(similarity_list)

        # Diversity is the complement of similarity
        diversity = 1 - average_similarity
        return diversity