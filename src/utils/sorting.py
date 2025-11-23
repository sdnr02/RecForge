from typing import Dict, List, Tuple

def bubble_sort_map_by_values(data_map: Dict) -> Dict:
        """Method to sort the map by values"""
        # Converting the map to a list of tuples
        item_list = []
        for key,item in data_map.items():
            item_list.append((key,item))
        
        # Getting length of the dictionary
        length_of_item_list = len(item_list)

        # Outer loop for first item
        for i in range(0, length_of_item_list):
            # Initializing variable to track swapping
            item_swapped = False

            # In bubble sort, that last i items are already in place
            for j in range(0, length_of_item_list-i-1):
                # Comparing the values of the items
                if item_list[j][1] < item_list[j+1][1]:
                    # Swap the elements
                    temp_var = item_list[j]
                    item_list[j] = item_list[j+1]
                    item_list[j+1] = temp_var

                    # Mark that a swap did happen
                    item_swapped = True

            # If no swap happened during the inner loop, that means elements are already sorted
            if not item_swapped:
                break

        # Converting the list of tuples back to a dictionary
        sorted_data_map = dict(item_list)
        return sorted_data_map

def bubble_sort_list_with_tuples(list_with_tuples: List[Tuple[str, str, int]]) -> List[Tuple[str,str,int]]:
        """Method to bubble sort a list with tuples in them"""
        # Iterating through the outer loop for item 1
        for i in range(0, len(list_with_tuples)):
            # Iterating through inner loop for item 2
            for j in range(i+1, len(list_with_tuples)):
                # Checking if item 1 < item 2 since we want to sort it in descending
                if list_with_tuples[i][2] < list_with_tuples[j][2]:
                    # Swapping the elements
                    temp_tuple = list_with_tuples[i]
                    list_with_tuples[i] = list_with_tuples[j]
                    list_with_tuples[j] = temp_tuple

        # Now the list is sorted
        return list_with_tuples
