from typing import Any

class MinHeap:
    """Class to implement the min-heap data structure from scratch"""

    def __init__(self) -> None:
        """"Initializing the heap object"""
        self.heap = []

    def __len__(self) -> None:
        """Finding the length of the heap"""
        # Using len function on the array
        return len(self.heap)
    
    def _get_parent_index(self, i: int) -> int:
        """In a Min Heap, the parent is at the beginning of the array"""
        # Mathematically, the child node is 16 -> 8 -> 4 -> 2 and so on
        return (i-1) // 2
    
    def _get_left_child_index(self, i: int) -> int:
        """In a Min Heap, the child node is after that node but one index before the right child"""
        # Mathematically, the left node is at 0 -> 1 -> 3 -> 7 and so on
        return (2*i) + 1
    
    def _get_right_child_index(self, i: int) -> int:
        """In a Min Heap, the child node is after that node but one index after the left child"""
        # Mathematically, the left node is at 0 -> 2 -> 4 -> 8 and so on
        return (2*i) + 2
    
    def _has_parent(self, i: int) -> bool:
        """Helper function to check if a node has a parent node"""
        # Checking if a node at index 'i' has a parent node with an index within the scope of the array
        if self._get_parent_index(i) >= 0:
            return True
        else:
            return False
    
    def _has_left_child(self, i: int) -> bool:
        """Helper function to check if a node has a left child node"""
        # Checking if the node at index 'i' has a left child node with an index within the scope of the array
        if self._get_left_child_index(i) < len(self.heap):
            return True
        else:
            return False
    
    def _has_right_child(self, i:int) -> bool:
        """Helper function to check if a node has a right child node"""
        # Checking if the node at index 'i' has a right child node with an index within the scope of the array
        if self._get_right_child_index(i) < len(self.heap):
            return True
        else:
            return False
    
    def _swap(self, i: int, j: int) -> None:
        """Helper function to swap two nodes in a heap"""
        # Swapping the elements using Tower of Hanoi strategy
        temp_node = self.heap[i]
        self.heap[j] = self.heap[i]
        self.heap[i] = temp_node

    def _bubble_up(self, index: int) -> None:
        """A helper method that preserves heap structure during adding an element"""
        # Continue performing the operation as long as current index has a parent node
        while self._has_parent(index):
            # Retrieving the parent index
            parent_index = self._get_parent_index(index)

            # Checking to see if the node at parent index is less than the node at current index
            if self.heap[parent_index] > self.heap[index]:
                # Swapping the nodes to maintain heap property
                self._swap(index,parent_index)
                index = parent_index # Move up

            # The heap property of parent < child is satisfied
            else:
                break

    def _bubble_down(self, index: int) -> None:
        """A helper method that moves the nodes down until the heap structure is preserved"""
        # While the node has atleast one valid child node
        while self._has_left_child(index):
            # Find child node index
            smaller_child_index = self._get_left_child_index(index)

            # Checking to see if node also has a right child
            if self._has_right_child(index):
                right_child_index = self._get_right_child_index(index)
                # Checking to see which is the smaller child - left or right child node
                if self.heap[smaller_child_index] > self.heap[right_child_index]:
                    smaller_child_index = right_child_index

            # Comparing the index element with the smaller child index
            if self.heap[index] > self.heap[smaller_child_index]:
                # Swapping the element if the current index has a child smaller than it
                self._swap(index, smaller_child_index)
                index = smaller_child_index # Move down

            # The min-heap property of parent < child is satisfied
            else:
                break

    def peek(self) -> int:
        """View the minimum element (root) without removing it"""
        # Nothing to return if heap is empty
        if len(self.heap) == 0:
            return None
        
        # Return the first element in the array as the minimum element
        return self.heap[0]
    
    def push(self, item: Any) -> None:
        """Adding the element to an array and positioning it to the right position on the heap"""
        # First we append to the end of the array
        self.heap.append(item)

        # Bubble up operation to restore the heap structure
        self._bubble_up(len(self.heap) - 1)

    def pop(self) -> Any:
        """Removes and returns the smallest element in the heap - the root node"""
        # Storing the value of the smallest element at the root index
        minimum_value = self.heap[0]

        # Moving the last element of the array to the root node (to in effect remove the root node)
        self.heap[0] = self.heap[-1]

        # Removing the duplicate last element of the array
        self.heap.pop()

        # Restructuring the heap to ensure that the order is correct
        if len(self.heap) > 0:
            # We start from the root node now and swap down
            self._bubble_down(0) 

        # Return the original root value
        return minimum_value