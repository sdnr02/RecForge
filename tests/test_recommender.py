# WARNING: This file was AI-Generated

import sys
sys.path.insert(0, 'src')

from core.user import User
from core.item import Item
from core.recommender import RecommendationEngine


def test_user_management():
    """Test adding and retrieving users"""
    print("Testing User Management...")
    engine = RecommendationEngine()
    
    # Add users
    alice = User("alice", preferences={"Sci-Fi": 0.9})
    bob = User("bob", demographics={"age": 25})
    engine.add_user(alice)
    engine.add_user(bob)
    
    # Test retrieval
    assert engine.get_user("alice").user_id == "alice"
    assert engine.get_user("bob").demographics["age"] == 25
    assert engine.get_user("charlie") is None
    
    # Test index mappings
    assert engine.user_to_index["alice"] == 0
    assert engine.user_to_index["bob"] == 1
    assert engine.index_to_user[0] == "alice"
    
    print("[PASS] User Management Passed!")


def test_item_management():
    """Test adding and retrieving items"""
    print("Testing Item Management...")
    engine = RecommendationEngine()
    
    # Add items
    inception = Item("inception", "Sci-Fi", tags=["mind-bending"])
    titanic = Item("titanic", "Romance")
    engine.add_item(inception)
    engine.add_item(titanic)
    
    # Test retrieval
    assert engine.get_item("inception").category == "Sci-Fi"
    assert engine.get_item("titanic").category == "Romance"
    assert engine.get_item("matrix") is None
    
    # Test index mappings
    assert engine.item_to_index["inception"] == 0
    assert engine.item_to_index["titanic"] == 1
    
    print("[PASS] Item Management Passed!")


def test_rating_system():
    """Test adding and retrieving ratings"""
    print("Testing Rating System...")
    engine = RecommendationEngine()
    
    # Setup
    engine.add_user(User("alice"))
    engine.add_user(User("bob"))
    engine.add_item(Item("inception", "Sci-Fi"))
    engine.add_item(Item("titanic", "Romance"))
    
    # Add ratings
    engine.add_rating("alice", "inception", 5.0)
    engine.add_rating("alice", "titanic", 2.0)
    engine.add_rating("bob", "inception", 5.0)
    
    # Test rating retrieval
    assert engine.get_rating("alice", "inception") == 5.0
    assert engine.get_rating("alice", "titanic") == 2.0
    assert engine.get_rating("bob", "inception") == 5.0
    
    # Test rating_store structure
    assert "alice" in engine.rating_store
    assert engine.rating_store["alice"]["inception"] == 5.0
    
    print("[PASS] Rating System Passed!")


def test_matrix_operations():
    """Test matrix and vector operations"""
    print("Testing Matrix Operations...")
    engine = RecommendationEngine()
    
    # Setup
    engine.add_user(User("alice"))
    engine.add_user(User("bob"))
    engine.add_user(User("carol"))
    engine.add_item(Item("inception", "Sci-Fi"))
    engine.add_item(Item("titanic", "Romance"))
    
    # Add ratings
    engine.add_rating("alice", "inception", 5.0)
    engine.add_rating("alice", "titanic", 2.0)
    engine.add_rating("bob", "inception", 5.0)
    engine.add_rating("carol", "titanic", 3.0)
    
    # Test user vector
    alice_vector = engine.get_user_vector("alice")
    assert alice_vector == [5.0, 2.0], f"Expected [5.0, 2.0], got {alice_vector}"
    
    bob_vector = engine.get_user_vector("bob")
    assert bob_vector == [5.0, 0], f"Expected [5.0, 0], got {bob_vector}"
    
    # Test item vector
    inception_vector = engine.get_item_vector("inception")
    assert inception_vector == [5.0, 5.0, 0], f"Expected [5.0, 5.0, 0], got {inception_vector}"
    
    titanic_vector = engine.get_item_vector("titanic")
    assert titanic_vector == [2.0, 0, 3.0], f"Expected [2.0, 0, 3.0], got {titanic_vector}"
    
    print("[PASS] Matrix Operations Passed!")


def test_matrix_expansion():
    """Test that matrix expands correctly"""
    print("Testing Matrix Expansion...")
    engine = RecommendationEngine()
    
    # Add first user and item
    engine.add_user(User("alice"))
    engine.add_item(Item("inception", "Sci-Fi"))
    assert len(engine.user_item_matrix) == 1
    assert len(engine.user_item_matrix[0]) == 1
    
    # Add second user
    engine.add_user(User("bob"))
    assert len(engine.user_item_matrix) == 2
    assert len(engine.user_item_matrix[1]) == 1
    
    # Add second item
    engine.add_item(Item("titanic", "Romance"))
    assert len(engine.user_item_matrix[0]) == 2
    assert len(engine.user_item_matrix[1]) == 2
    
    # Matrix should be 2x2 filled with zeros
    assert engine.user_item_matrix == [[0, 0], [0, 0]]
    
    print("[PASS] Matrix Expansion Passed!")


def test_integration():
    """Full integration test"""
    print("Testing Full Integration...")
    engine = RecommendationEngine()
    
    # Create complete system
    users = [
        User("alice", preferences={"Sci-Fi": 0.9}),
        User("bob", demographics={"age": 30}),
        User("carol")
    ]
    
    items = [
        Item("inception", "Sci-Fi", tags=["mind-bending"]),
        Item("titanic", "Romance", tags=["classic"]),
        Item("matrix", "Sci-Fi", tags=["action"])
    ]
    
    for user in users:
        engine.add_user(user)
    
    for item in items:
        engine.add_item(item)
    
    # Add ratings
    ratings = [
        ("alice", "inception", 5.0),
        ("alice", "titanic", 2.0),
        ("alice", "matrix", 4.0),
        ("bob", "inception", 5.0),
        ("bob", "matrix", 4.0),
        ("carol", "titanic", 3.0)
    ]
    
    for user_id, item_id, rating in ratings:
        engine.add_rating(user_id, item_id, rating)
    
    # Verify complete matrix
    expected_matrix = [
        [5.0, 2.0, 4.0],  # alice
        [5.0, 0, 4.0],    # bob
        [0, 3.0, 0]       # carol
    ]
    
    assert engine.user_item_matrix == expected_matrix, \
        f"Expected {expected_matrix}, got {engine.user_item_matrix}"
    
    # Verify item vectors
    inception_ratings = engine.get_item_vector("inception")
    assert inception_ratings == [5.0, 5.0, 0]
    
    print("[PASS] Integration Test Passed!")


def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("Running Unit Tests")
    print("=" * 50)
    
    try:
        test_user_management()
        test_item_management()
        test_rating_system()
        test_matrix_expansion()
        test_matrix_operations()
        test_integration()
        
        print("\n" + "=" * 50)
        print("[SUCCESS] ALL TESTS PASSED!")
        print("=" * 50)
        
    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n[FAIL] ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()