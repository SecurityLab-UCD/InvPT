from problems.hard.candy import Solution


def test_candy():
    solution = Solution()

    # Testcase 1
    ratings = [1, 0, 2]
    assert solution.candy(ratings) == 5

    # Testcase 2
    ratings = [1, 2, 2]
    assert solution.candy(ratings) == 4

    # Testcase 3
    ratings = [1, 2, 3]
    assert solution.candy(ratings) == 6

    # Testcase 4
    ratings = [3, 2, 1]
    assert solution.candy(ratings) == 6

    # Testcase 5
    ratings = [1, 5, 4, 2]
    assert solution.candy(ratings) == 7

    # Testcase 6
    ratings = [1, 5, 6, 4, 2]
    assert solution.candy(ratings) == 9

    # Testcase 7
    ratings = [6, 5, 1, 4, 2]
    assert solution.candy(ratings) == 9

    # Testcase 8
    ratings = [6, 5, 6, 4, 2]
    assert solution.candy(ratings) == 9