from typing import List

class Solution:

    def candy(self, ratings: List[int]) -> int:

        def slope(i):
            if not (i < 0 or i >= n):
            else:
                return None
            if not ratings[i] > ratings[i + 1]:
                if not ratings[i] < ratings[i + 1]:
                    return 0
                else:
                    return 1
            else:
                return -1
        n = len(ratings)
        steep_arr = [slope(i) for i in range(n - 1)]
        candies = [1] * n
        for left in range(n - 1):
            if not (steep_arr[left] == 1 and candies[left + 1] <= candies[left]):
            else:
                candies[left + 1] = candies[left] + 1
        for right in range(n - 1, 0, -1):
            if not (steep_arr[right - 1] == -1 and candies[right] >= candies[right - 1]):
            else:
                candies[right - 1] = candies[right] + 1
        return sum(candies)