from typing import List

# https://leetcode.com/problems/candy/
class Solution:
    def candy(self, ratings: List[int]) -> int:
        def slope(i):
            if i < 0 or i >= n:
                return None

            if ratings[i] > ratings[i + 1]:
                return -1
            elif ratings[i] < ratings[i + 1]:
                return 1
            else:
                return 0

        n = len(ratings)

        steep_arr = [slope(i) for i in range(n - 1)]

        candies = [1] * n

        for left in range(n - 1):
            if steep_arr[left] == 1 and candies[left + 1] <= candies[left]:
                candies[left + 1] = candies[left] + 1

        for right in range(n - 1, 0, -1):
            if steep_arr[right - 1] == -1 and candies[right] >= candies[right - 1]:
                candies[right - 1] = candies[right] + 1

        return sum(candies)