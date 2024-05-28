import heapq

class MedianFinder:
    def __init__(self):
        self.min_heap = []  # Min heap for larger half
        self.max_heap = []  # Max heap for smaller half

    def add_num(self, num):
        if not self.max_heap or num <= -self.max_heap[0]:
            heapq.heappush(self.max_heap, -num)
        else:
            heapq.heappush(self.min_heap, num)

        # Balance the heaps if necessary
        if len(self.max_heap) > len(self.min_heap) + 1:
            heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        elif len(self.min_heap) > len(self.max_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))

    def find_median(self):
        if len(self.max_heap) == len(self.min_heap):
            return (-self.max_heap[0] + self.min_heap[0]) / 2
        else:
            return -self.max_heap[0]

# Example usage:
median_finder = MedianFinder()
traffic_data = [100, 200, 150, 300, 250, 400, 350]

for data_point in traffic_data:
    median_finder.add_num(data_point)
    print("Median so far:", median_finder.find_median())
