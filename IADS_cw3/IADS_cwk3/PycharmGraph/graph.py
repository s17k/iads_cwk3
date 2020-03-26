import math
import sys
import itertools
import random

# This is a constant that determines the maximum number of cuts we consider (up to 4)
# As making it 4 slows down my heuristic significantly and, from what I saw,
#  doesn't actually change the results, this is currently set to 3 (but the tests were performed with 4).
MAX_M_OPT = 3

def euclid(p, q):
    x = p[0] - q[0]
    y = p[1] - q[1]
    return math.sqrt(x * x + y * y)


class Graph:
    # Complete as described in the specification, taking care of two cases:
    # the -1 case, where we read points in the Euclidean plane, and
    # the n>0 case, where we read a general graph in a different format.
    # self.perm, self.dists, self.n are the key variables to be set up.
    def __init__(self, n, filename):
        if n == -1:
            self.readEuclidean(filename)
        else:
            self.readGeneral(n, filename)
        self.perm = [i for i in range(self.n)]

    def readEuclidean(self, filename):
        with open(filename, "r") as input_file:
            lines = list(input_file)
            self.n = len(lines)
            vertices_coords = [self.parse_line_euclidean(line) for line in
                               lines]
            self.dists = self.empty_dists()
            for i in range(self.n):
                for j in range(self.n):
                    if i != j:
                        self.dists[i][j] = euclid(vertices_coords[i],
                                                  vertices_coords[j])

                        # converts a line string into a vertex represented as a coordinates list

    @staticmethod
    def parse_line_euclidean(line):
        blocks = line.split()
        if len(blocks) != 2:
            raise ValueError("wrong number of numbers in a line")
        nums = [int(b) for b in blocks]
        return [nums[0], nums[1]]

    def readGeneral(self, n, filename):
        self.n = n
        self.dists = self.empty_dists()
        with open(filename, "r") as input_file:
            lines = list(input_file)
            edges = [self.parse_line_general(line) for line in lines]
            for i, j, c in edges:
                self.dists[i][j] = self.dists[j][i] = c

    # converts a line string into an edge represented as a tuple
    @staticmethod
    def parse_line_general(line):
        blocks = line.split()
        if len(blocks) != 3:
            raise ValueError("wrong number of numbers in a line")
        nums = [int(b) for b in blocks]
        return (nums[0], nums[1], nums[2])

    def empty_dists(self):
        return [[0.0 for __ in range(self.n)] for _ in range(self.n)]

    # Complete as described in the spec, to calculate the cost of the
    # current tour (as represented by self.perm).
    def tourValue(self):
        return sum(self.dists[x][y] for x, y in
                   zip(self.perm, [self.perm[-1]] + self.perm))

    def prev(self, i):
        return (i + self.n - 1) % self.n

    def nxt(self, i):
        return (i + 1) % self.n

    # indexed distance (distance between perm[i], perm[j])
    def idx_distance(self, i, j):
        return self.dists[self.perm[i]][self.perm[j]]

    # Attempt the swap of cities i and i+1 in self.perm and commit
    # commit to the swap if it improves the cost of the tour.
    # Return True/False depending on success.
    def trySwap(self, i):
        gained = self.idx_distance(self.prev(i), i) + \
                 self.idx_distance(self.nxt(i), self.nxt(self.nxt(i)))
        lost = self.idx_distance(i, self.nxt(self.nxt(i))) + \
               self.idx_distance(self.prev(i), self.nxt(i))

        if gained > lost:  # if cost improved
            self.doSwap(i)
            return True
        else:
            return False

    def doSwap(self, i):
        self.perm[i], self.perm[self.nxt(i)] = self.perm[self.nxt(i)], \
                                               self.perm[i]

    # Consider the effect of reversiing the segment between
    # self.perm[i] and self.perm[j], and commit to the reversal
    # if it improves the tour value.
    # Return True/False depending on success.
    def tryReverse(self, i, j):
        gained = self.idx_distance(self.prev(i), i) + self.idx_distance(j,self.nxt(j))
        lost = self.idx_distance(i, self.nxt(j)) + self.idx_distance(
            self.prev(i), j)

        if gained > lost:  # if cost improved
            self.doReverse(i, j)
            return True
        else:
            return False

    def doReverse(self, i, j):
        self.perm[i:(j + 1)] = reversed(self.perm[i:(j + 1)])

    def swapHeuristic(self):
        better = True
        while better:
            better = False
            for i in range(self.n):
                if self.trySwap(i):
                    better = True

    def TwoOptHeuristic(self):
        better = True
        while better:
            better = False
            for j in range(self.n - 1):
                for i in range(j):
                    if self.tryReverse(i, j):
                        better = True

    # Implement the Greedy heuristic which builds a tour starting
    # from node 0, taking the closest (unused) node as 'next'
    # each time.
    def Greedy(self):
        self.perm = [0]
        unused = list(range(1, self.n))
        while len(self.perm) < self.n:
            last = self.perm[-1]
            _, best = min([(self.dists[last][cand], cand) for cand in unused])
            self.perm.append(best)
            unused = [v for v in unused if v != best]

    def myHeuristic(self):
        random.seed()
        m = 1
        for _ in range(self.n*30):
            print("m is currently equal to " + str(m))

            if m == MAX_M_OPT + 1: # if transformations of all degrees failed, finish early
                return;
                random.shuffle(self.perm)
                m = 0

            print("current tour value:" + str(self.tourValue()))
            best_improv, self.perm = self.myHeuristicTryBetter(m)
            if best_improv < self.tourValue()/1000.0:
                m = m + 1
            else:
                m = 1


    def myHeuristicTryBetter(self, m):
        very_good_improv = self.tourValue() / 100.0

        def get_new_range(at_least, less_than, left_to_choose):
            res = []
            if left_to_choose <= 0:
                res = [-1]
            else:
                res = list(range(at_least, less_than))
            random.shuffle(res)
            return res

        best = 0.0, self.perm  # the first value is the tourValue improvement

        for i in get_new_range(0, self.n, m):
            for j in get_new_range(i+1, self.n, m-1):
                for k in get_new_range(j+1, self.n, m-2):
                    for l in get_new_range(k+1, self.n, m-3):
                        if j == -1:  # doing a swap
                            edg_rem = self.idx_distance(self.prev(i), i) + \
                                      self.idx_distance(self.nxt(i), self.nxt(self.nxt(i)))
                            edg_add = self.idx_distance(i, self.nxt(self.nxt(i))) + \
                                      self.idx_distance(self.prev(i), self.nxt(i))
                            cost_improv = edg_rem - edg_add
                            realised_by = [x for x in self.perm]
                            realised_by[i], realised_by[self.nxt(i)] = realised_by[self.nxt(i)], realised_by[i]

                            best = max(best, (cost_improv, realised_by))
                            if cost_improv > very_good_improv:
                                return best
                        else:
                            indexes = [x for x in [i, j, k, l] if x != -1]

                            inters = []
                            for f in range(0, len(indexes) - 1):
                                inters.append(self.perm[(indexes[f]):(indexes[f + 1])])

                            ending = self.perm[indexes[-1]:]
                            beginning = self.perm[0:indexes[0]]
                            if len(ending+beginning) > 0:
                                inters.append(ending + beginning)

                            for mask in range(0, 2 ** (len(inters))):
                                reversed_inters = []
                                for f in range(len(inters)):
                                     reversed_inter = list(reversed(inters[f])) if (mask & (2 ** f)) > 0 else inters[f]
                                     reversed_inters.append(reversed_inter)

                                # consider only shifting them ?
                                #for shift_amnt in range(len(reversed_inters)):
                                for inters_perm_without_last in itertools.permutations(reversed_inters[:-1]):
                                    inters_perm = inters_perm_without_last + (reversed_inters[-1],)
                                    #inters_perm = reversed_inters[shift_amnt:] + reversed_inters[:shift_amnt]
                                    tour = list(itertools.chain(*inters_perm))
                                    backup = self.perm
                                    valnow = self.tourValue()
                                    self.perm = tour
                                    diff = valnow - self.tourValue()
                                    best = max(best, (diff, self.perm))
                                    if diff > very_good_improv:
                                        return best
                                    self.perm = backup

        return best

# #

# g = Graph(-1,"cool_tests/many_circles_n_5rs_1000_2000_4000_8000")
# g.TwoOptHeuristic()
# print(g.tourValue())
#
# g = Graph(6,"sixnodes")
# g.myHeuristic()
# print(g.tourValue())
