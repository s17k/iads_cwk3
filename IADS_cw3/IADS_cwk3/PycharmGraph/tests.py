import random
import math
import matplotlib.pyplot as plt
from graph import Graph
import itertools


class TSPEuclideanTestCase:
    def __init__(self):
        self.n = -1
        self.coordinates = []
        self.test_name = "undefined"

    def save_to_file(self):
        filename = 'tests/' + self.test_name
        with open(filename, 'w') as f:
            for x, y in self.coordinates:
                print(str(x) + ' ' + str(y), file=f)
        return filename


class ExistingTestCase(TSPEuclideanTestCase):
    def __init__(self):
        super().__init__()
        self.coordinates = []


class RandomEuclideanTestCase(TSPEuclideanTestCase):
    def __init__(self, n, minx, maxx, miny, maxy):
        super().__init__()
        self.coordinates = [(random.randint(minx,maxx), random.randint(miny, maxy)) for _ in range(n)]
        self.test_name = "rand_euclid_n_" + str(n) + "x_" + str(minx) + "_" + str(maxx) + "_y_" + str(miny) + "_" + str(maxy)


class PointsOnACircleTestCase(TSPEuclideanTestCase):
    def __init__(self, n, r):
        super().__init__()
        self.coordinates = points_on_circle(n, r)
        random.shuffle(self.coordinates)
        self.test_name = "circle_n_" + str(n) + "r_" + str(r)


class ManyCirclesTestCase(TSPEuclideanTestCase):
    def __init__(self, n, rs, add_random_points):
        super().__init__()
        self.coordinates = list(itertools.chain(*[points_on_circle(n, r) for r in rs]))
        if add_random_points:
            m = max(rs)/10
            self.coordinates += [(random.randint(-m,m), random.randint(-m,m)) for _ in range(n)]
        self.coordinates.sort(key= lambda coords : coords[0]*coords[0] + coords[1]* coords[1])
        self.test_name = "many_circles_n_" + str(n) + "rs_" + "_".join([str(r) for r in rs])


class TwoLinesTestcase(TSPEuclideanTestCase):
    def __init__(self, n):
        super().__init__()
        self.coordinates = list(itertools.chain(*[[(4*i,0),(4*i,3)] for i in range(n)]))
        self.test_name = "two_lines_n_" + str(n)


class GeneralTSPTestCase:
    def __init__(self, n, test_name):
        self.graph = empty_dists(n)
        self.n = n
        self.test_name = test_name

    def save_to_file(self):
        filename = 'tests/' + self.test_name
        with open(filename, 'w') as f:
            for i in range(self.n):
                for j in range(i+1, self.n):
                    print(str(i) + ' ' + str(j) + ' ' + str(self.graph[i][j]), file=f)
        return filename


class RandomNonEuclideanTestCase(GeneralTSPTestCase):
    def __init__(self, n):
        super().__init__(n, "gen_rand_n_" + str(n))
        for i in range(n):
            for j in range(i+1, n):
                self.graph[i][j] = self.graph[j][i] = random.randint(1,100)

def empty_dists(n):
    return [[0.0 for __ in range(n)] for _ in range(n)]

def points_on_circle(n, r):
    points_floats = [(r * math.cos(2.0 * math.pi / n * i), r * math.sin(2.0 * math.pi / n * i)) for i in range(n)]
    return [(round(x), round(y)) for x, y in points_floats]

def run_test(test, plotting):
    filename = test.save_to_file()
    heuristics = [Graph.swapHeuristic, Graph.TwoOptHeuristic, Graph.Greedy, Graph.myHeuristic]
    results = []

    for heuristic in heuristics:
        g = Graph(test.n, filename)
        heuristic(g)
        results.append(round(g.tourValue()))
        print(heuristic.__name__, results[-1])
        if plotting:
            plot_solution(test.coordinates, g.perm)

    print("test:" + str(filename) + " results:" + ", ".join([str(r) for r in results]))


# a function to show a plot of coordinates, helpful to visualise the solution
def plot_solution(coords, perm):
    coords_in_order = [coords[perm[i]] for i in range(len(perm))]
    plt.plot([x[0] for x in coords_in_order], [x[1] for x in coords_in_order])
    plt.show()


random.seed()

euclidean_test_cases = [
                            #RandomEuclideanTestCase(30,-10, 10,-10, 10),
                            #RandomEuclideanTestCase(30,-10000, 10000, -10000, 10000),
                            #RandomEuclideanTestCase(30,-10, 10, -(10**5), 10**5),
                            #PointsOnACircleTestCase(30, 1000),
                            #ManyCirclesTestCase(5,[1000,2000,4000,8000], True),
                            #ManyCirclesTestCase(20, [1000, 2000, 4000, 8000], False)
                            TwoLinesTestcase(20)
                        ]

general_test_cases = [
                            RandomNonEuclideanTestCase(50)
                      ]

for test in euclidean_test_cases:
    run_test(test, plotting = False)

for test in general_test_cases:
    run_test(test, plotting = False)
