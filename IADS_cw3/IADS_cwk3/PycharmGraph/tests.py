import random
import math
import matplotlib.pyplot as plt
from graph import Graph
import itertools


class TSPEuclideanTestCase:
    def __init__(self):
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
        self.coordinates = []


class RandomEuclideanTestCase(TSPEuclideanTestCase):
    def __init__(self, n, minx, maxx, miny, maxy):
        self.coordinates = [(random.randint(minx,maxx), random.randint(miny, maxy)) for _ in range(n)]
        self.test_name = "rand_euclid_n_" + str(n) + "x_" + str(minx) + "_" + str(maxx) + "_y_" + str(miny) + "_" + str(maxy)


class PointsOnACircleTestCase(TSPEuclideanTestCase):
    def __init__(self, n, r):
        self.coordinates = points_on_circle(n, r)
        random.shuffle(self.coordinates)
        self.test_name = "circle_n_" + str(n) + "r_" + str(r)

class ManyCirclesTestCase(TSPEuclideanTestCase):
    def __init__(self, n, rs, add_random_points):
        self.coordinates = list(itertools.chain(*[points_on_circle(n, r) for r in rs]))
        if add_random_points:
            m = max(rs)/10
            self.coordinates += [(random.randint(-m,m), random.randint(-m,m)) for _ in range(n)]
        self.coordinates.sort(key= lambda coords : coords[0]*coords[0] + coords[1]* coords[1])
        self.test_name = "many_circles_n_" + str(n) + "rs_" + "_".join([str(r) for r in rs])

class TwoLinesTestcase(TSPEuclideanTestCase):
    def __init__(self, n):
        self.coordinates = list(itertools.chain(*[[(4*i,0),(4*i,3)] for i in range(n)]))
        self.test_name = "two_lines_n_" + str(n)

def empty_dists(self):
    return [[0.0 for __ in range(self.n)] for _ in range(self.n)]

def points_on_circle(n, r):
    points_floats = [(r * math.cos(2.0 * math.pi / n * i), r * math.sin(2.0 * math.pi / n * i)) for i in range(n)]
    return [(round(x), round(y)) for x, y in points_floats]

def run_test(test):
    filename = test.save_to_file()
    heuristics = [Graph.swapHeuristic, Graph.TwoOptHeuristic, Graph.Greedy, Graph.myHeuristic]
    results = []

    for heuristic in heuristics:
        g = Graph(-1, filename)
        heuristic(g)
        results.append(round(g.tourValue()))
        print(results[-1])
        plot_solution(test.coordinates, g.perm)


    print("test:" + str(filename) + " results:" + ", ".join([str(r) for r in results]))

# a function to show a plot of coordinates, helpful to visualise the solution
def plot_solution(coords, perm):
    coords_in_order = [coords[perm[i]] for i in range(len(perm))]
    plt.plot([x[0] for x in coords_in_order], [x[1] for x in coords_in_order])
    plt.show()
    # ax = plt.axes()
    # for c, d in zip(coords_in_order, [coords_in_order[-1]]+coords_in_order):
    #     ax.arrow(c[0], c[1], d[0]-c[0], d[1]-c[1], head_width=0.05, head_length=0.1, fc='k', ec='k')
    # plt.show()

random.seed()

euclidean_test_cases = [
                        #RandomEuclideanTestCase(30,-10, 10,-10, 10),
                        #RandomEuclideanTestCase(30,-10000, 10000, -10000, 10000),
                        #RandomEuclideanTestCase(30,-10, 10, -10000, 10000),
                        #PointsOnACircleTestCase(30, 1000),
                        #ManyCirclesTestCase(5,[1000,2000,4000,8000], True),
                        ManyCirclesTestCase(20, [1000, 2000, 4000, 8000], False)
                        #TwoLinesTestcase(20)
                        ]



for test in euclidean_test_cases:
    run_test(test)

