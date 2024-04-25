import random
import matplotlib.markers as mat_mar
import matplotlib.pyplot as mat_pyp
import numpy as numpy_np
import sklearn.preprocessing as mat_pre
from sklearn.neighbors import KNeighborsClassifier

class Learn:
    def __init__(self):
        self.tr_datas = None
        self.datas = None
        self.t_datas = None
        self.props = ('Długość działki kielicha', 'Szerokość działki kielicha',
                      'Długość płatka', 'Szerokość płatka')
        self.types = ['setosa', 'versicolor', 'virginica']
        self.colors = ("green", "black", "blue")
        self.min = (4.1, 2.0, 0.5, 0.0)
        self.max = (8.1, 4.6, 7.0, 2.5)
        self.m_result = dict()
        self.n_results = dict()

    def load(self, f, f_tr, f_t):
        self.datas = numpy_np.loadtxt(f, delimiter=',')
        self.tr_datas = numpy_np.loadtxt(f_tr, delimiter=',')
        self.t_datas = numpy_np.loadtxt(f_t, delimiter=',')

    def prop_num(self):
        return self.datas.shape[1] - 1

    def typeof(self, id):
        return self.datas[id, -1]

    def num(self, type):
        numb = 0
        for i in range(len(self.datas)):
            if self.typeof(i) == type:
                numb += 1
        return numb

    def filters(self, type):
        result = []
        for i in range(len(self.datas)):
            if self.typeof(i) == type:
                result.append(self.datas[i, :])
        return numpy_np.array(result)

    def project(self, datas, one_prop, two_prop):
        result = []
        for i in range(len(datas)):
            result.append([datas[i, :][id] for id in
                           [one_prop, two_prop]])
        return numpy_np.array(result)

    def set_filters(self, vect, assig, id):
        result = []
        for i in range(len(vect)):
            if assig[i] == id:
                result.append(vect[i])
        return numpy_np.array(result)

    def k_means_ML(self, vect, clst):
        indices = random.sample(range(len(vect)), clst)
        mid_array = []
        for i in indices:
            mid_array.append(vect[i].copy())
        iterations = 0
        inertia = 0
        assig = numpy_np.zeros((len(vect)))
        centers = numpy_np.array(mid_array)
        assig.fill(-1)
        while True:
            old_assig = assig.copy()
            for i in range(len(vect)):
                dist = []
                for j in range(clst):
                    dist.append(numpy_np.linalg.norm(vect[i] - centers[
                        j]) ** 2)
                assig[i] = numpy_np.argmin(dist)
                inertia += dist[int(assig[i])]

            if numpy_np.array_equal(assig, old_assig):
                break

            element = []
            for i in range(clst):
                element.append([])
            for i in range(len(vect)):
                element[int(assig[i])].append(vect[i])
            element = numpy_np.array(element, dtype=numpy_np.ndarray)
            for i in range(clst):
                centers[i] = numpy_np.mean(element[i], axis=0)
            inertia = 0
            iterations = iterations + 1
        return centers, assig, iterations, inertia

    def cluster(self, clst_from, clst_to):
        clst = range(clst_from, clst_to + 1)
        for k in clst:
            self.m_result[k] = self.k_means_ML(self.datas[:, :-1], k)

    def draw_clst(self, one_prop, two_prop, clst):
        vect = self.project(self.datas, one_prop, two_prop)
        centers, assig, iterations, inertia = self.m_result[clst]

        left_limit = self.min[one_prop] - 0.3
        right_limit = self.max[one_prop] + 0.1
        mat_pyp.xlim(left_limit, right_limit)

        print(f"k = {clst}")
        for i in range(clst):
            v = self.set_filters(vect, assig, i)
            mat_pyp.scatter(v[:, 0], v[:, 1], marker=mat_mar.MarkerStyle("o", fillstyle="full"), color=self.colors[i])

        for i in range(clst):
            mat_pyp.scatter(centers[i][one_prop], centers[i][two_prop],
                            marker=mat_mar.MarkerStyle("D", fillstyle="full"),
                            color="red")
        mat_pyp.xlabel(self.props[one_prop])
        mat_pyp.ylabel(self.props[two_prop])
        mat_pyp.tick_params(axis='x', labelsize=15)
        mat_pyp.tick_params(axis='y', labelsize=15)

        mat_pyp.show()

    def draw_diff_clst(self, clst_from, clst_to):
        iterations = []
        inertions = []
        clst = range(clst_from, clst_to + 1)
        for k in clst:
            iterations.append(self.m_result[k][2])
            inertions.append(self.m_result[k][3])

        mat_pyp.figure(figsize=(10, 6))
        mat_pyp.plot(clst, iterations, color='blue', marker='o')
        mat_pyp.xlabel("k")
        mat_pyp.ylabel("Iterations")
        mat_pyp.title("Iterations per k")
        mat_pyp.xticks(clst)
        mat_pyp.grid(True)
        mat_pyp.tick_params(axis='x', labelsize=19)
        mat_pyp.tick_params(axis='y', labelsize=19)
        mat_pyp.show()

        mat_pyp.figure(figsize=(10, 6))
        mat_pyp.plot(clst, inertions, color='orange', marker='o')
        mat_pyp.xlabel("k")
        mat_pyp.ylabel("WCSS")
        mat_pyp.title("WCSS per k")
        mat_pyp.xticks(clst)
        mat_pyp.ylim(0, 160)
        mat_pyp.grid(True)
        mat_pyp.tick_params(axis='x', labelsize=19)
        mat_pyp.tick_params(axis='y', labelsize=19)
        mat_pyp.show()

    def k_neighbrs(self, tr_datas, types, t_datas, neighbrs):
        all_data = numpy_np.concatenate(
            (tr_datas, t_datas))
        for i in range(tr_datas.shape[1]):
            all_data[:, i] = mat_pre.minmax_scale(all_data[:, i],
                                                  feature_range=(
                                                      0, 100))
        sep_data = numpy_np.vsplit(all_data, [tr_datas.shape[0]])
        tr_datas = sep_data[0]
        t_datas = sep_data[1]
        kneighbrs = KNeighborsClassifier(neighbrs,
                                           weights="distance")
        return kneighbrs.fit(tr_datas, types).predict(t_datas)

    def classifying(self, neighbours_from, neighbours_to):
        neighbours = range(neighbours_from, neighbours_to + 1)
        for k in neighbours:
            self.n_results[k] = self.k_neighbrs(self.tr_datas[:, :-1], self.tr_datas[:, -1], self.t_datas[:, :-1], k)

    def compare(self, real, predicted):
        correct = 0
        conf_grid = numpy_np.zeros((len(self.types), len(self.types)))
        for i in range(len(real)):
            if real[i] == predicted[i]:
                correct = correct + 1
            conf_grid[int(real[i])][int(predicted[i])] += + 1
        return round(correct / len(real) * 100), conf_grid

    def save_diag(self, f, table):
        out = open(f, 'w')
        out.write('\"\",\"' + self.types[0] + '\",\"' + self.types[1] + '\",\"' + self.types[
            2] + '\"\n')
        for i in range(len(table)):
            out.write(
                '\"' + self.types[i] + '\",' + str(int(table[i][0])) + ',' + str(int(table[i][1])) + ',' + str(
                    int(table[i][2])) + '\n')
        out.close()

    def draw_diff_neighbrs(self, neighbrs_from, neighbrs_to):
        neighbrs = range(neighbrs_from, neighbrs_to + 1)
        perc = []
        grid = []
        for k in neighbrs:
            result = self.compare(self.t_datas[:, -1], self.n_results[k])
            perc.append(result[0])
            grid.append(result[1])

        best_k_id = numpy_np.argmax(perc)
        print(f"Best nnk: {best_k_id}")
        best_grid = grid[best_k_id]

        mat_pyp.figure(figsize=(10, 6))
        bars = mat_pyp.bar(neighbrs, perc, color='skyblue')
        mat_pyp.xlabel("k")
        mat_pyp.ylabel("Wynik klasyfikacji [%]")
        mat_pyp.title("Summary classification result")
        mat_pyp.xticks(numpy_np.arange(neighbrs_from, neighbrs_to + 1, 2))
        mat_pyp.ylim(60, 102)
        mat_pyp.yticks(numpy_np.arange(60, 101, 5))
        mat_pyp.grid(True)
        mat_pyp.tight_layout()

        mat_pyp.tick_params(axis='x', labelsize=18)
        mat_pyp.tick_params(axis='y', labelsize=18)

        mat_pyp.show()
        return best_grid

    def draw_neighbrs(self, one_prop, two_prop, neighbrs_from, neighbrs_to):
        perc = []
        grid = []
        train_vect = self.project(self.tr_datas, one_prop, two_prop)
        test_vect = self.project(self.t_datas, one_prop, two_prop)
        neighbrs = range(neighbrs_from, neighbrs_to + 1)

        for k in neighbrs:
            result = self.compare(self.t_datas[:, -1], self.k_neighbrs(train_vect, self.tr_datas[:, -1], test_vect, k))
            perc.append(result[0])
            grid.append(result[1])

        best_k_id = numpy_np.argmax(perc)
        print(f"Best nnk: {best_k_id}")
        best_grid = grid[best_k_id]

        mat_pyp.figure(figsize=(10, 6))
        mat_pyp.bar(neighbrs, perc, color='skyblue')
        mat_pyp.title(f"Summary classification result\n{self.props[one_prop]} & {self.props[two_prop]}")
        mat_pyp.xlabel("k")
        mat_pyp.ylabel("Wynik klasyfikacji [%]")
        mat_pyp.xticks(numpy_np.arange(neighbrs_from, neighbrs_to + 1, 2))
        mat_pyp.yticks(numpy_np.arange(60, 101, 5))

        if (one_prop, two_prop) == (0, 1) or (one_prop, two_prop) == (1, 0):
            mat_pyp.ylim(60, 85)
        else:
            mat_pyp.ylim(60, 102)

        mat_pyp.grid(True)
        mat_pyp.tight_layout()

        mat_pyp.tick_params(axis='x', labelsize=18)
        mat_pyp.tick_params(axis='y', labelsize=18)

        mat_pyp.show()
        return best_grid








