from Learning2 import Learn

if __name__ == '__main__':
    p = Learn()
    p.load('data.csv', 'data_train.csv', 'data_test.csv')

    p.cluster(2, 10)

    for i in range(p.prop_num() - 1):
        for j in range(i + 1, p.prop_num()):
            p.draw_clst(i, j, 3)

    p.draw_diff_clst(2, 10)

    p.classifying(1, 15)

    best_grid = p.draw_diff_neighbrs(1, 15)
    p.save_diag('grid1.csv', best_grid)

    number = 2
    for i in range(p.prop_num() - 1):
        for j in range(i + 1, p.prop_num()):
            best_grid = p.draw_neighbrs(i, j, 1, 15)
            p.save_diag(f'grid{number}.csv', best_grid)
            number += 1

