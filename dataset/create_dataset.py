import argparse
import os
import sys
sys.path.append("../")
import random

import numpy as np
import networkx as nx
from src.utils import mkdir


def get_parser():
    parser = argparse.ArgumentParser(description='dataset')

    parser.add_argument('--trans_type', type=str, default='MIS', help='TRA, PER, MIS')
    parser.add_argument('--dimension', type=str, default="20,20", help='')

    parser.add_argument('--ds_size', type=int, default=20000, help='')
    parser.add_argument('--ds_path', type=str,
                        default='/public/home/pengwei2022/workspace/_matrix_deep/_diagonal_position_encoding/dataset',
                        help='')

    parser.add_argument('--per_law', type=str, default='', help='')
    parser.add_argument('--per_dimension', type=str, default="5,5", help='')

    return parser


def generate_adj_and_mis(n, density=0.4):
    while True:
        rand_upper = (np.random.rand(n, n) < density).astype(int)
        adj_matrix = np.triu(rand_upper) + np.triu(rand_upper, 1).T
        adj_matrix[np.diag_indices(n)] = 0

        connect_sta = np.sum(adj_matrix, axis=0)

        if np.any(connect_sta == 0):
            continue
        else:
            break

    '''
    Obtain the complement of the original graph
    find the largest group of the complement to 
    obtain the largest independent set of the original graph
    '''
    comp_matrix = np.where((adj_matrix == 0) | (adj_matrix == 1), adj_matrix ^ 1, adj_matrix)
    comp_matrix[np.diag_indices(n)] = 0
    comp_graph = nx.from_numpy_matrix(comp_matrix)

    '''
    Unannotated figure
    '''
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    #
    # pos1 = nx.circular_layout(nx.from_numpy_matrix(adj_matrix))
    # nx.draw(nx.from_numpy_matrix(adj_matrix), pos=pos1, ax=ax[0], with_labels=True, node_color='lightblue',
    #         edge_color='grey', node_size=800)
    # ax[0].set_title('ori_graph')
    #
    # pos2 = nx.circular_layout(comp_graph)
    # nx.draw(comp_graph, pos=pos2, ax=ax[1], with_labels=True, node_color='lightgreen', edge_color='grey', node_size=800)
    # ax[1].set_title('comp_graph')
    #
    # plt.show()

    max_cliques = list(nx.find_cliques(comp_graph))
    assert len(max_cliques) >= 1

    max_num_cliques = [clique for clique in max_cliques if len(clique) == len(max(max_cliques, key=len))]

    sortidx_cliques = [sorted(ns_clique) for ns_clique in max_num_cliques]

    for idx_clique in range(len(sortidx_cliques[0]) + 1):
        if (len(sortidx_cliques) == 1):

            binary_list = [1] * n
            for index in sortidx_cliques[0]:
                binary_list[index] = 0
            mis_list = [bit for bit in binary_list if bit in [0, 1]]

            return adj_matrix, [mis_list]

        emu_idx_clique = [s_clique[idx_clique] for s_clique in sortidx_cliques]
        minidx_clique = min(emu_idx_clique)
        idx_del = [i for i, x in enumerate(emu_idx_clique) if x != minidx_clique]

        sortidx_cliques = [sortidx_cliques[i] for i in range(len(sortidx_cliques)) if i not in idx_del]


if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()

    trans_type = params.trans_type
    input_dimension = (int(params.dimension.split(",")[0]), int(params.dimension.split(",")[1]))
    ds_size = params.ds_size

    if not (trans_type == 'TRA' or trans_type == 'PER' or trans_type == 'MIS'):
        raise ValueError('Invalid value for trans_type: {}'.format(trans_type))

    exp_id = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(5))

    ds_path = os.path.join(params.ds_path, trans_type)
    mkdir(ds_path)

    print(f"The dataset {exp_id} will be created at {ds_path}")

    file_path = os.path.join(ds_path, '_'.join([str(ds_size),
                                                str(input_dimension[0]),
                                                str(input_dimension[1])
                                                ]))

    if trans_type == 'PER':

        pmu_index = None

        if len(params.per_law) == 0:
            pmu_index = list(range(input_dimension[0] * input_dimension[1]))
            random.shuffle(pmu_index)

            with open(os.path.join(ds_path, exp_id), 'w') as f:
                f.writelines(str(pmu_index))
        else:
            per_law = open(params.per_law, 'r')
            pmu_index = eval(per_law.read())
            per_law.close()

            exp_id = params.per_law.split('/')[-1]

    file_path = os.path.join('_'.join([file_path, exp_id]))

    assert not os.path.exists(file_path)

    with open(file_path, 'w') as f:

        for i in range(ds_size):

            if trans_type == 'TRA':
                input_ori = np.random.randint(0, 10, input_dimension)
                output_ori = input_ori.T
            elif trans_type == 'PER':
                input_ori = np.random.randint(0, 10, input_dimension)

                assert pmu_index is not None
                assert input_dimension is not None

                input_per = input_ori[:int(params.per_dimension.split(",")[0]), :int(params.per_dimension.split(",")[1])]

                output_ori = np.ones(input_per.shape[0] * input_per.shape[1], dtype=int) * -1

                for input_i, input_n in enumerate(input_per.flatten()):
                    output_ori[pmu_index[input_i]] = input_n

                output_ori = output_ori.reshape(int(params.per_dimension.split(",")[0]), int(params.per_dimension.split(",")[1]))

            elif trans_type == 'MIS':
                input_ori, output_ori = generate_adj_and_mis(input_dimension[0])

            input_str = ''.join(str(i) for r in input_ori for i in r)
            output_str = ''.join(str(i) for r in output_ori for i in r)

            ds_in = ','.join([input_str, output_str])

            if i == (ds_size - 1):
                f.write(ds_in)
            else:
                f.write(''.join([ds_in, '\n']))

        f.close()
