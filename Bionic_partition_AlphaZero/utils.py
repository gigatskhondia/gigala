from fea import PlaneTrussElementLength, FEA_u
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# print('matplotlib: {}'.format(matplotlib.__version__))
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import math


def total_length(coord,elcon):
    coord=np.array(coord)
    elcon=np.array(elcon)
    t_length=0
    for i in range(len(elcon)):
        l=PlaneTrussElementLength(coord[elcon[i][0]][0],\
                                    coord[elcon[i][0]][1],\
                                    coord[elcon[i][0]][2],\
                                    coord[elcon[i][1]][0],\
                                    coord[elcon[i][1]][1],\
                                    coord[elcon[i][1]][2])
        t_length+=l
    return t_length


def max_u(FEA_output_arr):
    t = 1
    A = []
    while t < len(FEA_output_arr):
        A.append(FEA_output_arr[t])
        t += 6
    return min(A)


def draw(color,coord,elcon):
    c=coord
    e=elcon
    c=np.array(c)
    e=np.array(e)
    coord=c.reshape(np.max(e)+1,3)
    # coord=c
    fig=plt.figure(figsize=(13,5))
    for item in e:
        ax = fig.gca(projection='3d')
        ax.plot([coord[item[0]][0],coord[item[1]][0]],\
                 [coord[item[0]][1],coord[item[1]][1]],\
                 [coord[item[0]][2],coord[item[1]][2]],
                 color=color)
#             ax.view_init(70,300)
    ax.view_init(-90,90)
#         ax1 = plt.subplot(131)
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])
    plt.show()


def real_dim(width, height):

    tmp = (width - 1) * (height - 1) * 4 + width + height - 2
    width = int(math.sqrt(tmp))
    height = int(tmp/width)

    return width, height


# lines dic
def match_lines_to_coordinates(grid_dim, dx=1, dy=1):
    # grid_dim - list with 2D grids dimensions e.g. [5,5]
    # print(grid_dim)
    dic = {}
    k = 0
    for i in range(grid_dim[1]-1):
        for j in range(grid_dim[0]-1):
            x = dx * j
            y = dy * i
            dic[k]=(x, y, x+dx, y+0)
            k+=1
            dic[k]=(x, y, x+dx, y+dx)
            k+=1
            dic[k] = (x, y+dy, x + dx, y + 0)
            k+=1
            dic[k] = (x, y, x, y+dy)
            k+=1

    for i in range(grid_dim[0]-1):
        x=i*dx
        y=(grid_dim[1]-1)*dy
        dic[k]=(x,y,x+dx,y)
        k+=1

    for i in range(grid_dim[1]-1):
        x=(grid_dim[0]-1)*dy
        y= i*dy
        dic[k]=(x,y,x,y+dy)
        k+=1

    return dic


def checkpoints_visit(moved, checkpoints, lines_dic):
    # checkpoints e.g. [(0,0),(4,0),(0,4),(4,4)]
    t = 0
    for item in checkpoints:
        for it in moved:
            if (lines_dic[it][0],lines_dic[it][1])==item or (lines_dic[it][2],lines_dic[it][3])==item:
                t+=1
                break

    if t == len(checkpoints):
        return True
    return False


def graph_connected(elcon):
    G=nx.Graph()
    for item in elcon:
        G.add_edge(item[0], item[1])

    return nx.is_connected(G)


def has_at_least_two_neighbors(moved, lines_dic):
    dic={}
    for item in moved:
        if (lines_dic[item][0],lines_dic[item][1]) not in dic:
            dic[(lines_dic[item][0], lines_dic[item][1])] = [(lines_dic[item][2], lines_dic[item][3])]
        else:
            dic[(lines_dic[item][0], lines_dic[item][1])].append((lines_dic[item][2], lines_dic[item][3]))

        if (lines_dic[item][2], lines_dic[item][3]) not in dic:
            dic[(lines_dic[item][2], lines_dic[item][3])] = [(lines_dic[item][0], lines_dic[item][1])]
        else:
            dic[(lines_dic[item][2], lines_dic[item][3])].append((lines_dic[item][0], lines_dic[item][1]))

    if all([len(value) >= 2 for key, value in dic.items()]):
        return True
    return False


# weight and strength either improve or at least stay the same
def weight_and_strength_did_not_deteriorate(moved, old_weight, old_strength, checkpoints, lines_dic, force=-500):
    dic={}
    el=0
    coord=[]
    for item in moved:
        if (lines_dic[item][0],lines_dic[item][1]) not in dic:
            dic[(lines_dic[item][0],lines_dic[item][1])]=el
            el+=1
            coord.append([lines_dic[item][0],lines_dic[item][1],0])
        if(lines_dic[item][2], lines_dic[item][3]) not in dic:
            dic[(lines_dic[item][2], lines_dic[item][3])] = el
            el += 1
            coord.append([lines_dic[item][2], lines_dic[item][3], 0])

    elcon=[]
    for item in moved:
        elcon.append([dic[(lines_dic[item][0], lines_dic[item][1])],dic[(lines_dic[item][2], lines_dic[item][3])]])

    f_after_u_elim=[]
    bc_u_elim=[]
    for item in coord:
        if (item[0],item[1]) in checkpoints:
            bc_u_elim += list(range((dic[(item[0],item[1])]+1) * 6 - 6, (dic[(item[0], item[1])]+1) * 6))
        else:
            f_after_u_elim+=[0,force,0,0,0,0]
    # print(f_after_u_elim)
    # print(bc_u_elim)
    u=FEA_u(coord, elcon, bc_u_elim, f_after_u_elim)

    new_strength=max_u(u)
    new_weight=total_length(coord,elcon)

    if new_weight<=old_weight and new_strength>=old_strength:
        # update old weight and strength
        return True,new_weight, new_strength, coord, elcon
    return False, old_weight,old_strength, coord, elcon


if __name__ == '__main__':
    print(match_lines_to_coordinates([5, 5], 1, 1))
    print(has_at_least_two_neighbors([0,1,2], {0: (0, 0, 1, 0), 1: (0, 0, 1, 1), 2: (0, 1, 1, 0)}))
    print(checkpoints_visit([0,1,2],[(0,4)], {0: (0, 0, 1, 0), 1: (0, 0, 1, 1), 2: (0, 1, 1, 0)}))
    # print(weight_and_strength_did_not_deteriorate([0,1,2], 0, 0, 0, {0: (0, 0, 1, 0), 1: (0, 0, 1, 1), 2: (0, 1, 1, 0)}))
    # print(draw("green",[[0,0,0],[1,0,0]],[[0,1]]))
    # print(graph_connected([0,1,2], {0: (0, 0, 1, 0), 1: (0, 0, 1, 1), 2: (0, 1, 1, 0)}))