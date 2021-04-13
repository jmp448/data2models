#!/usr/bin/env python

# imports that are potentially quite useful
import numpy as np
import pandas as pd
from scipy.linalg import expm
from itertools import combinations
import statsmodels.api as sm
from statsmodels.formula.api import ols
from random import randint
import random
import time

# comment these lines if you want to visualize
# your final DAG using something other than Ananke
from ananke.graphs import DAG


def directed_cycle_score(A):
    """
    Compute a score for the number of directed cycles in a graph
    given its binary adjacency matrix A. Returns a float that is
    non-negative and is zero if and only if the graph is a DAG.
    """

    # Implement your cycle score given Problem 4 Part 2
    cycle_score = np.trace(expm(A))-len(A)

    return cycle_score

def bic_score(A, data, idx_to_var_map):
    """
    Compute the BIC score for a DAG given by its adjacency matrix A
    and data as a pandas data frame. idx_to_var_map is a dictionary
    mapping row/column indices in the adjacency matrix to variable names
    in the data frame
    """

    bic = 0
    num_vars = len(A)

    for i in range(num_vars):

        v_i = idx_to_var_map[i]
        formula_vi = str(v_i) + " ~ 1"
        for j in range(num_vars):
            if A[j, i] == 1:
                formula_vi += (" + " + idx_to_var_map[j])

        model = ols(formula=formula_vi, data=data).fit()
        bic += model.bic

    return bic

def bic_score_node(A, data, idx_to_var_map, i):
    """
    Compute the BIC score for a DAG given by its adjacency matrix A
    and data as a pandas data frame. idx_to_var_map is a dictionary
    mapping row/column indices in the adjacency matrix to variable names
    in the data frame
    """

    num_vars = len(A)

    v_i = idx_to_var_map[i]
    formula_vi = str(v_i) + " ~ 1"
    for j in range(num_vars):
        if A[j, i] == 1:
            formula_vi += (" + " + idx_to_var_map[j])

    model = ols(formula=formula_vi, data=data).fit()
    bic = model.bic

    return bic

def causal_discovery(data, num_steps=100, cycle_score_tolerance=1e-9):
    """
    Take in data and perform causal discovery according to a set of moves
    described in the write up for a given number of steps.
    Since the output of the cycle score function is a float, comparison to
    zero is a little tricky. We use a really small tolerance to say the number
    is close enough to zero. That is, for x < cycle_score_tolerance x is close
    enough to 0.
    """
    idx_to_var_map = {i: var_name for i, var_name in enumerate(data.columns)}
    num_vars = len(data.columns)

    # initialize an empty graph
    A_opt = np.zeros((num_vars, num_vars), dtype=int)
    # besides the adjacency matrix keep a set of edges present
    # in the graph making for easy delete/reverse moves. each entry in the
    # set is a tuple of integers (i, j) corresponding to indices
    # for the end points of a directed edge Vi-> Vj
    edges = set([])

    # get initial BIC score for empty graph and set it to the current optimal
    bic_opt = bic_score(A_opt, data, idx_to_var_map)

    for step in range(num_steps):

        # random edge addition
        if A_opt.__contains__(0):
            A_add = A_opt.copy()
            add_edge = (np.random.randint(0, num_vars-1), np.random.randint(0, num_vars-1))
            while edges.__contains__(add_edge):
                add_edge = (np.random.randint(0, num_vars - 1), np.random.randint(0, num_vars - 1))
            A_add[add_edge[0], add_edge[1]] = 1
            if directed_cycle_score(A_add) < cycle_score_tolerance:
                bic_add = bic_score(A_add, data, idx_to_var_map)
            else:
                bic_add = np.inf
        else:
            bic_add = np.inf

        # random edge deletion
        if len(edges) > 0:
            A_del = A_opt.copy()
            del_edge = edges.copy().pop()
            A_del[del_edge[0], del_edge[1]] = 0
            bic_del = bic_score(A_del, data, idx_to_var_map)
        else:
            bic_del = np.inf

        # random edge reversal
        if len(edges) > 0:
            A_rev = A_opt.copy()
            rev_edge = edges.copy().pop()
            A_rev[rev_edge[0], rev_edge[1]] = 0
            A_rev[rev_edge[1], rev_edge[0]] = 1
            if directed_cycle_score(A_rev) < cycle_score_tolerance:
                bic_rev = bic_score(A_rev, data, idx_to_var_map)
            else:
                bic_rev = np.inf
        else:
            bic_rev = np.inf

        bic_move = min(bic_add, bic_del, bic_rev)
        if bic_move < bic_opt:
            bic_opt = bic_move
            if bic_move == bic_add:
                edges.add(add_edge)
                A_opt = A_add
            elif bic_move == bic_del:
                edges.remove(del_edge)
                A_opt = A_del
            else:
                edges.remove(rev_edge)
                edges.add((rev_edge[1], rev_edge[0]))
                A_opt = A_rev

    return A_opt, edges, idx_to_var_map


def causal_discovery_efficient(data, num_steps=100, cycle_score_tolerance=1e-9):
    """
    Take in data and perform causal discovery according to a set of moves
    described in the write up for a given number of steps.
    Since the output of the cycle score function is a float, comparison to
    zero is a little tricky. We use a really small tolerance to say the number
    is close enough to zero. That is, for x < cycle_score_tolerance x is close
    enough to 0.
    """
    idx_to_var_map = {i: var_name for i, var_name in enumerate(data.columns)}
    num_vars = len(data.columns)

    # initialize an empty graph
    A_opt = np.zeros((num_vars, num_vars), dtype=int)
    # besides the adjacency matrix keep a set of edges present
    # in the graph making for easy delete/reverse moves. each entry in the
    # set is a tuple of integers (i, j) corresponding to indices
    # for the end points of a directed edge Vi-> Vj
    edges = set([])

    # get initial BIC score for empty graph and set it to the current optimal
    bic_opt_nodewise = []
    for j in range(len(A_opt)):
        bic_opt_nodewise.append(bic_score_node(A_opt, data, idx_to_var_map, j))
    bic_opt = sum(bic_opt_nodewise)

    for step in range(num_steps):

        # random edge addition
        bic_add_nodewise = bic_opt_nodewise.copy()
        if A_opt.__contains__(0):
            A_add = A_opt.copy()
            add_edge = (np.random.randint(0, num_vars-1), np.random.randint(0, num_vars-1))
            while edges.__contains__(add_edge):
                add_edge = (np.random.randint(0, num_vars - 1), np.random.randint(0, num_vars - 1))
            A_add[add_edge[0], add_edge[1]] = 1
            if directed_cycle_score(A_add) < cycle_score_tolerance:
                bic_add_nodewise[add_edge[1]] = bic_score_node(A_add, data, idx_to_var_map, add_edge[1])
        bic_add = sum(bic_add_nodewise)

        # random edge deletion
        bic_del_nodewise = bic_opt_nodewise.copy()
        if len(edges) > 0:
            A_del = A_opt.copy()
            del_edge = edges.copy().pop()
            A_del[del_edge[0], del_edge[1]] = 0
            bic_del_nodewise[del_edge[1]] = bic_score_node(A_del, data, idx_to_var_map, del_edge[1])
        bic_del = sum(bic_del_nodewise)

        # random edge reversal
        bic_rev_nodewise = bic_opt_nodewise.copy()
        if len(edges) > 0:
            A_rev = A_opt.copy()
            rev_edge = edges.copy().pop()
            A_rev[rev_edge[0], rev_edge[1]] = 0
            A_rev[rev_edge[1], rev_edge[0]] = 1
            if directed_cycle_score(A_rev) < cycle_score_tolerance:
                bic_rev_nodewise[rev_edge[0]] = bic_score_node(A_rev, data, idx_to_var_map, rev_edge[0])
                bic_rev_nodewise[rev_edge[1]] = bic_score_node(A_rev, data, idx_to_var_map, rev_edge[1])
        bic_rev = sum(bic_rev_nodewise)

        bic_move = min(bic_add, bic_del, bic_rev)
        if bic_move < bic_opt:
            bic_opt = bic_move
            if bic_move == bic_add:
                edges.add(add_edge)
                A_opt = A_add
                bic_opt_nodewise = bic_add_nodewise
            elif bic_move == bic_del:
                edges.remove(del_edge)
                A_opt = A_del
                bic_opt_nodewise = bic_del_nodewise
            else:
                edges.remove(rev_edge)
                edges.add((rev_edge[1], rev_edge[0]))
                A_opt = A_rev
                bic_opt_nodewise = bic_rev_nodewise

    return A_opt, edges, idx_to_var_map

def test_causal_discovery_function():
    ################################################
    # Tests for your causal_discovery function
    ################################################
    np.random.seed(1000)
    random.seed(0)
    data = pd.read_csv("data.txt")
    t0 = time.time()
    A_opt, edges, idx_to_var_map = causal_discovery(data, num_steps=100)
    t1 = time.time()
    print(t1 - t0)

    # comment these lines if visualizing the DAG using something other
    # than Ananke. Make sure to supply alternative visualization
    # code though!
    vertices = idx_to_var_map.values()
    edges = [(idx_to_var_map[i], idx_to_var_map[j]) for i, j in edges]
    G = DAG(vertices, edges)
    # the DAG will be stored in a PDF final_DAG.gv.pdf
    G.draw().render("final_DAG_2.gv", view=False)


def test_directed_cycle_score_function():
    ################################################
    # Tests for your directed_cycle_score function
    ################################################

    # Treating X, Y, Z as indices 0, 1, 2 in the adjacency matrix
    # X->Y<-Z, Z->X
    A1 = np.array([[0, 1, 0],
                   [0, 0, 0],
                   [1, 1, 0]])

    # X->Y->Z, Z->X
    A2 = np.array([[0, 1, 0],
                   [0, 0, 1],
                   [1, 0, 0]])

    print(directed_cycle_score(A1))
    print(directed_cycle_score(A2))

def test_bic_score_function():
    ################################################
    # Tests for your bic_score function
    ################################################
    data = pd.read_csv("bic_test_data.txt")
    idx_to_var_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    # fit model for G1: A->B->C->D, B->D and get BIC
    # you can also use this as additional tests for your cycle score function
    A1 = np.array([[0, 1, 0, 0],
                   [0, 0, 1, 1],
                   [0, 0, 0, 1],
                   [0, 0, 0, 0]])
    print(bic_score(A1, data, idx_to_var_map), directed_cycle_score(A1))


    # fit model for G2: A<-B->C->D, B->D and get BIC
    A2 = np.array([[0, 0, 0, 0],
                   [1, 0, 1, 1],
                   [0, 0, 0, 1],
                   [0, 0, 0, 0]])
    print(bic_score(A2, data, idx_to_var_map), directed_cycle_score(A2))

    # fit model for G3: A->B<-C->D, B->D and get BIC
    A3 = np.array([[0, 1, 0, 0],
                   [0, 0, 0, 1],
                   [0, 1, 0, 1],
                   [0, 0, 0, 0]])
    print(bic_score(A3, data, idx_to_var_map), directed_cycle_score(A3))

    # fit model for G4: A<-B->C<-D, B->D and get BIC
    A4 = np.array([[0, 0, 0, 0],
                   [1, 0, 1, 1],
                   [0, 0, 0, 0],
                   [0, 0, 1, 0]])
    print(bic_score(A4, data, idx_to_var_map), directed_cycle_score(A4))



def main():
    # This function must be left unaltered at submission (e.g. you can fiddle with this function for debugging, but return it to its original state before you submit)
    test_causal_discovery_function()
    test_directed_cycle_score_function()
    test_bic_score_function()

if __name__ == "__main__":
    # main()
    test_causal_discovery_function()
