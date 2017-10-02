include "globals.mzn";

int: n_nodes;
set of int: NODES = 1..n_nodes;
set of int: COLORS = 1..n_nodes;
array[NODES] of set of NODES: neighbors;
array[NODES] of var COLORS: colors;
array[NODES] of NODES: nodes_by_degree;

constraint forall(i in NODES)( forall(j in neighbors[i])(colors[i] != colors[j]) );

solve :: seq_search([
  int_search(nodes_by_degree, input_order, indomain_max, complete),
  int_search(colors, first_fail, indomain_min, complete),
])
minimize max(colors);
