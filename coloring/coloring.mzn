int: n_nodes;
set of int: NODES = 1..n_nodes;
set of int: COLORS = 1..n_nodes;
array[NODES] of set of NODES: neighbors;
array[NODES] of var COLORS: colors;

constraint forall(i in NODES)( forall(j in neighbors[i])(colors[i] != colors[j]) );

solve minimize max(colors);
