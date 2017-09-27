int: nsets;
int: nitems;

set of int: SETS = 1..nsets;
set of int: ITEMS = 1..nitems;

array[SETS] of var bool: x;               % set choices
array[SETS] of float: costs;              % set costs
array[ITEMS] of set of SETS: covered_by;  % items covered by

% example problem instance

% nsets = 6;
% nitems = 9;

% costs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

% covered_by = [
% {1, 2, 5},  % 0
% {2, 3, 6},  % 1
% {2, 3, 6},  % 2
% {1, 5},  % 3
% {5},  % 4
% {2, 3, 5},  % 5
% {2, 3, 4, 5},  % 6
% {4, 6},  % 7
% {2, 3, 4, 6},  % 8
% ];

% covers = [
% {0, 3},
% {0, 1, 2, 5, 6, 8},
% {1, 2, 5, 6, 8},
% {6, 7, 8},
% {0, 3, 4, 5, 6},
% {1, 2, 7, 8}
% ];

% Every item must be covered by at least one set
constraint forall(i in ITEMS)(exists([j in covered_by[i] | j in SETS where x[j]]));

% If an item is covered only by a single set, then that set must be part of the solution
constraint forall(i in ITEMS)(if length(covered_by[i]) = 1 then x[i] = true else true endif);

% Minimize the cost of the chosen sets
solve :: int_search(x, first_fail, indomain_min, complete)
  minimize sum(i in SETS)(costs[i] * x[i]);