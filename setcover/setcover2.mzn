int: nsets;
int: nitems;

set of int: SETS = 1..nsets;
set of int: ITEMS = 1..nitems;

array[SETS] of var bool: x;               % set choices
array[SETS] of float: costs;              % set costs
array[SETS] of set of int: covers;

% example problem instance

% nsets = 6;
% nitems = 9;

% costs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

% covers = [
% {0, 3},
% {0, 1, 2, 5, 6, 8},
% {1, 2, 5, 6, 8},
% {6, 7, 8},
% {0, 3, 4, 5, 6},
% {1, 2, 7, 8}
% ];


% Every item must be covered by at least one set:
constraint forall(i in ITEMS)(sum(s in SETS where (i - 1) in covers[s])(x[s]) >= 1);

% If there is only one choice for the item, then the variable must be chosen:
constraint forall(s in SETS)(
  forall(i in covers[s])(
    if not exists([(i - 1) in covers[s2] | s2 in SETS where s2 != s]) then
      x[s] = true
    else
      true
    endif
  )
);

% Minimize the cost of the chosen sets
solve :: seq_search([
  bool_search(x, first_fail, indomain_random, complete),
])
minimize sum(i in SETS)(costs[i] * x[i]);
  
  
  