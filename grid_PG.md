## Grid World Policy Grammar

The complexity of the policy grammar should depend on how much information the hands (algorithm) have about the environment of the world. 
If the algorithm is given complete information, the hierarchy is relatively flat; we can define: 

`N = {Deposit}`
 
 `Pi = {1 -> object (pi_1), 1 -> goal (pi_2), 2 -> object (pi_3), 2 -> goal (pi_4)}`
 
 `Psi = {Deposit -> pi_1 | pi_2 | pi_3 | pi_4}`

I'm using the notation of the proposal here, so `N` is the set of all non-terminals (in this case containing the single goal `Deposit`,
as in "Deposit all objects in the goal area"), `Pi` is the set of all policy primitives, and `Psi` is the set of all production rules. 
In this case, because the algorithm has full information about the environment, there is only one production rule: we can expand the 
`Deposit` goal to the policy primitives, which allow us to move the hands to the objects or the goal. 

If the algorithm is not given full information about the environment, however, we can make the hierarchy less flat. In this case, 
the hands have to locate the objects and goal areas, so we can expand our policy grammar definition: 

`N = {Deposit, locate_obj, locate_goal}`

`Pi = {1 -> object (pi_1), 1 -> goal (pi_2), 2 -> object (pi_3), 2 -> goal (pi_4)}`

`Psi = {Deposit -> locate_obj | locate_goal, locate_obj -> pi_1 | pi_3, locate_goal -> pi_2 | pi_4 }`

That is, we now expand `Deposit` to the subgoals `locate_obj` (locate an object) and `locate_goal`. This expands the hierarchy and 
will allow us to experiment with a non-trivial hierarchy. As far as I can tell, this is as complex as we can make the policy grammar 
while still working with the simple task "deposit objects in the goal." I think this is a good start, and we can expand the goal 
(perhaps add a time limit, navigation of obstacles, etc.) once the algorithm has been shown to work on this very simple system. 
