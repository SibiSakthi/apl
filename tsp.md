# TSP
Simulated Annealing for the Traveling Salesman Problem
The provided code is an implementation of the Simulated Annealing (SA) algorithm to solve the Traveling Salesman Problem (TSP). The TSP is a classic algorithmic problem in the field of computer science and operations research which focuses on optimization.

Code Overview
1. `parse_input_file(file_path)`: This function reads city coordinates from a file. It assumes that each line in the file contains two floats representing the X and Y coordinates of a city. It returns a list of city coordinates.

2. `distance(cities, cityorder)`: This function calculates the total distance traveled for a given order of cities. It iterates through the order of cities and calculates the distance between them. It then returns the total distance.

3. `simulated_annealing(cities, initial_order, initial_temperature, cooling_rate, max_iterations)`: This function uses the simulated annealing algorithm to find the best order of cities. It starts with an initial order, initial temperature, and other parameters. It iterates over a fixed number of iterations, generating neighboring solutions by swapping two cities. It accepts the neighboring solution if it's better or with a certain probability based on the change in distance and the current temperature. The best order and distance found during the process are returned, along with a list of improvements over iterations.

4. Example Usage: This section demonstrates how to use the functions to solve the TSP problem for a set of example cities. It creates a list of cities, generates an initial random order for them, sets parameters for simulated annealing, and then calls the `simulated_annealing` function to find the best order. It also prints the best order, best distance, and a list of percentage improvements over iterations.

5. Finally, the code plots the best TSP tour using Matplotlib. It shows the order in which the cities should be visited to minimize the total distance traveled.

The percentage improvement is calculated by comparing each improvement to the previous iteration. This gives you an idea of how the solution is getting better over time.


# Algorithm Explanation
Simulated Annealing is a probabilistic technique for approximating the global optimum of a given function. Specifically, it is a metaheuristic to approximate global optimization in a large search space for an optimization problem. It is often used when the search space is discrete (e.g., all tours that visit a given set of cities).

In this code, SA is used to solve the TSP, which states that: “Given a list of cities and the distances between each pair of cities, what is the shortest possible route that visits each city exactly once and returns to the origin city?”

The algorithm starts with a random solution (i.e., a random order of cities) and calculates its cost using the distance function. It then generates a neighboring solution by swapping two cities in the order. If this neighbor solution has a lower cost, it becomes the new current solution. If it has a higher cost, it might still become the new current solution based on a probability that decreases with worsening cost differences and decreasing ‘temperature’. This allows for exploration of suboptimal solutions in the search for an optimal solution.

The ‘temperature’ parameter is crucial in SA. It guides the probability of accepting worse solutions and is gradually decreased during the search to allow more focus on exploitation of good solutions.

# Conclusion
This code provides an efficient and effective method to solve the TSP using Simulated Annealing. By adjusting parameters like initial temperature, cooling rate, and maximum iterations, one can control the trade-off between solution quality and computational time.

# output :
Best Distance: 7.457800713283651

intial improvements =[1.5993624543705607, 2.5523812073666137, 5.210971753152845, 0.43948428633557685, 5.709816548002987, 2.2006719668007766, 4.900160050353959, 3.032145540433157, 2.3183308557888287, 0.2661349813662194, 1.2621424029316692, 0.8380815883287109, 0.5428059445157144, 1.2477352078445423, 5.758795228791658, 0.31290940549400115, 0.40100537872884107, 0.8233034302736706, 4.1324520351849365, 2.107117708259182, 0.6435378584190083, 3.980973415794258, 0.5781726810516457, 0.8090623092063358, 2.727859090148963, 1.222891883021843, 3.433008338080102, 2.5975270984684395, 0.7935431530782685, 1.8360386903708863, 0.26050770970497233, 0.5498728253311352, 0.14561950020480935, 1.0916428305284718, 0.2063829613513737, 0.9246737524226366, 0.05686878370455445, 0.8609782659449662, 1.929429944042901, 1.1401920729319124, 0.12860612191018825, 0.16948237337903382, 0.08539195028282623, 0.037756256221897495, 2.289114235136558, 0.9544139881328068, 0.0463579018263474, 3.7554039014151703, 0.09183302576840711, 0.33378410121644636, 0.5181624017126659, 0.8011190503454041, 0.7743533430788979, 1.234612552219267, 1.5584536533480218, 1.1553279745466007, 0.29786097195853356, 0.0840022034346966, 0.051846694852117285, 0.40301330372962574, 0.34313317334475807, 0.06732762997075259, 0.9185604217996017, 0.5476235910297347, 0.670113508175279, 2.380685195856272, 0.19152789184964256, 0.8948728893635416, 0.38179724503882195, 1.20823379213099, 1.0328532397841181, 0.5697041428370665, 0.07702341570197145, 0.2368664816713157, 1.2939560054699806, 0.3903255373935267, 0.955067721700765, 1.7602448532503006, 1.9068453273129116, 1.5388220978119511, 0.22930991150156832, 1.6038309598644551, 0.3112819685841321, 0.3774239921361686, 1.0303518841955617]
<img src="40 city.png" alt="lenier regression" width="400" height="400" />
