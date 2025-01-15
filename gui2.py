import numpy as np
import itertools
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
from calc import *

def check_point(np_val, pp_val, prec, step_nn):
    """Helper function to check a single point in the parameter space."""
    print(f"\rChecking point: np={np_val:.2f}, pp={pp_val:.2f}", end="")
    for nn_val in np.arange(1, 3 + step_nn, step_nn):
        # pair_nums = [np48, nn48, pp48, np40, nn40, pp40]
        pair_nums = [np_val, nn_val, pp_val, 1, 1, 1]
        r = get_ratios_from_pair_nums(pair_nums)
        if any(np.isnan(r)) or any(np.isinf(r)):
            continue
        if (abs(r[0] - 1.02) < prec and
            abs(r[1] - 1.28) < prec and
            abs(r[2] - 1.17) < prec and
            r[3] > 0 and
            r[4] > 0):
            print()  # New line after finding a valid point
            return (np_val, pp_val, r)
    return None

def check_nn_region():
    """Check the region of valid points."""
    print("Starting region check...")
    step = 0.01
    prec = 0.05
    step_nn = 0.01
    results = []
    pool = None

    try:
        # Create parameter grid
        np_vals = np.arange(0.5, 2 + step, step)
        pp_vals = np.arange(0, 3 + step, step)
        points = [(np_val, pp_val) for np_val in np_vals for pp_val in pp_vals]

        # Initialize multiprocessing pool
        num_cores = mp.cpu_count()
        pool = mp.Pool(processes=num_cores)

        # Create partial function with fixed parameters
        check_point_partial = partial(check_point, prec=prec, step_nn=step_nn)

        # Process points in parallel
        results = pool.starmap(check_point_partial, points)

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
    
    finally:
        if pool:
            pool.close()
            pool.join()

        # Filter out None results and separate valid points
        valid_points = [r for r in results if r is not None]
        if valid_points:
            valid_x, valid_y, ratios = zip(*valid_points)
        else:
            valid_x, valid_y, ratios = [], [], []

        # Save results to file
        with open("points.csv", "w") as f:
            f.write("np48/np40,pp48/pp40,R(e,e'p),R(e,e'n),R(e,e'),R(e,e'np),R(e,e'pp)\n")
            for x_val, y_val, r in zip(valid_x, valid_y, ratios):
                f.write(f"{x_val},{y_val},{r[0]},{r[1]},{r[2]},{r[3]},{r[4]}\n")
        
        try:
            # Plot results only if there are valid points
            if valid_points:
                plt.figure()
                plt.scatter(valid_x, valid_y)
                plt.xlim(0, 3)
                plt.ylim(0.5, 2)
                plt.xlabel("np48/np40")
                plt.ylabel("pp48/pp40")
                plt.title("Points with nn48/nn40 in [1,3] satisfying constraints")
                plt.show()
        except Exception as e:
            print(f"Error during plotting: {str(e)}")

if __name__ == "__main__":
    check_nn_region()
