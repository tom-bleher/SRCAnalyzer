import sys
from calc import *
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QTableWidgetItem
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from main import Ui_MainWindow  # Import the generated class
import numpy as np
import itertools
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial

def check_point(np_val, pp_val, prec, step_nn):
    """Helper function to check a single point in the parameter space."""
    print(f"\rChecking point: np={np_val:.2f}, pp={pp_val:.2f}", end="")
    for nn_val in np.arange(1, 3 + step_nn, step_nn):
        # pair_nums = [np48, nn48, pp48, np40, nn40, pp40]
        pair_nums = [np_val, nn_val, pp_val, 1, 1, 1]
        r = get_ratios_from_pair_nums(pair_nums)
        if (abs(r[0] - 1.02) < prec and
            abs(r[1] - 1.31) < prec and
            abs(r[2] - 1.17) < prec and
            r[3] > 0 and
            r[4] > 0):
            print()  # New line after finding a valid point
            return (np_val, pp_val, r)
    return None

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Data variables
        self.ratio_data = {}  # Ratio data
        self.pn_data = {}  # Pair number data

        # Precision: numbers after floating point
        self.prec = 4

        self.pn_inputs = [self.ui.box_np48, self.ui.box_nn48, self.ui.box_pp48, self.ui.box_np40, self.ui.box_nn40, self.ui.box_pp40]
        self.ratio_inputs = [self.ui.Reep, self.ui.Reen, self.ui.Ree, self.ui.Reenp, self.ui.Reepp, self.ui.ppp48, self.ui.npn48, self.ui.ppp40, self.ui.npn40]
        self.ratio_inputs_2 = [self.ui.Reep_2, self.ui.Reen_2, self.ui.Ree_2, self.ui.Reenp_2, self.ui.Reepp_2, self.ui.ppp48_2, self.ui.npn48_2, self.ui.ppp40_2, self.ui.npn40_2]
        self.all_inputs = [self.pn_inputs, self.ratio_inputs, self.ratio_inputs_2]

        # Connect button actions
        self.ui.button_findratio.clicked.connect(self.calculate_ratios)  # Calc ratios from pair nums
        self.ui.button_findpn.clicked.connect(self.calc_pair_nums)  # Calc pair nums from ratios
        self.ui.button_findpn_2.clicked.connect(self.calc_pair_nums_r)

        # Save data (ratios/pair nums) to table
        self.ui.buttonSavePN.clicked.connect(self.save_pair_nums)
        self.ui.buttonSavePN_2.clicked.connect(self.save_pair_nums_r)
        self.ui.buttonSaveRatios.clicked.connect(self.save_ratios)

        # Clear buttons
        self.ui.btn_clear_1.clicked.connect(lambda: self.clear(self.pn_inputs))
        self.ui.btn_clear_2.clicked.connect(lambda: self.clear(self.ratio_inputs))
        self.ui.btn_clear_3.clicked.connect(lambda: self.clear(self.ratio_inputs_2))

        self.curr_pn_data_index = -1
        self.ui.left_1.clicked.connect(lambda: self.load_pn_data(-1))
        self.ui.right_1.clicked.connect(lambda: self.load_pn_data(1))
        
        self.norm_by_atom = False
        self.ui.checkNormAtom.clicked.connect(self.toggle_norm_by_atom)

        self.graph_type_index = 0
        self.ui.calcType.addItems(['Find Ratio','Find PN (5)', 'Find PN (4)'])
        self.ui.calcType.currentIndexChanged.connect(self.update_graph_index)
        self.ui.graphButton.clicked.connect(self.createGraph)

        self.ui.button.clicked.connect(self.createCustomGraph3Ratios)
        self.ui.button_2.clicked.connect(self.createCheckNNRegion)

    def update_graph_index(self, index):
        self.graph_type_index = index

    def toggle_norm_by_atom(self):
        self.norm_by_atom = not self.norm_by_atom

    def calculate_ratios(self):
        """Calculate ratios from pair numbers."""
        pair_nums = [float(u.text()) for u in self.pn_inputs]
        ratios = get_ratios_from_pair_nums(pair_nums)
        self.ui.ratio_output.setPlainText(f"{get_ratio_print_msg(ratios)}")

        return ratios, pair_nums

    def calc_pair_nums(self):
        """Calculate pair numbers (e.g #np48/#SRC) from 5 ratios. 
        Returns pair number array (length 6) and array of ratios used (zero for those unused)."""
        ratios = []
        for r in self.ratio_inputs:
            ratios.append(r.text())
        ratios_used = []
        for i in range(len(ratios)):
            if ratios[i] != "":
                if len(ratios_used) < 5:
                    ratios_used.append(i)
                ratios[i] = float(ratios[i])
            else:
                ratios[i] = 0
        print(ratios)
        select_ratio_eqs(ratios_used)
        pair_nums, src_ratio, matrix = find_pair_nums(ratios, self.norm_by_atom)

        msg = get_pair_nums_msg(pair_nums)
        self.ui.pn_output.setPlainText(msg)
        return pair_nums, src_ratio, ratios
    
    def calc_pair_nums_r(self):
        """Calculate pair numbers ratio (e.g #np48/#np40) from 4 ratios."""
        ratios = []
        for r in self.ratio_inputs_2:
            ratios.append(r.text())
        ratios_used = []
        for i in range(len(ratios)):
            if ratios[i] != "":
                ratios_used.append(i)
                ratios[i] = float(ratios[i])
            else:
                ratios[i] = 0
        print("ratios used", ratios_used)
        print("ratios", ratios)
        select_ratio_eqs(ratios_used)
        pair_nums, src_ratio = find_pair_nums_r(ratios, self.norm_by_atom)

        msg = get_pair_nums_msg(pair_nums)
        self.ui.pn_output_2.setPlainText(msg)
        return pair_nums, src_ratio, ratios

    # Save data ----------------------

    def save_pair_nums(self):
        """Save pair numbers to the table."""
        model_name = self.ui.box_PNname.text()
        if model_name == "" or model_name in self.pn_data.keys():
            return
        pair_nums, src_ratio, ratios = self.calc_pair_nums()
        self.pn_data[model_name] = pair_nums

        # Add ratios from which pair numbers were computed
        self.ratio_data[model_name] = ratios
        self.insert_ratios_to_table(model_name, ratios)
        self.insert_pn_to_table(model_name, pair_nums, src_ratio)

    def save_pair_nums_r(self):
        """Save pair numbers RATIOS to the table (calculated from 4 ratios scattering cross sections, as opposed to 5 in save_pair_nums() )."""
        model_name = self.ui.box_PNname_2.text()
        if model_name == "" or model_name in self.pn_data.keys():
            return
        pair_nums, src_ratio, ratios = self.calc_pair_nums_r()
        self.pn_data[model_name] = pair_nums

        # Add ratios from which pair numbers were computed
        self.ratio_data[model_name] = ratios
        self.insert_ratios_to_table(model_name, ratios)
        self.insert_pn_to_table(model_name, pair_nums, src_ratio)

    def save_ratios(self):
        """Save ratios to the table."""
        model_name = self.ui.box_RATIOname.text()
        if model_name == "" or model_name in self.ratio_data.keys():
            return
        ratios, pair_nums = self.calculate_ratios()
        src_ratio = np.sum(pair_nums[0:3]) / np.sum(pair_nums[3:])  # src48/src40
        self.ratio_data[model_name] = ratios

        # Insert RATIO data to table
        self.insert_ratios_to_table(model_name, ratios)
        self.insert_pn_to_table(model_name, pair_nums, src_ratio)

    # Insert to table ----------------------------

    def insert_pn_to_table(self, model_name, pair_nums, src_ratio):
        """Insert pair number data into the table."""
        col_count = self.ui.pn_table.columnCount()
        self.ui.pn_table.insertColumn(col_count)
        self.ui.pn_table.setItem(0, col_count, QTableWidgetItem(model_name))
        for i in range(1, 7):
            self.ui.pn_table.setItem(i, col_count, QTableWidgetItem(str(round(pair_nums[i - 1], self.prec))))
        self.ui.pn_table.setItem(7, col_count, QTableWidgetItem(str(round(src_ratio, self.prec))))

    def insert_ratios_to_table(self, model_name, ratios):
        """Insert ratio data into the table."""
        col_count = self.ui.ratio_table.columnCount()
        self.ui.ratio_table.insertColumn(col_count)
        self.ui.ratio_table.setItem(0, col_count, QTableWidgetItem(model_name))
        for i in range(1, 10):
            curr_ratio = round(ratios[i - 1], self.prec)
            if curr_ratio != 0:
                self.ui.ratio_table.setItem(i, col_count, QTableWidgetItem(str(curr_ratio)))

    def clear(self, container):
        for u in container:
            u.setText("")
    
    def load_pn_data(self, dir):
        #TODO: save data on index=-1
        self.curr_pn_data_index += dir
        if self.curr_pn_data_index <= -1:
            self.curr_pn_data_index = -1
            self.clear(self.pn_inputs)
            self.ui.model_label1.setText("")
            return
        try:
            keys = list(self.pn_data.keys())
            model_name = keys[self.curr_pn_data_index]
            self.ui.model_label1.setText(str(model_name))
            curr_data = self.pn_data[model_name]
            
            for i in range(len(self.pn_inputs)):
                self.pn_inputs[i].setText(str(round(curr_data[i], self.prec)))
        except:
            print(f"No data found at index {self.curr_pn_data_index}")
            self.curr_pn_data_index -= dir
        
    def createGraph(self):
        # Read DATA from GUI
        try:
            ranges = list(map(float, self.ui.rangeEdit.text().split(',')))
            steps = list(map(float, self.ui.stepEdit.text().split(',')))
            exprs = list(map(str,  self.ui.exprEdit.text().split(',')))
        except:
            print("Invalid range/step.")
            return
        if len(ranges) != len(steps) :
            print("Mismatch in number of <step> and <range> parameters.")
            return
        input_container = self.all_inputs[self.graph_type_index]
        input_data = []
        indp_index = [] # indices of independent variables
        for u in input_container:
            input_data.append(u.text())
        data_found = []
        for i in range(len(input_data)):
            if input_data[i] != "":
                data_found.append(i)
                
                if "T" in input_data[i]:
                    indp_index.append(i)
                    input_data[i] = float(input_data[i][2:-1])
                else:
                    input_data[i] = float(input_data[i])
            else:
                input_data[i] = 0
        if (len(ranges)>2 or len(indp_index)>2):
            print("Too many independent variables - max 2.")
            return
        if len(ranges) != len(indp_index):
            print("Mismatch between num. of independent vars. and num. of range values.")
            return

        # Check for correct length
        correct_data_len =  [6,5,4][self.graph_type_index]
        if len(data_found) != correct_data_len:
            print(f"Incorrect number of inputs. Should be {correct_data_len}")
            return
        print("ranges", ranges)
        print("steps", steps)
        print("input_data", input_data)
        print("data_found", data_found)

        vals0 = np.array(input_data)[np.array(indp_index).astype('int')] # center values of independent variables
        vals = [] # all possible values for each indp. var.
        for j in range(len(indp_index)):
            vals.append(np.arange(vals0[j]-ranges[j]/2, vals0[j]+ranges[j]/2, steps[j]))
        func = [get_ratios_from_pair_nums, find_pair_nums, find_pair_nums_r][self.graph_type_index]

        # Names for title and expression
        pn_names = ['np48', 'nn48', 'pp48', 'np40', 'nn40', 'pp40']
        ratio_names = ["p", "n", "e", "np", "pp", "pp48", "np48", "pp40", "np40"]
        names = [ratio_names, pn_names, pn_names][self.graph_type_index] # name of dependent variable

        # Names for independent variables
        indp_names = pn_names
        if names == pn_names:
            indp_names = ratio_names
            select_ratio_eqs(data_found)


        # Calculate result
        results = [[] for i in exprs]
        for v in itertools.product(*vals):
            for j in range(len(v)):
                input_data[indp_index[j]] = v[j]
            calc = func(input_data) # output of calculations (list of pair nums / ratios)
            if self.graph_type_index!=0:
                calc = calc[0]
            # Calculate relevant value from given expression
            for j in range(len(exprs)):
                Expr = exprs[j]
                for i in range(len(names)):
                    Expr = Expr.replace(f'{names[i]}', str(calc[i]))
                try:
                    results[j].append(eval(Expr))
                except:
                    print("Invalid expression.")
                    return
                
        results = np.array(results)
        
        # Display Phyiical Region on pair number space --------------
        fig = plt.figure()
        z1 = results[0].flatten()
        z2 = results[1].flatten()
        # z3 = results[2].flatten()
        ax = fig.add_subplot(111)
        ax.scatter(z1,z2, c='r')
        ax.set_xlabel(f"{exprs[0]}")
        ax.set_ylabel(f"{exprs[1]}")
        plt.plot()

        # Display Physical Region on R space --------------
        # x, y = np.meshgrid(vals[0], vals[1])
        # z1 = results[0].reshape(len(vals[0]), len(vals[1])).T
        # z2 = results[1].reshape(len(vals[0]), len(vals[1])).T
        # z3 = results[2].reshape(len(vals[0]), len(vals[1])).T
        
        # region_data = np.zeros_like(z1)
        # region_data[(z1>0.5)&(z1<2)  &(z2>1)&(z2<3)   &(z3>0.01)&(z3<3)] = 3
        # region_data[(z1>0.5)&(z1<1.5)&(z2>1)&(z2<2.33)&(z3>0.01)&(z3<2)] = 2
        # region_data[(z1>0.5)&(z1<1.25)  &(z2>1)&(z2<2)&(z3>0.01)&(z3<1.5)] = 1

        # Single plot with ALL discrete levels (0<pp<3, 1<nn<3, 0.5<np<2)
        # region_data[(z1>0.5)&(z1<2)  &(z2>1)&(z2<3)&(z3>0.01)&(z3<3)] = 1
        

        # fig5 = plt.figure()
        # ax5 = fig5.add_subplot(111)
        # im5 = ax5.imshow(region_data, cmap='viridis',
        #                  extent=[x.min(), x.max(), y.min(), y.max()],
        #                  origin='lower', aspect='auto')
        # cbar = fig5.colorbar(im5, ax=ax5)
        # cbar.set_label("Region Code")
        # ax5.set_title("Physical Region")
        # ax5.set_xlabel("R(e,e'np)")
        # ax5.set_ylabel("R(e,e'pp)")
        # plt.show()


        # Display Graphs
        for i in range(len(exprs)):
            # plt.title(f"{expr} vs. R({indp_names[indp_index[0]]})")
            # plt.xlabel(f"R({indp_names[indp_index[0]]})")
            # plt.ylabel(f"{expr}")
            if len(indp_index) == 1:
                fig = plt.figure()
                plt.plot(vals[0], results[i].flatten())
                plt.xlabel(f"R({indp_names[indp_index[0]]})")
                plt.ylabel(f"{exprs[i]}")
                plt.title(f"{exprs[i]} vs. R({indp_names[indp_index[0]]})")
                plt.show()
            elif len(indp_index) == 2:
                x, y = np.meshgrid(vals[0], vals[1])
                z = results[i].reshape(len(vals[0]), len(vals[1])).T
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(x, y, z, cmap='viridis')
                ax.set_xlabel(f"R({indp_names[indp_index[0]]})")
                ax.set_ylabel(f"R({indp_names[indp_index[1]]})")
                if z.max() > 15:
                    ax.set_zlim(0, 4)
                ax.set_zlabel(f"{exprs[i]}")
                ax.set_title(f"{exprs[i]} vs. R({indp_names[indp_index[0]]}) and R({indp_names[indp_index[1]]})")
                plt.show()

                # Add heightmap with contour lines
                fig2 = plt.figure()
                ax2 = fig2.add_subplot(111)
                c = ax2.contourf(x, y, z, cmap='viridis')
                ax2.contour(x, y, z, colors='black', linewidths=0.5)
                fig2.colorbar(c, ax=ax2)
                ax2.set_xlabel(f"R({indp_names[indp_index[0]]})")
                ax2.set_ylabel(f"R({indp_names[indp_index[1]]})")
                ax2.set_title(f"{exprs[i]} (heightmap)")
                plt.show()

                # Add imshow plot
                fig3 = plt.figure()
                ax3 = fig3.add_subplot(111)
                im = ax3.imshow(z, cmap='viridis',
                                extent=[x.min(), x.max(), y.min(), y.max()],
                                origin='lower', aspect='auto')
                fig3.colorbar(im, ax=ax3)
                ax3.set_xlabel(f"R({indp_names[indp_index[0]]})")
                ax3.set_ylabel(f"R({indp_names[indp_index[1]]})")
                ax3.set_title(f"{exprs[i]} (imshow)")

                # Draw the line where z=0.65
                xx = np.linspace(x.min(), x.max(), z.shape[1])
                yy = np.linspace(y.min(), y.max(), z.shape[0])
                XX, YY = np.meshgrid(xx, yy)
                # ax3.contour(XX, YY, z, levels=[0.65], colors='white', linewidths=2)
                
                # Compute approximate midpoints
                mx = (x.min() + x.max()) / 2
                my = (y.min() + y.max()) / 2

                # c1 = ax3.contour(XX, YY, np_r, levels=[1.4], colors='white', linewidths=2)
                # ax3.clabel(c1, [1.4], fmt="np48/np40=1.4", inline=True, fontsize=8,
                #            manual=[(mx-0.1, my)])

                # c2 = ax3.contour(XX, YY, nn_r, levels=[2.0], colors='white', linewidths=2)
                # ax3.clabel(c2, [2.0], fmt="nn48/nn40=2", inline=True, fontsize=8,
                #            manual=[(mx-0.1, my)])

                # ax3.scatter(1.28, 1.2, color='red', s=50)

                plt.show()

    def createCustomGraph3Ratios(self):
        # Assume first 3 ratio inputs in PN (4) mode are set
        base_ratios = []
        for r in self.ratio_inputs_2[:3]:
            val = r.text()
            if val == "":
                return
            base_ratios.append(float(val))
        select_ratio_eqs([0, 1, 2,3])

        # Sweep the 4th ratio (index=3 in ratio_inputs_2)
        sweep_min, sweep_max, step = 1.0, 1.5, 0.01
        np_values, nn_values, pp_values = [], [], []
        Reepp, Reenp = [], []
        for ratio4 in np.arange(sweep_min, sweep_max, step):
            ratios = base_ratios + [ratio4, 0, 0, 0, 0, 0]  # length 9
            pair_nums, _ = find_pair_nums_r(ratios, False)
            # pair_nums = [np48, nn48, pp48, np40, nn40, pp40]
            np_ratio = pair_nums[0] / pair_nums[3] # np48/np40
            nn_ratio = pair_nums[1] / pair_nums[4] # nn48/nn40
            pp_ratio = pair_nums[2] / pair_nums[5] # pp48/pp40
            np_values.append(np_ratio)
            nn_values.append(nn_ratio)
            pp_values.append(pp_ratio)
            R = get_ratios_from_pair_nums(pair_nums)
            Reepp.append(R[4])
            Reenp.append(R[3])

        plt.title("")
        plt.xlabel("np48/np40")
        plt.plot(np_values, nn_values, '-', linewidth=3.0)
        plt.plot(np_values, Reepp, '--')
        plt.plot(np_values, Reenp, '--')
        plt.legend(["nn48/nn40","R(e,e'pp)", "R(e,e'np)"])

        # Mark x=1.4 on the x-axis
        plt.axvline(1.4, color='red', linestyle='--')
        plt.text(1.4, 0.0, "Combinatorial Limit", rotation=90, color='red',
                 va='bottom', ha='right')

        plt.show()

    def createCheckNNRegion(self):
        print("Starting region check...")
        step = 0.02
        prec = 0.05
        step_nn = 0.1

        # Create parameter grid
        np_vals = np.arange(0.5, 2 + step, step)
        pp_vals = np.arange(0.5, 1.5 + step, step)
        points = [(np_val, pp_val) for np_val in np_vals for pp_val in pp_vals]

        # Initialize multiprocessing pool
        num_cores = mp.cpu_count()
        pool = mp.Pool(processes=num_cores)

        # Create partial function with fixed parameters
        check_point_partial = partial(check_point, prec=prec, step_nn=step_nn)

        # Process points in parallel
        results = pool.starmap(check_point_partial, points)
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
        
        # Plot results
        plt.figure()
        plt.scatter(valid_x, valid_y)
        plt.xlim(0, 3)
        plt.ylim(0.5, 2)
        plt.xlabel("np48/np40")
        plt.ylabel("pp48/pp40")
        plt.title("Points with nn48/nn40 in [1,3] satisfying constraints")
        plt.show()

        

        

   
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
