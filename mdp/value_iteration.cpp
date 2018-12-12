#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

using namespace std;
vector<double> compute(vector<double> reward, int n_states, double discount, vector<vector<double>> transit_table){
	vector<double> value;
	for(int i=0; i<n_states; i++) value.push_back(0.0);
	int step = 0;
	double thresh = 0.1;
	double max_update = (double) INT_MAX;
	while (max_update > thresh){
		max_update = 0.0;
		step += 1;

		for (int i=0; i<n_states; i++){
			double max_val = (double) (-INT_MAX);
			double new_val;
			for (int j=0; j< (int)transit_table[i].size(); j++){
				new_val = reward[i] + discount * value[transit_table[i][j]];
				if (new_val > max_val){
					max_val = new_val;
				}
			}
			if (abs(max_val-value[i]) > max_update) max_update = abs(max_val-value[i]);
			value[i] = max_val;
		}
		if (step > 1000){
			cout<<"max iter exceeded!"<<endl;
		}
	}
  cout << step << endl;
	return value;
}

PYBIND11_MODULE(value_iteration, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("compute", &compute, "value iteration compute");
}
