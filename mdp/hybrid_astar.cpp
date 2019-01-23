#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
// #include "planner_utils.cpp"
#include <time.h>
#include <queue>
#include <cmath>
#include <math.h>
#include <unordered_map>
#include<bits/stdc++.h>
#include <stdlib.h> 

#define INF INT_MAX

using namespace std;

namespace py = pybind11;

class HybridAStarNode {
    public:
        // check what all params needed or not
        double x, y, theta;
        int x_discrete, y_discrete, theta_discrete;
        double f, g, h;
        vector<HybridAStarNode *> children;
        HybridAStarNode * parent;
        // need to store the action also that got to this state

        HybridAStarNode(){
            this->parent = NULL;
            //ac = NULL;
            this->g = INF; //check this
            this->h = 0;
            this->f = this->g+this->h;
        }

        HybridAStarNode(double _x, double _y, double _theta, double _x_grid_size, double _y_grid_size, double _theta_grid_size) : HybridAStarNode(){
            this->x = _x;
            this->y = _y;
            this->theta = _theta;
            this->x_discrete = int(_x / _x_grid_size);
            this->y_discrete = int(_y / _y_grid_size);
            this->theta_discrete = int(_theta / _theta_grid_size);
            // printf("x, x_dis: %.2f, %d, %.2f, %d\n", _x, this->x_discrete, _y, this->y_discrete);
            // cout<<"Cur node stats: "<<_x<<" "<<_y<<" "<<_theta*180./M_PI<<" "<<this->x_discrete<<" "<<this->y_discrete<<" "<<this->theta_discrete<<endl;
            // cout<<"grid_sizes: "<<_x_grid_size<<" "<<_y_grid_size<<endl;
        }

        // friend ostream& operator<<(ostream& os, const AStarNode& n)
        // {
        //     os << endl;
        //     // os << x << " " << y << " " << z << " " << x_round << " " << y_round << " " << ang_round << endl;
        //     return os;
        // }

        void update_f(){
            this->f = this->g + this->h;
        }

        bool operator==(const HybridAStarNode& rhs) const
        {
            if (this->x_discrete == rhs.x_discrete && this->y_discrete == rhs.y_discrete
                && this->theta_discrete == rhs.theta_discrete) return true;
            else return false;
        }
};

class NodeCompare
{
public:
    bool operator()(HybridAStarNode * p1, 
                    HybridAStarNode * p2)
    {
        return p1->f >= p2->f; // check this
    }
};

// struct NodeComparator
// {
//     bool operator()(const AStarNode& lhs, const AStarNode& rhs) const
//     {
//         return lhs == rhs; // need to check this how it works
//     }
// };

struct NodeHasher2
{
    size_t operator()(const HybridAStarNode n) const{
        // return std::hash<int>()(1000* n.x + n.y) ^ std::hash<int>()(n.y); // check if this is symmetric for x and y
        return std::hash<int>()((100000*n.theta) + (1000*n.x) + n.y); // check if this is symmetric for x and y
    }
};

struct NodeHasher
{
    size_t operator()(const HybridAStarNode* n) const{
        // return std::hash<int>()(n->x) ^ std::hash<int>()(n->y);
        return std::hash<int>()((1000000*n->theta) + (1000*n->x) + n->y); // check if this is symmetric for x and y
    }
};

class HybridAStar{
    public:
        int path_size;
        HybridAStarNode * start;
        HybridAStarNode * goal;
        double collision_threshold;

        vector<vector<double>> reward;
        vector<tuple<double, double, double>> plan;

        priority_queue<HybridAStarNode*, vector<HybridAStarNode*>, NodeCompare> open_nodes_queue;// check the declaration
        unordered_map<HybridAStarNode*, double, NodeHasher> closed_list;
        unordered_map<HybridAStarNode, double, NodeHasher2> open_nodes_map;

        double x_grid_size, y_grid_size, theta_grid_size;
        double v, L, del_t;

        // for motion primitive
        vector<double> turning_angles {0., 15., -15.};

        HybridAStar(vector<vector<double>> _reward, double start_x, double start_y, double start_theta, double goal_x, double goal_y, 
                    double goal_theta, double _x_grid_size, double _y_grid_size, double _theta_grid_size, double _collision_threshold,
                    double _vel, double _L, double _del_t){
            // need to see how to interface with the reward and what inputs to take
            // initialize the vars here
            // cout<<_reward.size()<<" "<<_reward[0].size()<<endl;
            this->start = new HybridAStarNode(start_x, start_y, start_theta, _x_grid_size, _y_grid_size, _theta_grid_size);
            this->goal = new HybridAStarNode(goal_x, goal_y, goal_theta, _x_grid_size, _y_grid_size, _theta_grid_size);
            this->collision_threshold = _collision_threshold;

            this->start->h = compute_heuristic(*(this->start));
            this->start->g = 0;
            this->start->update_f();

            this->open_nodes_queue.push(this->start); // can print values here and see
            this->open_nodes_map[*(this->start)] = this->start->g; 
            // check this also, node comparison also have to see

            this->x_grid_size = _x_grid_size;
            this->y_grid_size = _y_grid_size;
            this->theta_grid_size = _theta_grid_size;

            this->v = _vel;
            this->L = _L;
            this->del_t = _del_t;
            // for (int i=0; i< (int)_reward.size(); i++) this->reward.push_back(_reward[i]);
            //construct reward in proper way
            // cout<<"here2 "<<_reward.size()<<" "<<_reward[0].size()<<endl;
            for (int i=0; i< (int) _reward.size(); i++){
                // cout<<"here\n";
                vector<double> cur_row;
                for (int j=0; j< (int)_reward[i].size(); j++){
                    cur_row.push_back(_reward[i][j]);
                    // cout<<_reward[i][j]<<" ";
                }
                this->reward.push_back(cur_row);
            }
            cout<<_reward[0][0]<<" "<<this->reward[0][0]<<endl;
            cout<<"Initializing complete2\n";
        }

        double compute_dist(HybridAStarNode & n1, HybridAStarNode & n2){
            double val = sqrt(pow(n1.x - n2.x, 2) + pow(n1.y - n2.y, 2));
            return val;
        }

        double compute_heuristic(HybridAStarNode & n){
            // take the euclidean distance for now
            // check the sqrt and otehr things
            // return 0.;
            double val = compute_dist(n, *this->goal); //sqrt(pow(n.x - this->goal->x, 2) + pow(n.y - this->goal->y, 2));
            cout<<val<<endl;
            return val;
        }

        HybridAStarNode * get_next_top(){
            HybridAStarNode * cur_node = this->open_nodes_queue.top();
            while (this->open_nodes_map.find(*cur_node) != (this->open_nodes_map).end() &&
                    cur_node->g > (this->open_nodes_map.find(*cur_node))->second)
            {
                this->open_nodes_queue.pop();
                cur_node = this->open_nodes_queue.top();
            }
            this->open_nodes_queue.pop();
            return cur_node;
        }

        bool is_goal(HybridAStarNode & n){
            // i.e. if lie in same bucket, then its the goal
            // if (n.x_discrete == this->goal->x_discrete && n.y_discrete == this->goal->y_discrete && n.theta_discrete == this->goal->theta_discrete) return true;
            if (n.x_discrete == this->goal->x_discrete && n.y_discrete == this->goal->y_discrete) return true;
            else return false;
        }

        vector<tuple<double, double, double>> make_final_plan(HybridAStarNode * cur_node){
            vector<tuple<double, double, double>> plan;
            while (cur_node->parent != NULL){
                // cout<<"here3"<<endl;
                // cout<<*(cur_node->ac)<<endl;
                // cout<<cur_node->ac<<endl;
                // plan.push_back(*(cur_node->ac));
                plan.push_back(make_tuple(cur_node->x, cur_node->y, cur_node->theta));
                cur_node = cur_node->parent;
            }
            plan.push_back(make_tuple(cur_node->x, cur_node->y, cur_node->theta));
            reverse(plan.begin(), plan.end()); // check this also
            return plan;
        }

        vector<HybridAStarNode*> get_successors(HybridAStarNode * n){
            double x = n->x, y = n->y, theta = n->theta;

            // cout<<"\nInside get successors\n";
            // cout<<"Cur node: "<<n->x<<" "<<n->y<<" "<<n->theta*180./M_PI<<endl;
            vector<HybridAStarNode *> successors;
            // need to add for reverse direction as well, for now its just forward
            for (double ang: this->turning_angles){
                double new_theta = theta + this->v*tan(ang*M_PI/180.)/this->L;
                new_theta = fmod(new_theta, 2.*M_PI);
                double move_theta = (theta + new_theta)/2.;
                double new_x_forward = x + cos(move_theta)*this->v*this->del_t;
                double new_y_forward = y + cos(move_theta)*this->v*this->del_t;
                cout<<"New forward succ: "<<new_x_forward<<" "<<new_y_forward<<" "<<new_theta*180./M_PI<<endl;
                // printf("old theta:%.2f, new theta:%.2f\n", theta*180./M_PI, new_theta*180./M_PI);
                HybridAStarNode * new_node_forward = new HybridAStarNode(new_x_forward, new_y_forward, new_theta, this->x_grid_size, this->y_grid_size, this->theta_grid_size);

                successors.push_back(new_node_forward);


                double new_x_backward = x - cos(move_theta)*this->v*this->del_t;
                double new_y_backward = y - cos(move_theta)*this->v*this->del_t;
                // cout<<"New backward succ: "<<new_x_backward<<" "<<new_y_backward<<" "<<new_theta*180./M_PI<<endl;
                HybridAStarNode * new_node_backward = new HybridAStarNode(new_x_backward, new_y_backward, new_theta, this->x_grid_size, this->y_grid_size, this->theta_grid_size);                
                successors.push_back(new_node_backward);
            }
            cout<<endl;
            // exit(0);
            return successors;
        }

        vector<tuple<double, double, double>> find_plan(){
            clock_t begin_time = clock();
            // pop one element
            while(!this->open_nodes_queue.empty()){
                // cout<<"here"<<endl;
                // cout<<this->open_nodes_queue.size()<<endl;
                HybridAStarNode * cur_node = get_next_top();
                //cout<<"Working on "<<cur_node.x<<" "<<cur_node.y<<" "<<open_nodes_queue.size()<<endl;
                if (is_goal(*cur_node)){
                    // make the plan
                    cout<<"here2"<<endl;
                    this->plan = make_final_plan(cur_node);
                    path_size = this->plan.size();
                    // add other metrics here
                    float time_taken = float( clock () - begin_time ) /  CLOCKS_PER_SEC;
                    int states_expanded = this->open_nodes_map.size();
                    int closed_list_size = this->closed_list.size();
                    printf("Time Taken: %.3f sec, states_expanded: %d, closed_list_size: %d, plan_size: %d \n", time_taken, states_expanded,
                                                                                    closed_list_size, path_size);
                    return this->plan;
                }

                // insert in closed list
                this->closed_list[cur_node] = cur_node->g;

                // double cur_x = cur_node->x;
                // double cur_y = cur_node->y;
                // double cur_theta = cur_node->theta;

                vector<HybridAStarNode *> succ = get_successors(cur_node); // check if cur_node is in pointer format or not

                for (HybridAStarNode * new_n: succ)
                {
                    double new_x_discrete = new_n->x_discrete;
                    double new_y_discrete = new_n->y_discrete;
                    // double new_theta_discrete = new_n->theta_discrete;

                    // if (reward[newx][newy] <= this->collision_threshold){ // need to check how to convert from one frame to another
                    // if (this->reward[newx+newy] <= this->collision_threshold){ // fix this
                    if (new_x_discrete < 0 || new_x_discrete >= this->reward.size() ||  new_y_discrete < 0 || new_y_discrete >= this->reward.size()
                        || this->reward[new_x_discrete][new_y_discrete] >= this->collision_threshold) continue;

                    // HybridAStarNode * new_x = new HybridAStarNode(new_x, new_y, new_theta, this->x_grid_size, this->y_grid_size, this->theta_grid_size);

                    // is new_x in open, then use its g and update it, else g is inf
                    if (this->open_nodes_map.find(*new_n) == this->open_nodes_map.end()){
                        cout<<"new_n dist: "<<compute_dist(*cur_node, *new_n)<<endl;
                        new_n->g = cur_node->g + compute_dist(*cur_node, *new_n); //this->reward[new_n->x_discrete][new_n->y_discrete]; // reward is also discretized
                        new_n->h = compute_heuristic(*new_n); // check this thing
                        new_n->update_f();
                        this->open_nodes_queue.push(new_n);
                        // this->open_nodes_set.insert(*new_x);
                        this->open_nodes_map[*new_n] = new_n->g;

                        cur_node->children.push_back(new_n);
                        new_n->parent = cur_node;
                    }
                    else{
                        // already there
                        pair<HybridAStarNode, double> found_elem = *(this->open_nodes_map.find(*new_n));

                        if (found_elem.second > cur_node->g + this->reward[new_x_discrete][new_y_discrete]){
                            new_n->g = cur_node->g + compute_dist(*cur_node, *new_n); // need to change this //this->reward[new_n->x_discrete][new_n->y_discrete];
                            new_n->h = compute_heuristic(*new_n);
                            new_n->update_f();
                            this->open_nodes_queue.push(new_n);
                            this->open_nodes_map[found_elem.first] = new_n->g;
                            cur_node->children.push_back(new_n);
                            new_n->parent = cur_node;
                        }
                        else{
                            // cout<<"skipping this guy\n";
                        }
                    }
                    // need to check the following better
                    if ((this->closed_list).find(new_n) != (this->closed_list).end()){
                        cout<<"expaned but in closed\n";
                    }

                    // }
                }
            
            }
            return std::vector<tuple<double,double,double>>();
        }
};


PYBIND11_MODULE(hybrid_astar, m) {
    py::class_<HybridAStarNode>(m, "HybridAStarNode")
        .def(py::init<>())
        .def(py::init<double, double, double, double, double, double>())
        .def("update_f", &HybridAStarNode::update_f);

    py::class_<HybridAStar>(m, "HybridAStar")
        .def(py::init<const vector<vector<double>> &, double, double, double, double, double, double, double, double, double, double, double, double, double>())
        .def("make_final_plan", &HybridAStar::make_final_plan)
        .def("is_goal", &HybridAStar::is_goal)
        .def("compute_heuristic", &HybridAStar::compute_heuristic)
        .def("get_successors", &HybridAStar::get_successors)
        .def("find_plan", &HybridAStar::find_plan);
}