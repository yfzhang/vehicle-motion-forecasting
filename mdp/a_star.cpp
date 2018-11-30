#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
// #include "planner_utils.cpp"
#include <time.h>
#include <queue>
#include <unordered_map>
#include<bits/stdc++.h>

#define NUMOFDIRS 8
#define INF INT_MAX

using namespace std;

namespace py = pybind11;

class AStarNode {
    public:
        // check what all params needed or not
        int x, y;
        double f,g,h;
        vector<AStarNode *> children;
        AStarNode * parent;
        // need to store the action also that got to this state

        AStarNode(){
            this->parent = NULL;
            //ac = NULL;
            this->g = INF; //check this
            this->h = 0;
            this->f = this->g+this->h;
        }

        AStarNode(int _x, int _y) : AStarNode(){
            this->x = _x;
            this->y = _y;
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

        bool operator==(const AStarNode& rhs) const
        {
            if (this->x == rhs.x && this->y == rhs.y) return true;
            else return false;
        }
};

class NodeCompare
{
public:
    bool operator()(AStarNode * p1, 
                    AStarNode * p2)
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
    size_t operator()(const AStarNode n) const{
        // return std::hash<int>()(1000* n.x + n.y) ^ std::hash<int>()(n.y); // check if this is symmetric for x and y
        return std::hash<int>()((1000*n.x) + n.y); // check if this is symmetric for x and y
    }
};

struct NodeHasher
{
    size_t operator()(const AStarNode* n) const{
        // return std::hash<int>()(n->x) ^ std::hash<int>()(n->y);
        return std::hash<int>()((1000*n->x) + n->y);
    }
};

class AStar{
    public:
        int path_size;
        AStarNode * start;
        AStarNode * goal;
        double collision_threshold;

        vector<vector<double>> reward;
        vector<pair<int, int>> plan;

        priority_queue<AStarNode*, vector<AStarNode*>, NodeCompare> open_nodes_queue;// check the declaration
        unordered_map<AStarNode*, double, NodeHasher> closed_list;
        unordered_map<AStarNode, double, NodeHasher2> open_nodes_map;

        int dX[NUMOFDIRS] = {-1, -1, -1,  0,  0,  1, 1, 1};
        int dY[NUMOFDIRS] = {-1,  0,  1, -1,  1, -1, 0, 1};

        AStar(vector<vector<double>> _reward, int start_x, int start_y, int goal_x, int goal_y, double _collision_threshold){
            // need to see how to interface with the reward and what inputs to take
            // initialize the vars here
            cout<<_reward.size()<<" "<<_reward[0].size()<<endl;
            this->start = new AStarNode(start_x, start_y);
            this->goal = new AStarNode(goal_x, goal_y);
            this->collision_threshold = _collision_threshold;

            this->start->h = compute_heuristic(*(this->start));
            this->start->g = 0;
            this->start->update_f();

            this->open_nodes_queue.push(this->start); // can print values here and see
            this->open_nodes_map[*(this->start)] = this->start->g; 
            // check this also, node comparison also have to see

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

        double compute_heuristic(AStarNode & n){
            // take the euclidean distance for now
            // check the sqrt and otehr things
            return 0.;
            // return sqrt(sqr(n.x - this->goal.x) + sqr(n.y - this->goal.y));
        }

        AStarNode * get_next_top(){
            AStarNode * cur_node = this->open_nodes_queue.top();
            while (this->open_nodes_map.find(*cur_node) != (this->open_nodes_map).end() &&
                    cur_node->g > (this->open_nodes_map.find(*cur_node))->second)
            {
                this->open_nodes_queue.pop();
                cur_node = this->open_nodes_queue.top();
            }
            this->open_nodes_queue.pop();
            return cur_node;
        }

        bool is_goal(AStarNode & n){
            // assuming int vals in n and goal
            if (n.x == this->goal->x && n.y == this->goal->y) return true;
            else return false;
        }

        vector<pair<int, int>> make_final_plan(AStarNode * cur_node){
            vector<pair<int, int>> plan;
            while (cur_node->parent != NULL){
                cout<<"here3"<<endl;
                // cout<<*(cur_node->ac)<<endl;
                // cout<<cur_node->ac<<endl;
                // plan.push_back(*(cur_node->ac));
                plan.push_back(make_pair(cur_node->x, cur_node->y));
                cur_node = cur_node->parent;
            }
            plan.push_back(make_pair(cur_node->x, cur_node->y));
            reverse(plan.begin(), plan.end()); // check this also
            return plan;
        }

        vector<pair<int,int>> find_plan(){
            clock_t begin_time = clock();
            // pop one element
            while(!this->open_nodes_queue.empty()){
                // cout<<"here"<<endl;
                AStarNode * cur_node = get_next_top();
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

                int cur_x = cur_node->x;
                int cur_y = cur_node->y;
                // cout<<cur_x<<" "<<cur_y<<endl;
                for(int dir = 0; dir < NUMOFDIRS; dir++)
                {
                    int newx = cur_x + dX[dir];
                    int newy = cur_y + dY[dir];

                    // if (reward[newx][newy] <= this->collision_threshold){ // need to check how to convert from one frame to another
                    // if (this->reward[newx+newy] <= this->collision_threshold){ // fix this
                    if (newx < 0 || newx >= this->reward.size() ||  newy < 0 || newy >= this->reward.size()) continue;

                    AStarNode * new_x = new AStarNode(newx, newy);

                    // is new_x in open, then use its g and update it, else g is inf
                    if (this->open_nodes_map.find(*new_x) == this->open_nodes_map.end()){
                        new_x->g = cur_node->g + this->reward[newx][newy];
                        new_x->h = compute_heuristic(*new_x); // check this thing
                        new_x->update_f();
                        this->open_nodes_queue.push(new_x);
                        // this->open_nodes_set.insert(*new_x);
                        this->open_nodes_map[*new_x] = new_x->g;

                        cur_node->children.push_back(new_x);
                        new_x->parent = cur_node;
                    }
                    else{
                        // already there
                        pair<AStarNode, double> found_elem = *(this->open_nodes_map.find(*new_x));

                        if (found_elem.second > cur_node->g + this->reward[newx][newy]){
                            new_x->g = cur_node->g + this->reward[newx][newy];
                            new_x->h = compute_heuristic(*new_x);
                            new_x->update_f();
                            this->open_nodes_queue.push(new_x);
                            this->open_nodes_map[found_elem.first] = new_x->g;
                            cur_node->children.push_back(new_x);
                            new_x->parent = cur_node;
                        }
                    }

                    if ((this->closed_list).find(new_x) != (this->closed_list).end()){
                        cout<<"expaned but in closed\n";
                    }

                    // }
                }
            
            }
            return std::vector<pair<int,int>>();
        }
};


PYBIND11_MODULE(a_star, m) {
    py::class_<AStarNode>(m, "AStarNode")
        .def(py::init<>())
        .def(py::init<int, int>())
        .def("update_f", &AStarNode::update_f);

    py::class_<AStar>(m, "AStar")
        .def(py::init<const vector<vector<double>> &, int, int, int, int, double>())
        .def("make_final_plan", &AStar::make_final_plan)
        .def("is_goal", &AStar::is_goal)
        .def("compute_heuristic", &AStar::compute_heuristic)
        .def("find_plan", &AStar::find_plan);
}