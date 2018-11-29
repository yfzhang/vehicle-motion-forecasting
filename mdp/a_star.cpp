#include <algorithm>
// #include "planner_utils.cpp"
#include <time.h>
#include <queue>
#include <unordered_map>

#define NUMOFDIRS 8

using namespace std;

class NodeCompare
{
public:
    bool operator()(const AStarNode * p1, 
                    const AStarNode * p2) const
    {
        return p1->f >= p2->f; // check this
    }
};

class AStarNode {
    public:
        // check what all params needed or not
        double x, y;
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

        AStarNode(double _x, double _y){
            this->x = _x;
            this->y = _y;
        }

        friend ostream& operator<<(ostream& os, const AStarNode& n)
        {
            os << endl;
            os << x << " " << y << " " << z << " " << x_round << " " << y_round << " " << ang_round << endl;
            return os;
        }

        // bool operator==(const Node& rhs) const
        // {
        //     if (this->conditions == rhs.conditions) return true;
        //     else return false;
        // }
};

struct NodeComparator
{
    bool operator()(const AStarNode& lhs, const AStarNode& rhs) const
    {
        return lhs == rhs; // need to check this how it works
    }
};

struct NodeHasher2
{
    size_t operator()(const AStarNode n) const{
        return std::hash<double>()(n.x) ^ std::hash<double>()(n.y);
    }
};

struct NodeHasher
{
    size_t operator()(const AStarNode* n) const{
        return std::hash<double>()(n->x) ^ std::hash<double>()(n->y);
    }
};

class AStar{
    public:
        list<GroundedAction> plan;
        vector<GroundedAction> all_grounded_actions;
        int path_size;
        AStarNode * start;
        AStarNode * goal;
        double collision_threshold;

        priority_queue<AStarNode*, vector<AStarNode*>, NodeCompare> open_nodes_queue;// check the declaration
        unordered_map<AStarNode*, int, NodeHasher> closed_list;
        unordered_map<AStarNode, int, NodeHasher2> open_nodes_map;

        int dX[NUMOFDIRS] = {-1, -1, -1,  0,  0,  1, 1, 1};
        int dY[NUMOFDIRS] = {-1,  0,  1, -1,  1, -1, 0, 1};

        AStar(vector<double> reward, double start_x, double start_y, double goal_x, double goal_y, double _collision_threshold){
            // need to see how to interface with the reward and what inputs to take
            // initialize the vars here
            this->start = new AStarNode(start_x, start_y);
            this->goal = new AStarNode(goal_x, goal_y);
            this->collision_threshold = _collision_threshold;

            this->start->h = compute_heuristic(this->start);
            this->start->g = 0;
            this->start->update_f();

            this->open_nodes_queue.push(this->start);
            this->open_nodes_map[*(this->start)] = this->start->g; 
            // check this also, node comparison also have to see

            cout<<"Initializing complete\n";
        }

        double compute_heuristic(AStarNode & n){
            // take the euclidean distance for now
            // check the sqrt and otehr things
            return sqrt(sqr(n.x - this->goal.x) + sqr(n.y - this->goal.y));
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
            if (n.x == this->goal.x && n.y == this->goal.y) return true;
            else return false;
        }

        vector<pair<double, double>> make_final_plan(AStarNode * cur_node){
            vector<pair<double, double>> plan;
            while (cur_node->parent != NULL){
                // cout<<"here3"<<endl;
                // cout<<*(cur_node->ac)<<endl;
                // cout<<cur_node->ac<<endl;
                // plan.push_back(*(cur_node->ac));
                plan.push_back(make_pair(cur_node->x, cur_node->y));
                cur_node = cur_node->parent;
            }
            // plan.push_back(cur_node->ac);
            reverse(plan.begin(), plan.end());
            return plan;
        }

        void find_plan(){
            clock_t begin_time = clock();
            // pop one element
            while(!this->open_nodes_queue.empty()){
                //cout<<"here"<<endl;
                AStarNode * cur_node = get_next_top();
                //cout<<"Working on "<<cur_node.x<<" "<<cur_node.y<<" "<<open_nodes_queue.size()<<endl;
                if (is_goal(*cur_node)){
                    // make the plan
                    this->plan = make_plan(cur_node);
                    path_size = this->plan.size();
                    // add other metrics here
                    float time_taken = float( clock () - begin_time ) /  CLOCKS_PER_SEC;
                    int states_expanded = this->open_nodes_map.size();
                    int closed_list_size = this->closed_list.size();
                    printf("Time Taken: %.3f sec, states_expanded: %d, closed_list_size: %d, plan_size: %d \n", time_taken, states_expanded,
                                                                                    closed_list_size, path_size);

                    break;
                }

                // insert in closed list
                this->closed_list[cur_node] = cur_node->g;

                double cur_x = cur_node->x;
                double cur_y = cur_node->y;

                for(int dir = 0; dir < NUMOFDIRS; dir++)
                {
                    double newx = cur_x + dX[dir];
                    double newy = cur_y + dY[dir];

                    if (reward[newx][newy] <= this->collision_threshold){ // need to check how to convert from one frame to another

                        AStarNode * new_x = new AStarNode(newx, newy);

                        // is new_x in open, then use its g and update it, else g is inf
                        if (this->open_nodes_map.find(*new_x) == this->open_nodes_map.end()){
                            new_x->g = cur_node->g + 1;
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

                            if (found_elem.second > cur_node->g + 1){
                                new_x->g = cur_node->g + 1;
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

                    }
                }
            
            }
        }
};


PYBIND11_MODULE(a_star, m) {
    py::class_<AStar>(m, "AStar")
        .def(py::init<const vector<double> &, double, double, double, double, double>())
        .def("find_plan", &AStar::find_plan);
}