#include<bits/stdc++.h>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include <set>
#include <list>
#include <unordered_map>
#include "utils.cpp"

#define INF INT_MAX

using namespace std;


class Env{
    public:
        static double grid_res_x, grid_res_y;
        static vector<double> turn_angles;

        Env (double x, double y, vector<double> _turn_angles){
            grid_res_x = x;
            grid_res_y = y;

            for (double x1: _turn_angles) turn_angles.push_back(x1);

        }

        static pair<double, double> get_rounded_coords(Node & n){
            pair<double, double> rounded_coords;
            rounded_coords.first = n.x / grid_res_x; // check that this is float
            rounded_coords.second = n.y / grid_res_y;
            return rounded_coords;
        }

        static void set_loc_node(Node * node, Node & old_node){
            // do the bicycle calculation
            double old_x = old_node.x;
            double old_y = old_node.y;
            double old_ang = old_node.ang;

            for (double t_ang: turn_angles){
                // for each turn angle compute the new x, y and angle
            }
        }

        static vector<Node *> get_succ(Node & n, double dist){ 
        // this should call for all the resolution angles
        // TD whether n should be pointer or by reference
            vector<Node *> new_nodes;
            for (double ang: turn_angles){
                Node * new_node = new Node();
                // get the new node from the dist and the new angle
                set_loc_node(new_node, n);
                new_nodes.push_back(new_node);
            }
            return new_nodes;

        }
}

class Node{
    public:
        // check what all params needed or not
        double x, y, ang, x_round, y_round, ang_round;
        double f,g,h;
        vector<Node *> children;
        Node * parent;
        double trans_distance, trans_angle; // these might be used 
        // need to store the action also that got to this state

        Node(){
            this->parent = NULL;
            //ac = NULL;
            this->g = INF; //check this
            this->h = 0;
            this->f = this->g+this->h;
        }

        Node(double _x, double _y){
            this->x = _x;
            this->y = _y;
        }

        friend ostream& operator<<(ostream& os, const Node& n)
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
    bool operator()(const Node& lhs, const Node& rhs) const
    {
        return lhs == rhs;
    }
};

struct NodeHasher2
{
    size_t operator()(const Node n) const{
        vector<string> gc_vector;
        string temp = "";
        for (GroundedCondition gc: n.conditions){
            gc_vector.push_back(gc.toString());
        }
        sort(gc_vector.begin(), gc_vector.end());
        
        for (string s: gc_vector){
            temp += s + " ";
        }
        // cout<<*n<<endl;
        size_t val = hash<string>{}(temp);
        // cout<<val<<endl;
        return val;
    }
};

struct NodeHasher
{
    size_t operator()(const Node* n) const{
        vector<string> gc_vector;
        string temp = "";
        for (GroundedCondition gc: n->conditions){
            gc_vector.push_back(gc.toString());
        }
        sort(gc_vector.begin(), gc_vector.end());
        
        for (string s: gc_vector){
            temp += s + " ";
        }
        // cout<<*n<<endl;
        size_t val = hash<string>{}(temp);
        // cout<<val<<endl;
        return val;
    }
};


list<GroundedAction> make_plan(Node * cur_node){
    list<GroundedAction> plan;
    while (cur_node->parent != NULL){
        // cout<<"here3"<<endl;
        // cout<<*(cur_node->ac)<<endl;
        // cout<<cur_node->ac<<endl;
        // plan.push_back(*(cur_node->ac));
        plan.push_back(cur_node->ac);
        cur_node = cur_node->parent;
    }
    // plan.push_back(cur_node->ac);
    plan.reverse();
    return plan;
}
