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


list<list<string>> get_permutation(vector<string> & symbols, int num_args){
    vector<string> symbol_vector;
    for (string s: symbols) symbol_vector.push_back(s);

    // create 0,1,2 vector
    vector<vector<int>> idxs;
    //for (int i=0; i<symbol_vector.size(); i++) idxs.push_back(i);

    int * arr = new int[num_args];
    comb(symbol_vector.size(), num_args, arr, num_args, &idxs);

    // check the idxs array

    list<list<string>> symbol_permutations;
    for (int i=0; i<idxs.size(); i++){
        list<string> symbol_permutation;
        for (int j=0; j<idxs[i].size(); j++){
            symbol_permutation.push_back(symbols[idxs[i][j]]);
        }
        symbol_permutations.push_back(symbol_permutation);
    }

    return symbol_permutations;
}

// given the environment, get the grounded list of actions, 
// maybe return as unordered set, add the necessary type casts,
// need to add a state
vector<GroundedAction> get_grounded_actions(unordered_set<string> symbols, 
        const unordered_set<Action, ActionHasher, ActionComparator> & actions,
        Env * env){

    vector<string> symbols_vector;
    for (string s: symbols) symbols_vector.push_back(s);

    vector<GroundedAction> grounded_actions;
    for (Action a: actions){ // check this way of initialization
        // get the number of symbols in this action
        int num_args = a.get_args().size(); // check if this is correct or not
        //list<list<string>> * set_of_symbols;
        list<list<string>> set_of_symbols;
        set_of_symbols = get_permutation(symbols_vector, num_args);
        // cout<<set_of_symbols.size()<<endl;
        for (list<string> x: set_of_symbols){
            // cout<<x.size()<<endl;
            grounded_actions.push_back(GroundedAction(a.get_name(), x, env));
        }
        // check the size of grounded_actions here
    }
    // check the size of grounded_actions here
    // cout<<grounded_actions.size()<<endl;
    return grounded_actions;
}

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

        friend ostream& operator<<(ostream& os, const Node& n)
        {
            os << endl;
            os << x << " " << y << " " << z << " " << x_round << " " << y_round << " " << ang_round << endl;
            return os;
        }



        bool operator==(const Node& rhs) const
        {
            if (this->conditions == rhs.conditions) return true;
            else return false;
        }
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

bool satisfies(GroundedAction & ga, Node * n){
    // loop through the preconditions and see it they exist in the node's conditions
    for (GroundedCondition g: ga.get_preconditions()){
        if (n->conditions.find(g) == n->conditions.end()) // check this
            return false;
    }
    return true;
}

// in the following function can see if want to get any heuristic
Node * update_state(Node * n, GroundedAction & ga){
    Node * new_node = new Node(n->conditions);

    // add and delete based on the effects
    for (GroundedCondition gc: ga.get_effects()){
        if (gc.get_truth()){
            // add to the state
            new_node->conditions.insert(gc); // check if this works or not
        }
    }
    for (GroundedCondition gc: ga.get_effects()){
        if (!gc.get_truth()){
            // add to the state
            gc.set_truth(true);
            new_node->conditions.erase(gc); // check if this works or not
        }
    }

    return new_node;
}

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

bool is_goal(unordered_set<GroundedCondition, GroundedConditionHasher, GroundedConditionComparator> & potential,
            const unordered_set<GroundedCondition, GroundedConditionHasher, GroundedConditionComparator> & goal){
    for (GroundedCondition gc: goal){
        if (potential.find(gc) == potential.end()) return false;
    }
    return true;
}
