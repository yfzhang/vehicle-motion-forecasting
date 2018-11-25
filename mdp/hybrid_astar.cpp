#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// #include <vector>
using namespace std;

namespace py = pybind11;

class Pet {
    public:
        std::string name;
        Pet(const std::string &name) : name(name) { }
        void setName(const std::string &name_) { name = name_; }
        const std::string &getName() const { return name; }

        vector<int> return_ints(){
            vector<int> int_arr;
            int_arr.push_back(1);
            int_arr.push_back(2);
            return int_arr;
        }
};

PYBIND11_MODULE(hybrid_astar, m) {
    py::class_<Pet>(m, "Pet")
        .def(py::init<const std::string &>())
        .def("setName", &Pet::setName)
        .def("getName", &Pet::getName)
        .def("return_ints", &Pet::return_ints) \
        ;
}