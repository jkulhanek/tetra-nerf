#include "triangulation.h"

#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/compute_average_spacing.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils/exception.h"

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_with_info_3<unsigned int, K> Vb;
typedef CGAL::Triangulation_data_structure_3<Vb> Tds;
typedef CGAL::Delaunay_triangulation_3<K, Tds> Triangulation;

typedef K::FT FT;
typedef CGAL::Parallel_if_available_tag Concurrency_tag;

typedef Triangulation::Cell_handle Cell_handle;
typedef Triangulation::Vertex_handle Vertex_handle;
typedef Triangulation::Locate_type Locate_type;
typedef Triangulation::Point Point;

std::vector<uint4> triangulate(size_t num_points, float3 *points) {
    std::vector<std::pair<Point, unsigned int>> L(num_points);
    for (size_t i = 0; i < num_points; i++) {
        const auto p = Point(
            points[i].x,
            points[i].y,
            points[i].z);
        L[i] = std::make_pair(p, i);
    }

    Triangulation T(L.begin(), L.end());
    if (!T.is_valid()) {
        throw Exception("Triangulation failed");
    }

    // Export
    std::vector<uint4> cells(T.number_of_finite_cells());
    unsigned int *cells_uint = reinterpret_cast<unsigned int*>(cells.data());

    size_t i = 0;
    for (auto cell : T.finite_cell_handles()) {
        for (int j = 0; j < 4; ++j) {
            cells_uint[i * 4 + j] = cell->vertex(j)->info();
        }
        i++;
    }
    return cells;
}


float find_average_spacing(size_t num_points, float3 *points) {
    const unsigned int nb_neighbors = 6; // 1 ring
    std::vector<Point> L(num_points);
    for (size_t i = 0; i < num_points; i++) {
        const auto p = Point(
            points[i].x,
            points[i].y,
            points[i].z);
        L[i] = p;
    }

    FT average_spacing = CGAL::compute_average_spacing<Concurrency_tag>(L, nb_neighbors);
    return average_spacing;
}