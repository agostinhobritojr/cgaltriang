#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <vector>

typedef CGAL::Exact_predicates_inexact_constructions_kernel               Kernel;
typedef CGAL::Triangulation_vertex_base_with_info_2<unsigned int, Kernel> Vb;
typedef CGAL::Triangulation_data_structure_2<Vb>                          Tds;
typedef CGAL::Delaunay_triangulation_2<Kernel, Tds>                       Delaunay;
typedef Kernel::Point_2                                                   Point;

int main() {

  std::vector< std::pair<Point,unsigned> > points;
  points.push_back( std::make_pair( Point(0,0), 0 ) );
  points.push_back( std::make_pair( Point(100,0), 1 ) );
  points.push_back( std::make_pair( Point(100,200), 2 ) );
  points.push_back( std::make_pair( Point(0,200), 3 ) );

  Delaunay triangulation;
  triangulation.insert(points.begin(),points.end());

  for(Delaunay::Finite_faces_iterator fit = triangulation.finite_faces_begin();
      fit != triangulation.finite_faces_end(); ++fit) {

    Delaunay::Face_handle face = fit;
    std::cout << "Triangle:\t" << triangulation.triangle(face) << std::endl;
    std::cout << "Vertex 0:\t" << triangulation.triangle(face)[0] << std::endl;
    std::cout << "Vertex 1:\t" << triangulation.triangle(face)[1] << std::endl;
    std::cout << "Vertex 0:\t" << face->vertex(0)->info() << std::endl;
  }
}












