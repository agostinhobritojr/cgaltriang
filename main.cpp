#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/legacy/legacy.hpp"

#include <iostream>
#include <ctype.h>
#include <string>
#include <cstdio>

/* cgal stuff
 */
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <vector>

typedef CGAL::Exact_predicates_inexact_constructions_kernel               Kernel;
typedef CGAL::Triangulation_vertex_base_with_info_2<unsigned int, Kernel> Vb;
typedef CGAL::Triangulation_data_structure_2<Vb>                          Tds;
typedef CGAL::Delaunay_triangulation_2<Kernel, Tds>                       Delaunay;
typedef Kernel::Point_2                                                   Point2;
/*
 * cgal stuff
*/

using namespace cv;
using namespace std;

TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03);
Size subPixWinSize(50,50), winSize(100,100);
vector<Point2f> points[2];
Mat gray[2];
const int MAX_COUNT = 9;


std::vector< std::pair<Point2,unsigned> > cgalPoints;


static void draw_subdiv( Mat& img, Subdiv2D& subdiv, Scalar delaunay_color )
{
#if 1
  vector<Vec6f> triangleList;
  subdiv.getTriangleList(triangleList);
  vector<Point> pt(3);

  for( size_t i = 0; i < triangleList.size(); i++ )
  {
    Vec6f t = triangleList[i];
    pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
    pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
    pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
    line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
    line(img, pt[1], pt[2], delaunay_color, 1, CV_AA, 0);
    line(img, pt[2], pt[0], delaunay_color, 1, CV_AA, 0);
    Scalar color;
    color[0] = 50 + 10*i;
    color[1] = 20*i;
    color[2] = 100 + 5*i;
    fillConvexPoly(img, pt, color, 8, 0);

  }
#else
  vector<Vec4f> edgeList;
  subdiv.getEdgeList(edgeList);
  for( size_t i = 0; i < edgeList.size(); i++ )
  {
    Vec4f e = edgeList[i];
    Point pt0 = Point(cvRound(e[0]), cvRound(e[1]));
    Point pt1 = Point(cvRound(e[2]), cvRound(e[3]));
    line(img, pt0, pt1, delaunay_color, 1, CV_AA, 0);
  }
#endif
}

bool point_inside_trigon(Point s, Point a, Point b, Point c)
{
  int as_x = s.x-a.x;
  int as_y = s.y-a.y;

  bool s_ab = (b.x-a.x)*as_y-(b.y-a.y)*as_x > 0;

  if((c.x-a.x)*as_y-(c.y-a.y)*as_x > 0 == s_ab) return false;

  if((c.x-b.x)*(s.y-b.y)-(c.y-b.y)*(s.x-b.x) > 0 != s_ab) return false;

  return true;
}
Point2f warp_point2(Point2f p,double** warp){
  Point2f result;
  result.x=warp[0][0]*p.x+warp[0][1]*p.y+warp[0][2];
  result.y=warp[1][0]*p.x+warp[1][1]*p.y+warp[1][2];
  //    cout<< "Matriz\n";
  //    cout<< warp[0][0]<<" "<<warp[0][1]<<" "<<warp[0][2]<<"\n";
  //    cout<< warp[1][0]<<" "<<warp[1][1]<<" "<<warp[1][2]<<"\n";
  //    cout<<"Ponto escolhido\n";
  //    cout<<p.x<<" "<<p.y<<"\n";
  //    cout<<"Ponto resultante\n";
  //    cout<<result.x<<" "<<result.y<<"\n";
  return result;
}

void my_affine(double** warp,Point2f* src, Point2f* dst){
  double x1 =src[0].x,x2=src[1].x,x3=src[2].x;
  double y1 =src[0].y,y2=src[1].y,y3=src[2].y;
  double xt1 =dst[0].x,xt2=dst[1].x,xt3=dst[2].x;
  double yt1 =dst[0].y,yt2=dst[1].y,yt3=dst[2].y;

  warp[0][0]=(((xt3-xt1)*(y2-y1))-((y3-y1)*(xt2-xt1)))/(((x3-x1)*(y2-y1))-((x2-x1)*(y3-y1)));
  warp[0][1]=((xt2-xt1)-warp[0][0]*(x2-x1))/(y2-y1);
  warp[0][2]=xt1 - warp[0][0]*x1-warp[0][1]*y1;
  warp[1][0]=(((yt3-yt1)*(y2-y1))-((y3-y1)*(yt2-yt1)))/(((x3-x1)*(y2-y1))-((x2-x1)*(y3-y1)));
  warp[1][1]=((yt2-yt1)-warp[1][0]*(x2-x1))/(y2-y1);
  warp[1][2]=yt1 - warp[1][0]*x1-warp[1][1]*y1;

  //    cout<< "Matriz\n";
  //    cout<< warp[0][0]<<" "<<warp[0][1]<<" "<<warp[0][2]<<"\n";
  //    cout<< warp[1][0]<<" "<<warp[1][1]<<" "<<warp[1][2]<<"\n";
  //    cout<< "Source\n";
  //    cout<< src[0].x<<" "<<src[0].y<<"\n";
  //    cout<< src[1].x<<" "<<src[1].y<<"\n";
  //    cout<< src[2].x<<" "<<src[2].y<<"\n";
  //    cout<< "Destino\n";
  //    cout<< dst[0].x<<" "<<dst[0].y<<"\n";
  //    cout<< dst[1].x<<" "<<dst[1].y<<"\n";
  //    cout<< dst[2].x<<" "<<dst[2].y<<"\n";


}

// Pega uma imagem no Endereço file
// Calcula os pontos bons
// Armazena na variável points[id]
void calc(const string &file,int id){
  Mat image;
  image = imread(file);
  cvtColor(image, gray[id], CV_BGR2GRAY);
  goodFeaturesToTrack(gray[id], points[id], MAX_COUNT, 0.001, 10, Mat(), 3, 0, 0.0004);
  if( !points[id].empty() )
  {
    size_t i, k;
    for( i = k = 0; i < points[id].size(); i++ )
    {
      circle( image, points[id][i], 3, Scalar(0,255,0), -1, 8);
    }
  }

  char* tela= new char(40);
  std::sprintf(tela,"%d",id);
  imshow(tela, image);

}

void draw_points(Mat frame, vector<Point2f> pontosfinais,vector<uchar> status){
  for (unsigned int i = 0; i < pontosfinais.size(); i++) {
    if (status[i] == 0) {
      circle(frame, pontosfinais[i], 3, Scalar(0,0,255), -1, 8);
      //                    drawPixel(pontosfinais[i], frame, 2, Scalar(0, 0, 255));
      continue;
    }
    circle(frame, points[0][i], 3, Scalar(255,0,0), -1, 8);

    circle(frame, pontosfinais[i], 3, Scalar(0,255,0), -1, 8);
    //                drawPixel(pontosfinais[i], frame, 2, Scalar(0,255,0));
    line(frame, points[0][i], pontosfinais[i], Scalar(255, 0, 0));

  }
}
int found_triangle(int i,int j,vector<Vec6f> triangleListIntermediario,Mat img5){
  Point2f interTri[3];
  for (int k1=0; k1<triangleListIntermediario.size(); k1++) {
    Vec6f interT = triangleListIntermediario[k1];
    interTri[0] = Point(interT[0],interT[1]);
    interTri[1] = Point(interT[2],interT[3]);
    interTri[2] = Point(interT[4],interT[5]);
    //                cout<<interTri[0]<<"\n";
    //                cout<<interTri[1]<<"\n";
    //                cout<<interTri[2]<<"\n";
    if(point_inside_trigon(Point(j,i), interTri[0], interTri[1],interTri[2])){
      circle(img5, interTri[0], 3, Scalar(0,0,255), -1, 8);
      circle(img5, interTri[1], 3, Scalar(0,0,255), -1, 8);
      circle(img5, interTri[2], 3, Scalar(0,0,255), -1, 8);
      return k1;
    }
  }
  return 0;
}

int main( int argc, char** argv ){



  string adress_src ="/home/ambj/workspace/cgaltriang/Etapa1.jpg";
  string adress_dst ="/home/ambj/workspace/cgaltriang/Etapa3.jpg";
  Scalar active_facet_color(0, 255, 0), delaunay_color(0,0,0);
  string win = "Delaunay Demo";
  string win2 = "Delaunay Demo 2";
  Mat ImageDst = imread(adress_dst);
  Mat ImageSrc = imread(adress_src);
  int width = ImageDst.cols;
  int height = ImageDst.rows;
  int parametroSrc=0,parametroDst=0;


  cout<<"width"<<width<<"\n";
  cout<<"height"<<height<<"\n";
  Rect rect(0, 0, width, height);

  calc(adress_src,0);
  cvtColor(imread(adress_dst), gray[1], CV_BGR2GRAY);


  vector<uchar> status;
  vector<float> err;
  vector<Point2f> pontosfinais;

  calcOpticalFlowPyrLK(gray[0], gray[1], points[0], pontosfinais, status, err, winSize,3, termcrit, 0, 0.001);
  draw_points(ImageDst,pontosfinais,status);


  Mat img,img2,img3,img4,img5;
  img  = imread(adress_dst);
  img2 = imread(adress_dst);
  img3 = imread(adress_src);
  img4 = imread(adress_src);
  img5 = imread(adress_dst);

  imshow(win, img);

  Subdiv2D subdivDst(rect);
  subdivDst.insert(Point(0,0));
  subdivDst.insert(Point2f(0,height-1));
  subdivDst.insert(Point2f(width-1,0));
  subdivDst.insert(Point(width-1,height-1));

  for( int i = 0; i < pontosfinais.size(); i++ ){

    subdivDst.insert(pontosfinais[i]);

  }

  cgalPoints.push_back( std::make_pair( Point2(0,0), 0 ) );
  cgalPoints.push_back( std::make_pair( Point2(0,height-1), 1 ) );
  cgalPoints.push_back( std::make_pair( Point2(width-1,0), 2 ) );
  cgalPoints.push_back( std::make_pair( Point2(width-1,height-1), 3 ) );

  Point aux;
  for( int i = 0; i < points[0].size(); i++ ){
    aux = points[0][i];
    cgalPoints.push_back( std::make_pair( Point2(aux.x,aux.y), i+4 ) );
//    cout << "aux = " << aux.x << " " << aux.y << endl;

  }
  Delaunay triangulation;
  triangulation.insert(cgalPoints.begin(),cgalPoints.end());

  for(Delaunay::Finite_faces_iterator fit = triangulation.finite_faces_begin();
      fit != triangulation.finite_faces_end(); ++fit) {

    Delaunay::Face_handle face = fit;
    std::cout << "Triangle:\t" << triangulation.triangle(face) << std::endl;
  //  std::cout << "Vertex 0:\t" << triangulation.triangle(face)[0] << std::endl;
 //   std::cout << "Vertex 1:\t" << triangulation.triangle(face)[1] << std::endl;
    std::cout << "Vertex 0:\t" << face->vertex(0)->info() << std::endl;
    std::cout << "Vertex 1:\t" << face->vertex(1)->info() << std::endl;
    std::cout << "Vertex 2:\t" << face->vertex(2)->info() << std::endl;
  }



  Subdiv2D subdivSrc(rect);
  subdivSrc.insert(Point(0,0));
  subdivSrc.insert(Point2f(0,height-1));
  subdivSrc.insert(Point2f(width-1,0));
  subdivSrc.insert(Point(width-1,height-1));

  for( int i = 0; i < points[0].size(); i++ ){

    subdivSrc.insert(points[0][i]);
    cout<<"ponto "<<points[0][i]<<"\n";
  }
  for( int i = 0; i < points[0].size()+4; i++ ){

    cout<<"vertex "<<subdivSrc.getVertex(i)<<"\n";
  }

  draw_subdiv(img3, subdivSrc, delaunay_color );
  draw_subdiv(img2, subdivDst, delaunay_color );

  imshow( win, img3 );
  imshow( win2, img2 );

  Point2f srcTri[3];
  Point2f interTri[3];
  Point2f dstTri[3];

  vector<Vec6f> triangleListSrc;
  vector<Vec6f> triangleListDst;

  subdivSrc.getTriangleList(triangleListSrc);
  subdivDst.getTriangleList(triangleListDst);

  for (int i=0; i<triangleListSrc.size(); i++) {
    Vec6f interT = triangleListSrc[i];
    interTri[0] = Point(interT[0], interT[1]);
    interTri[1] = Point(interT[2], interT[3]);
    interTri[2] = Point(interT[4], interT[5]);
    circle(img5, interTri[0], 3, Scalar(0,0,255), -1, 8);
    circle(img5, interTri[1], 3, Scalar(0,0,255), -1, 8);
    circle(img5, interTri[2], 3, Scalar(0,0,255), -1, 8);
    cout<<"triangulo Src "<<i<<"\n";
    cout<<"ponto 1"<<interTri[0]<<"\n";
    cout<<"ponto 2"<<interTri[1]<<"\n";
    cout<<"ponto 3"<<interTri[2]<<"\n";
  }

  vector<Vec6f> triangleListIntermediario = triangleListDst;



  double*** warp_mat_vetor = new double**[triangleListSrc.size()];
  for (int i=0; i<triangleListDst.size(); i++) {
    warp_mat_vetor[i]= new double*[2];
    warp_mat_vetor[i][0]=new double[3];
    warp_mat_vetor[i][1]=new double[3];
  }
  double*** warp_mat_vetor_inv = new double**[triangleListDst.size()];
  for (int i=0; i<triangleListDst.size(); i++) {
    warp_mat_vetor_inv[i]= new double*[2];
    warp_mat_vetor_inv[i][0]=new double[3];
    warp_mat_vetor_inv[i][1]=new double[3];
  }


  for( size_t i = 0; i < triangleListDst.size(); i++ )
  {
    Vec6f t1 = triangleListSrc[i];
    srcTri[0] = Point(cvRound(t1[0]), cvRound(t1[1]));
    srcTri[1] = Point(cvRound(t1[2]), cvRound(t1[3]));
    srcTri[2] = Point(cvRound(t1[4]), cvRound(t1[5]));
    Vec6f t2 = triangleListDst[i];
    dstTri[0] = Point(cvRound(t2[0]), cvRound(t2[1]));
    dstTri[1] = Point(cvRound(t2[2]), cvRound(t2[3]));
    dstTri[2] = Point(cvRound(t2[4]), cvRound(t2[5]));
    /// Get the Affine Transform
    //        line(ImageDst, srcTri[0], srcTri[1], delaunay_color, 1, CV_AA, 0);
    //        line(ImageDst, srcTri[1], srcTri[2], delaunay_color, 1, CV_AA, 0);
    //        line(ImageDst, srcTri[2], srcTri[0], delaunay_color, 1, CV_AA, 0);
    //        line(ImageDst, dstTri[0], dstTri[1], delaunay_color, 1, CV_AA, 0);
    //        line(ImageDst, dstTri[1], dstTri[2], delaunay_color, 1, CV_AA, 0);
    //        line(ImageDst, dstTri[2], dstTri[0], delaunay_color, 1, CV_AA, 0);
    triangleListIntermediario[i]= Vec6f((t1[0]+t2[0])/2,(t1[1]+t2[1])/2, (t1[2]+t2[2])/2, (t1[3]+t2[3])/2, (t1[4]+t2[4])/2, (t1[5]+t2[5])/2);
    Vec6f interT = triangleListIntermediario[i];
    interTri[0] = Point(cvRound(interT[0]), cvRound(interT[1]));
    interTri[1] = Point(cvRound(interT[2]), cvRound(interT[3]));
    interTri[2] = Point(cvRound(interT[4]), cvRound(interT[5]));
    circle(img4, interTri[0], 3, Scalar(0,255,0), -1, 8);
    circle(img4, interTri[1], 3, Scalar(0,255,0), -1, 8);
    circle(img4, interTri[2], 3, Scalar(0,255,0), -1, 8);
    circle(img4, srcTri[0], 3, Scalar(255,0,0), -1, 8);
    circle(img4, srcTri[1], 3, Scalar(255,0,0), -1, 8);
    circle(img4, srcTri[2], 3, Scalar(255,0,0), -1, 8);
    circle(img4, dstTri[0], 3, Scalar(0,0,255), -1, 8);
    circle(img4, dstTri[1], 3, Scalar(0,0,255), -1, 8);
    circle(img4, dstTri[2], 3, Scalar(0,0,255), -1, 8);


    my_affine(warp_mat_vetor[i], interTri, dstTri);
    my_affine(warp_mat_vetor_inv[i], interTri, srcTri);
  }

  //confirir triangulos intermediários
  //
  for (int i=0; i<triangleListSrc.size(); i++) {
    Vec6f interT = triangleListSrc[i];
    interTri[0] = Point(cvRound(interT[0]), cvRound(interT[1]));
    interTri[1] = Point(cvRound(interT[2]), cvRound(interT[3]));
    interTri[2] = Point(cvRound(interT[4]), cvRound(interT[5]));
    circle(img5, interTri[0], 3, Scalar(0,0,255), -1, 8);
    circle(img5, interTri[1], 3, Scalar(0,0,255), -1, 8);
    circle(img5, interTri[2], 3, Scalar(0,0,255), -1, 8);
    cout<<"triangulo Intermediario "<<i<<"\n";
    cout<<"ponto 1"<<interTri[0]<<"\n";
    cout<<"ponto 2"<<interTri[1]<<"\n";
    cout<<"ponto 3"<<interTri[2]<<"\n";
  }


  for (int i=0; i<height; i++) {
    for (int j=0; j<width; j++) {
      found_triangle(i,j,triangleListIntermediario,img5);


      //                if(PointInTriangle(Point(j,i), Point(200, 200), Point(400, 200),Point(200, 100))){
      //                    circle(img5, Point(200,200), 3, Scalar(0,255,255), -1, 8);
      //                    circle(img5, Point(400,200), 3, Scalar(0,255,255), -1, 8);
      //                    circle(img5, Point(200,100), 3, Scalar(0,255,255), -1, 8);
      //                    circle(img5, Point(j,i), 3, Scalar(0,0,255), -1, 8);
      //                }



      Point posicaoDst = warp_point2(Point2f(i,j),warp_mat_vetor[parametroDst]);
      Point posicaoSrc = warp_point2(Point2f(i,j),warp_mat_vetor_inv[parametroSrc]);
      img4.at<Vec3b>(i,j)[0]= (ImageDst.at<Vec3b>(posicaoDst.x,posicaoDst.y)[0] + ImageSrc.at<Vec3b>(posicaoSrc.x,posicaoSrc.y)[0])/2;
      img4.at<Vec3b>(i,j)[1]= (ImageDst.at<Vec3b>(posicaoDst.x,posicaoDst.y)[1] + ImageSrc.at<Vec3b>(posicaoSrc.x,posicaoSrc.y)[1])/2;
      img4.at<Vec3b>(i,j)[2]= (ImageDst.at<Vec3b>(posicaoDst.x,posicaoDst.y)[2] + ImageSrc.at<Vec3b>(posicaoSrc.x,posicaoSrc.y)[2])/2;



    }
  }

  //        cvShowImage("teste_warp", dst);
  //        imshow("testatttt", mat_img);
  imshow("4", img4);
  imshow("5", img5);
  waitKey();


  return 0;
}


