/*****************************************************************************
 * Grupo: Luiz Gustavo Braganca dos Santos                                   *
 *        Maria Clara Dias                                                   *
 *        Pedro Augusto Prosdocimi                                           *
 * Materia: Computacao Paralela                                              *
 * Professor: Luis Fabricio Wanderley Goes                                   *
 * _________________________________________________________________________ *
 *                        MAQUINA LOCAL                                      *
 *                                                                           *
 *                      ---SEQUENCIAL---                                     *
 *                                                                           *
 * Rodando o comando: $ g++ kmeans.cpp -o main                               *
 *                    $ time ./kmeans < /datasets/pub.in                     *
 * real	0m29.518s
 * user	0m29.111s
 * sys	0m0.019s                                                             *
 * ------------------------------------------------------------------------- * 
 *                    ---PARALELO 1 THREAD---                                *
 *                                                                           *
 * Rodando o comando: $ g++ kmeans.cpp -o main -fopenmp                      *
 *                    $ time ./kmeans < /datasets/pub.in                     *
 *                                                                           *
 * real	0m25.807s                                                            *
 * user	1m31.609s                                                            *
 * sys	0m3.120s                                                             *
 * ------------------------------------------------------------------------- * 
 *                   ---PARALELO 2 THREADS---                                *
 *                                                                           *
 * Rodando o comando: $ g++ kmeans.cpp -o main -fopenmp                      *
 *                    $ time ./kmeans < /datasets/pub.in                     *
 *                                                                           *
 * real	0m20.995s                                                            *
 * user	0m38.681s                                                            *
 * sys	0m2.623s                                                             *
 * ------------------------------------------------------------------------- * 
 *                   ---PARALELO 4 THREADS---                                *
 *                                                                           *
 * Rodando o comando: $ g++ kmeans.cpp -o main -fopenmp                      *
 *                    $ time ./kmeans < /datasets/pub.in                     *
 *                                                                           *
 * real    0m14.242s                                                         *
 * user    0m59.432s                                                         *
 * sys     0m1.980s                                                          *
 * ------------------------------------------------------------------------- * 
 *                   ---PARALELO 8 THREADS---                                *
 *                                                                           *
 * Rodando o comando: $ g++ kmeans.cpp -o main -fopenmp                      *
 *                    $ time ./kmeans < /datasets/pub.in                     *
 *                                                                           *
 * real	0m21.881s                                                            *
 * user	0m40.146s                                                            *
 * sys	0m3.948s                                                             *
 * _________________________________________________________________________ *
 *                        ---SPEEDUP---                                      *
 *                                                                           *
 * 1 THREAD:  (29,518/25,807) = 1,144                                        *
 * 2 THREADS: (29,518/20,995) = 1,406                                        *
 * 4 THREADS: (29,518/14,242) = 2,072                                        *
 * 8 THREADS: (29,518/21,881) = 1,349                                        *
 * ========================================================================= *
 *                      SERVIDOR PARCODE                                     *
 *                                                                           *
 *                      ---SEQUENCIAL---                                     *
 *                                                                           *
 * Rodando o comando: $ g++ kmeans.cpp -o main                               *
 *                    $ time ./kmeans < /datasets/pub.in                     *
 *                                                                           *
 * real	0m46.779s                                                            *
 * user	0m46.713s                                                            *
 * sys	0m0.028s                                                             *
 * ------------------------------------------------------------------------- * 
 *                    ---PARALELO 1 THREAD---                                *
 *                                                                           *
 * Rodando o comando: $ g++ kmeans.cpp -o main -fopenmp                      *
 *                    $ time ./kmeans < /datasets/pub.in                     *
 *                                                                           *
 * real	m.s                                                                  *
 * user	m.s                                                                  *
 * sys	m.s                                                                  *
 * ------------------------------------------------------------------------- * 
 *                   ---PARALELO 2 THREADS---                                *
 *                                                                           *
 * Rodando o comando: $ g++ kmeans.cpp -o main -fopenmp                      *
 *                    $ time ./kmeans < /datasets/pub.in                     *
 *                                                                           *
 * real	m.s                                                                  *
 * user	m.s                                                                  *
 * sys	m.s                                                                  *
 * ------------------------------------------------------------------------- * 
 *                   ---PARALELO 4 THREADS---                                *
 *                                                                           *
 * Rodando o comando: $ g++ kmeans.cpp -o main -fopenmp                      *
 *                    $ time ./kmeans < /datasets/pub.in                     *
 *                                                                           *
 * real	m.s                                                                  *
 * user	m.s                                                                  *
 * sys	m.s                                                                  *
 * ------------------------------------------------------------------------- * 
 *                   ---PARALELO 8 THREADS---                                *
 *                                                                           *
 * Rodando o comando: $ g++ kmeans.cpp -o main -fopenmp                      *
 *                    $ time ./kmeans < /datasets/pub.in                     *
 *                                                                           *
 * real	m.s                                                                  *
 * user	m.s                                                                  *
 * sys	m.s                                                                  *
 * _________________________________________________________________________ *
 *                        ---SPEEDUP---                                      *
 *                                                                           *
 * 1 THREAD:  (46,779/) =                                                    *
 * 2 THREADS: (/) =                                                          *
 * 4 THREADS: (/) =                                                          *
 * 8 THREADS: (/) =                                                          *
 *****************************************************************************/


// Implementation of the KMeans Algorithm
// reference: http://mnemstudio.org/clustering-k-means-example-1.htm

#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <omp.h>

using namespace std;

class Point
{
	private:
		int id_point, id_cluster;
		vector<double> values;
		int total_values;
		string name;

	public:
		Point(int id_point, vector<double>& values, string name = "")
		{
			this->id_point = id_point;
			total_values = values.size();

			for(int i = 0; i < total_values; i++)
				this->values.push_back(values[i]);

			this->name = name;
			id_cluster = -1;
		}

		int getID()
		{
			return id_point;
		}

		void setCluster(int id_cluster)
		{
			this->id_cluster = id_cluster;
		}

		int getCluster()
		{
			return id_cluster;
		}

		double getValue(int index)
		{
			return values[index];
		}

		int getTotalValues()
		{
			return total_values;
		}

		void addValue(double value)
		{
			values.push_back(value);
		}

		string getName()
		{
			return name;
		}
};

class Cluster
{
	private:
		int id_cluster;
		vector<double> central_values;
		vector<Point> points;

	public:
		Cluster(int id_cluster, Point point)
		{
			this->id_cluster = id_cluster;

			int total_values = point.getTotalValues();

			for(int i = 0; i < total_values; i++)
				central_values.push_back(point.getValue(i));

			points.push_back(point);
		}

		void addPoint(Point point)
		{
			points.push_back(point);
		}

		bool removePoint(int id_point)
		{
			int total_points = points.size();

			for(int i = 0; i < total_points; i++)
			{
				if(points[i].getID() == id_point)
				{
					points.erase(points.begin() + i);
					return true;
				}
			}
			return false;
		}

		double getCentralValue(int index)
		{
			return central_values[index];
		}

		void setCentralValue(int index, double value)
		{
			central_values[index] = value;
		}

		Point getPoint(int index)
		{
			return points[index];
		}

		int getTotalPoints()
		{
			return points.size();
		}

		int getID()
		{
			return id_cluster;
		}
};

class KMeans
{
	private:
		int K; // number of clusters
		int total_values, total_points, max_iterations;
		vector<Cluster> clusters;

		// return ID of nearest center (uses euclidean distance)
		int getIDNearestCenter(Point point)
		{
			double sum = 0.0, min_dist;
			int id_cluster_center = 0;

			#pragma omp parallel for reduction(+:sum)
			for(int i = 0; i < total_values; i++)
			{
				sum += pow(clusters[0].getCentralValue(i) - point.getValue(i), 2.0);
			}

			min_dist = sqrt(sum);

			for(int i = 1; i < K; i++)
			{
				double dist;
				sum = 0.0;

				for(int j = 0; j < total_values; j++)
				{
					sum += pow(clusters[i].getCentralValue(j) - point.getValue(j), 2.0);
				}

				dist = sqrt(sum);

				if(dist < min_dist)
				{
					min_dist = dist;
					id_cluster_center = i;
				}
			}

			return id_cluster_center;
		}

	public:
		KMeans(int K, int total_points, int total_values, int max_iterations)
		{
			this->K = K;
			this->total_points = total_points;
			this->total_values = total_values;
			this->max_iterations = max_iterations;
		}

		void run(vector<Point> & points)
		{
			if(K > total_points)
				return;

			vector<int> prohibited_indexes;

			// choose K distinct values for the centers of the clusters
			for(int i = 0; i < K; i++)
			{
				while(true)
				{
					int index_point = rand() % total_points;

					if(find(prohibited_indexes.begin(), prohibited_indexes.end(),
							index_point) == prohibited_indexes.end())
					{
						prohibited_indexes.push_back(index_point);
						points[index_point].setCluster(i);
						Cluster cluster(i, points[index_point]);
						clusters.push_back(cluster);
						break;
					}
				}
			}

			int iter = 1;

			while(true)
			{
				bool done = true;

				// associates each point to the nearest center
				// regiao paralelizada que demora mais tempo
				#pragma omp parallel for schedule(static, 1500) num_threads(8)
				for(int i = 0; i < total_points; i++)
				{
					int id_old_cluster = points[i].getCluster();
					int id_nearest_center = getIDNearestCenter(points[i]);

					// Regiao critica por causa da modificacoes de variaveis
					#pragma omp critical
					{
						if(id_old_cluster != id_nearest_center)
						{
							if(id_old_cluster != -1)
								clusters[id_old_cluster].removePoint(points[i].getID());

							points[i].setCluster(id_nearest_center);
							clusters[id_nearest_center].addPoint(points[i]);
							done = false;
						}
					}
				}

				// recalculating the center of each cluster
				// reagiao paralelizada que auxilia o cauculo dos centroides dos clusters
				#pragma omp parallel for num_threads(8)
				for(int i = 0; i < K; i++)
				{
					for(int j = 0; j < total_values; j++)
					{
						int total_points_cluster = clusters[i].getTotalPoints();
						double sum = 0.0;

						if(total_points_cluster > 0)
						{
							// regiao paralela que acrescenta um ponto ao cluster
							#pragma omp parallel for reduction(+:sum) num_threads(8)
							for(int p = 0; p < total_points_cluster; p++)
								sum += clusters[i].getPoint(p).getValue(j);
							clusters[i].setCentralValue(j, sum / total_points_cluster);
						}
					}
				}

				if(done == true || iter >= max_iterations)
				{
					cout << "Break in iteration " << iter << "\n\n";
					break;
				}

				iter++;
			}
		}

		void show(vector<Point> & points)
		{
			// shows elements of clusters
			for(int i = 0; i < K; i++)
			{
				int total_points_cluster =  clusters[i].getTotalPoints();

				cout << "Cluster " << clusters[i].getID() + 1 << endl;
				for(int j = 0; j < total_points_cluster; j++)
				{
					cout << "Point " << clusters[i].getPoint(j).getID() + 1 << ": ";
					for(int p = 0; p < total_values; p++)
						cout << clusters[i].getPoint(j).getValue(p) << " ";

					string point_name = clusters[i].getPoint(j).getName();

					if(point_name != "")
						cout << "- " << point_name;

					cout << endl;
				}

				cout << "Cluster values: ";

				for(int j = 0; j < total_values; j++)
					cout << clusters[i].getCentralValue(j) << " ";

				cout << "\n\n";
			}
		}
};

int main(int argc, char *argv[])
{
	srand (time(NULL));

	int total_points, total_values, K, max_iterations, has_name;

	cin >> total_points >> total_values >> K >> max_iterations >> has_name;

	vector<Point> points;
	string point_name;

	for(int i = 0; i < total_points; i++)
	{
		vector<double> values;

		for(int j = 0; j < total_values; j++)
		{
			double value;
			cin >> value;
			values.push_back(value);
		}

		if(has_name)
		{
			cin >> point_name;
			Point p(i, values, point_name);
			points.push_back(p);
		}
		else
		{
			Point p(i, values);
			points.push_back(p);
		}
	}

	KMeans kmeans(K, total_points, total_values, max_iterations);
	kmeans.run(points);

	// mostrando
	// kmeans.show(points);

	return 0;
}
