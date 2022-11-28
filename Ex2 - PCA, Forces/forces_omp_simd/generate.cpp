#include <fstream>
#include <cstdlib>
#include <cassert>
#include <cmath>

/**
 * Structure to hold arrays of coordinates, masses and forces per particle
 * Number of particles "n" is included too
 * 
 * Asserts that number of particles is greater than zero when allocating memory
**/
struct Particles
{
	int n;
	double *x, *y, *z, *m;
	double *fx, *fy, *fz;
	
	Particles() : n(-1) {}
	Particles(int n) : n(n)
	{
		allocate(n);
	}
	
	void allocate(int n)
	{
		assert(n > 0);
		
		this->n = n;
		x  = new double[n];
		y  = new double[n];
		z  = new double[n];
		m  = new double[n];
		
		fx = new double[n];
		fy = new double[n];
		fz = new double[n];
	}
	
	~Particles()
	{
		if (n <= 0) return;
		
		delete[] x;
		delete[] y;
		delete[] z;
		delete[] m;
		delete[] fx;
		delete[] fy;
		delete[] fz;
	}	
};

/**
 * Compute the gravitational forces in the system of particles
 * Symmetry of the forces is NOT exploited
**/
void computeGravitationalForces(Particles& particles)
{
	const double G = 6.67408e-11;
	
	auto sqr = [](double x) { return x*x; };
	
	for (int i=0; i<particles.n; i++)
	{
		particles.fx[i] = 0;
		particles.fy[i] = 0;
		particles.fz[i] = 0;
		
		for (int j=0; j<particles.n; j++)
			if (i!=j)
			{
				double tmp = sqr(particles.x[i]-particles.x[j]) +
							 sqr(particles.y[i]-particles.y[j]) +
							 sqr(particles.z[i]-particles.z[j]);
				double tmp05 = sqrt(tmp);
				
				double magnitude = G * particles.m[i] * particles.m[j] / (tmp05 * tmp05 * tmp05);
				
				particles.fx[i] += (particles.x[i]-particles.x[j]) * magnitude;
				particles.fy[i] += (particles.y[i]-particles.y[j]) * magnitude;
				particles.fz[i] += (particles.z[i]-particles.z[j]) * magnitude;
			}
	}
}

void generate(int n)
{
	Particles p(n);
	
	for (int i=0; i<n; i++)
	{
		p.x[i] = 10*drand48();
		p.y[i] = 10*drand48();
		p.z[i] = 10*drand48();
		
		p.m[i] = 1e7 / sqrt((double)n) *drand48();
	}
	
	{
		std::ofstream fout("data/initial_conditions_"+std::to_string(n)+".dat");
		fout << n << std::endl;
		fout.write((char*)p.x, n*sizeof(double));
		fout.write((char*)p.y, n*sizeof(double));
		fout.write((char*)p.z, n*sizeof(double));
		fout.write((char*)p.m, n*sizeof(double));
		fout.close();
	}
	
	computeGravitationalForces(p);
	
	{
		std::ofstream fout("data/reference_forces_"+std::to_string(n)+".dat");
		fout << n << std::endl;
		fout.write((char*)p.fx, n*sizeof(double));
		fout.write((char*)p.fy, n*sizeof(double));
		fout.write((char*)p.fz, n*sizeof(double));
		fout.close();
	}
}

int main()
{
	
	srand48(42);
	generate(3);
	generate(32);
	generate(2017);
	generate(11111);
	generate(33333);
	
	return 0;
}
