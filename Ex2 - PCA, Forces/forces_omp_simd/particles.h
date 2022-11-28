#pragma once

#include <cassert>

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
