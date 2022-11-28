#pragma once

#include <fstream>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <vector>

#include "particles.h"
#include "timer.hpp"

/**
 * Read the number of particles and their coordinates
 * into the "particles" variable
 * Memory for "particles" will be allocated in this fuction
 *
 * File asserted to have correct contents
**/
void initialize(Particles& particles, std::string fileName)
{
	std::ifstream input(fileName);

	assert(input.good());

	int n;
	input >> n;
	input.get();
	particles.allocate(n);

	input.read( (char*)particles.x, n * sizeof(double) );
	input.read( (char*)particles.y, n * sizeof(double) );
	input.read( (char*)particles.z, n * sizeof(double) );
	input.read( (char*)particles.m, n * sizeof(double) );

	assert(!input.fail());
	input.close();
}

/**
 * Read the reference forces
 * Compute the L2 norm of the relative difference between the given and the reference forces
 *
 * Return true if the norm is less than threshold
 * Return false if the norm is too big
 *
 * File is asserted to have correct contents
**/
bool check(const Particles& particles, std::string refFileName, bool verbose=false)
{
	const double threshold = 1e-8;

	std::ifstream input(refFileName);
	assert(input.good());

	int n;
	input >> n;
	input.get();
	assert(n == particles.n);

	double *refFx = new double[n];
	double *refFy = new double[n];
	double *refFz = new double[n];

	input.read( (char*)refFx, n * sizeof(double) );
	input.read( (char*)refFy, n * sizeof(double) );
	input.read( (char*)refFz, n * sizeof(double) );
	assert(!input.fail());
	input.close();

	double l2diff = 0;
	for (int i=0; i<n; i++)
	{
		double localL2 = 0;
		localL2 += pow( (particles.fx[i] - refFx[i]) / refFx[i], 2.0);
		localL2 += pow( (particles.fy[i] - refFy[i]) / refFy[i], 2.0);
		localL2 += pow( (particles.fz[i] - refFz[i]) / refFz[i], 2.0);

		if (std::isnan(particles.fx[i]) || std::isnan(particles.fy[i]) || std::isnan(particles.fz[i]))
			localL2 = 1;

		if (verbose && sqrt(localL2) > 0.1*threshold)
			printf("    Particle %5d mismatch. Got [%12.5f %12.5f %12.5f],\n                       reference [%12.5f %12.5f %12.5f]\n\n",
					i, particles.fx[i], particles.fy[i], particles.fz[i],  refFx[i], refFy[i], refFz[i]);

		l2diff += localL2;
	}
	l2diff = sqrt(l2diff / (3.0*n));

	delete[] refFx;
	delete[] refFy;
	delete[] refFz;

	return l2diff < threshold;
}

template<typename Func>
bool test(std::string suffix, Func f, bool strict = true)
{
	std::cout << "Testing case '" << suffix << "'...  ";
	std::cout.flush();

	Particles particles;

	initialize(particles, "data/initial_conditions_"+suffix+".dat");

	timer tm;
	tm.start();
	f(particles);
	tm.stop();

	bool result = check(particles, "data/reference_forces_"+suffix+".dat", false);

	std::cout << ( result ? "PASSED" : "FAILED" ) << "  in " << tm.get_timing() << " sec" << std::endl;

	if (!result && strict)
	{
		std::cout << "  Details about failed test:" << std::endl << std::endl;
		check(particles, "data/reference_forces_"+suffix+".dat", true);
	}

	return result;
}

template<typename Func>
void testAll(Func f, bool strict = true)
{
	std::vector<std::string> cases = {"3", "32", "2017", "11111"};

	for (auto& s : cases)
	{
		bool result = test(s, f, strict);
		if (strict && !result) return;
	}

	std::cout << "All tests passed" << std::endl;
}
