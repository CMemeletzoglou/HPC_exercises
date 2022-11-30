// File       : pca.cpp
// Description: Principal component analysis applied to image compression
#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <cmath>

#include <omp.h>
#include <zlib.h>

// interface for LAPACK routines.
#include <lapack.h>

#include <iomanip> 

///////////////////////////////////////////////////////////////////////////////
// helpers
double *read_gzfile(char *filename, int frows, int fcols, int rows, int cols)
{
        double *A, *buf;
        gzFile fp;
        int i;

        A = new (std::nothrow) double[rows*cols];
        assert(A != NULL);

        buf = new (std::nothrow) double[fcols];
        assert(buf != NULL);

        fp = gzopen(filename, "rb");
        if (fp == NULL)
        {
                std::cout << "Input file not available!\n";
                exit(1);
        }

        for (i = 0; i < rows; i++)
        {
                gzread(fp, buf, fcols*sizeof(double));
                memcpy(&A[i*cols], buf, cols*sizeof(double));
        }
        gzclose(fp);

        delete[] buf;

        return A;
}

void write_ascii(const char* const filename, const double* const data, const int rows, const int cols)
{
        FILE *fp = fopen(filename, "w");
        if (fp == NULL)
        {
                std::cout << "Failed to create output file\n";
                exit(1);
        }

        for (int i = 0; i < rows; i++)
        {
                for (int j = 0; j < cols; j++)
                {
                        fprintf(fp, "%.4lf ", data[i*cols+j]);
                }
                fprintf(fp, "\n");
        }
        fclose(fp);
}

template <typename matrix_t>
void print_matrix(matrix_t* A, int n, int m, int limit_n, int limit_m)
{
        for (int i = 0; i < limit_n; i++)
        {
                for (int j = 0; j < limit_m-1; j++)
                {
                        std::cout << std::setw(10) << A[i * m + j] << ", ";
                }
                std::cout << A[i * m + limit_m - 1] << "\n";
        }
}
/* Helper function to gather the data contained in column col_idx of the n x m
 * array passed in the first argument, into the buf buffer
 */
void cp_column_in_buffer(double* M, int n, int m, int col_idx, double* buf) 
{
        for (int i = 0; i < n; i++)
        {
                buf[i] = M[i * m + col_idx];
        }
}

///////////////////////////////////////////////////////////////////////////////

/* compute the covariance of the two row-vectors corresponding to lines i and j
 * of matrix A
 */
double cov(double *A, int i, int j, int n, int m)
{
        double tmp = 0.0;
        for (int k = 0; k < m; k++)
        {
                // compute cov(A(:,i), A(:,j))
                tmp += (A[i * m + k]) * (A[j * m + k]);
        }
        return (tmp / (m - 1)); // cov calculated (C(i,j) element)
}

///////////////////////////////////////////////////////////////////////////////
//
// elvis.bin.gz:   469x700
// cyclone.bin.gz: 4096x4096
// earth.bin.gz:   9500x9500
//
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
        // input parameters (default)
        int m = 469, n = 700;                                   // image size (rows, columns)
        int npc = 50;                                           // number of principal components
        char *inp_filename = (char *)"../data/elvis.bin.gz";    // input filename (compressed binary)
        char *out_filename = (char *)"elvis.50.bin";            // output filename (text)

        // parse input parameters
        if ((argc != 1) && (argc != 11))
        {
                std::cout << "Usage: " << argv[0] << " -m <rows> -n <cols> -npc <number of principal components> -if <input filename> -of <output filename>\n";
                exit(1);
        }

        for(int i = 1; i < argc; i++ )
        {
                if( strcmp( argv[i], "-m" ) == 0 )
                {
                        m = atoi(argv[i+1]);
                        i++;
                }
                if( strcmp( argv[i], "-n" ) == 0 )
                {
                        n = atoi(argv[i+1]);
                        i++;
                }
                if( strcmp( argv[i], "-npc" ) == 0 )
                {
                        npc = atoi(argv[i+1]);
                        i++;
                }
                if( strcmp( argv[i], "-if" ) == 0 )
                {
                        inp_filename = argv[i+1];
                        i++;
                }
                if( strcmp( argv[i], "-of" ) == 0 )
                {
                        out_filename = argv[i+1];
                        i++;    
                }
        }

        if (npc > n)
                npc = n;
        
        double t_elapsed;

        ///////////////////////////////////////////////////////////////////////////
        // Read image data.  The image dimension is m x n.  The returned pointer
        // points to the data in row-major order.  That is, if (i,j) corresponds to
        // to the row and column index, respectively, you access the data with
        // pixel_{i,j} = I[i*n + j], where 0 <= i < m and 0 <= j < n.
        ///////////////////////////////////////////////////////////////////////////
        double *I = read_gzfile(inp_filename, m, n, m, n);

        // The algorithm works by processing the matrix data in a **columnwise** manner.
        // A = transpose(I), so image features (columns) are stored in rows.  More
        // efficient to work with the data in this layout.
        double *A = new (std::nothrow) double[n*m];
        assert(A != NULL);

        // get each column vector of I and store it as a row vector of A
        for (int i = 0; i < n; i++) 
        {
                for (int j = 0; j < m; j++)
                {
                        A[i*m + j] = I[j*n+i]; 
                }
        }
        delete[] I;

        ///////////////////////////////////////////////////////////////////////////
        // The compressed image file is stored in a **row-wise** manner
        ///////////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////////
        // TODO: Implement your PCA algorithm here
        // 1. Compute mean and standard deviation of your image features (= image columns) (**DONE**)
        // 2. Normalize the data (**DONE**)
        // 3. Build the covariance matrix (**DONE**)
        // 4. Compute the eigenvalues and eigenvectors of the covariance matrix.
        //    Use LAPACK here. (**DONE**)
        // 5. Compute the principal components and report the compression ratio (**DONE**)
        // 6. Reconstruct the image from the compressed data and dump the image in
        //    ascii.
        ///////////////////////////////////////////////////////////////////////////
        double start_t = omp_get_wtime();

        ///////////////////////////////////////////////////////////////////////////
        // TODO: 1.
        t_elapsed = -omp_get_wtime();

        double *AMean = new (std::nothrow) double[n];
        double *AStd  = new (std::nothrow) double[n];
        assert(AMean != NULL);
        assert(AStd  != NULL);

        for (int i = 0; i < n; i++)
        {
                // TODO: compute mean and standard deviation of features of A (**DONE**)

                double col_mean = 0.0;
                double col_sum = 0.0;

                // each **row** of A is a **column** of the image .. A is n x m
                // the original image matrix I is m x n

                // mean calculation
                for (int j = 0; j < m; j++)
                {
                        col_mean += A[i*m+j];
                }
                col_mean = col_mean / m; 
                AMean[i] = col_mean; // TODO(?): maybe we don't need the col_mean intermediate variable

                // standard deviation calculation
                for (int j = 0; j < m; j++)
                {
                        col_sum += pow((A[i*m+j] - col_mean), 2);
                }
                AStd[i] = sqrt(col_sum / (m - 1));
        }

        t_elapsed += omp_get_wtime();
        std::cout << "MEAN/STD TIME=" << t_elapsed << " seconds\n";
        ///////////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////////
        // TODO: 2.
        t_elapsed = -omp_get_wtime();

        for (int i = 0; i < n; i++)
        {
                for (int j = 0; j < m; j++)
                {
                        // TODO: normalize data here (**DONE**)
                        /* Data Normalization : First substract from each column of I
                         * (i.e from each row of A) its sample mean, and then divide by its
                         * standard deviation.
                         *
                         * AMean[i] = the mean of the i-th row of A (the i-th column of I)
                         * AStd[i] = the std dev of the i-th row of A (the i-th column of I)
                         */
                        A[i*m+j] = (A[i*m+j] - AMean[i]) / AStd[i];
                }
        }

        t_elapsed += omp_get_wtime();
        std::cout << "NORMAL TIME=" << t_elapsed << " seconds\n";
        ///////////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////////
        // TODO: 3.
        t_elapsed = -omp_get_wtime();
        double *C = new (std::nothrow) double[n*n]; // covariance matrix
        assert(C != NULL);

        // TODO: Compute covariance matrix here (**DONE**)

        /* For a m x n matrix, where each of the m rows is an observation and each of the 
         * n columns is a Random variable vector, the Covariance matrix can be calculated as :
         *
         * C(i,j) = cov(I(:,i), I(:,j)), where I is the original image array
         * 
         * Therefore, each element of the Covariance matrix is the pairwise covariance of 
         * each column vector combination.
         * For the array A, we have : C(i,j) = cov(A(i,:), A(j,:))
         *
         * In order to compute the covariance matrix, we need the mean and standard deviation
         * of each row of the **NORMALIZED** data matrix A. (remember : covariance = std^2)
         * 
         * So we CAN'T use the values AMean[i] and AStd[i] for the mean and standard deviation 
         * of the row vectors, because those values correspond to the initial 
         * **NON-NORMALIZED** matrix. 
         * 
         * For the normalized matrix, each row has a mean of 0 and an std of 1.
         */
        for (int i = 0; i < n; i++)
        {
                // C(i,i)  = var(Xi) = AStd[i]^2
                C[i * n + i] = 1;
                // for (int j = i + 1; j < n; j++)
                for (int j = 0; j < i; j++)
                {
                        // C(i,j) = C(j,i) = cov(Xi, Xj)
                        C[i * n + j] = cov(A, i, j, n, m);
                        // C[j * n + i] = C[i * n + j];
                }
        }
        
        t_elapsed += omp_get_wtime();
        // return 0;
        std::cout << "C-MATRIX TIME=" << t_elapsed << " seconds\n";
        ///////////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////////
        // TODO 4. LAPACK (dsyev call)
        t_elapsed = -omp_get_wtime();

        // see also for the interface to dsyev_():
        // http://www.netlib.org/lapack/explore-html/d2/d8a/group__double_s_yeigen_ga442c43fca5493590f8f26cf42fed4044.html#ga442c43fca5493590f8f26cf42fed4044
        const char jobz = 'V'; // TODO: compute both, eigenvalues and orthonormal eigenvectors
        // const char uplo = 'L'; // TODO: how did you compute the (symmetric) covariance matrix?
        const char uplo = 'U'; // TODO: how did you compute the (symmetric) covariance matrix?
        int info, lwork;

        double *W = new (std::nothrow) double[n]; // eigenvalues
        assert(W != NULL);

        double *work = new (std::nothrow) double[1];
        assert(work != NULL);

        // first call to dsyev_() with lwork = -1 to determine the optimal workspace (cheap call)
        lwork = -1;      
        dsyev_(&jobz, &uplo, &n, C, &n, W, work, &lwork, &info, 1, 1);

        lwork = (int)work[0];
        delete[] work;

        // allocate optimal workspace
        work = new (std::nothrow) double[lwork];
        assert(work != NULL);

        // second call to dsyev_(), eigenvalues and eigenvectors are computed here
        dsyev_(&jobz, &uplo, &n, C, &n, W, work, &lwork, &info, 1, 1);

        t_elapsed += omp_get_wtime();
        std::cout << "DSYEV TIME=" << t_elapsed << " seconds\n";
        ///////////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////////
        // TODO: 5.
        t_elapsed = -omp_get_wtime();
        double *PCReduced = new (std::nothrow) double[m*npc](); // Initializes memory to 0
        assert(PCReduced != NULL);

        double *buf = new (std::nothrow) double[n];
        assert(buf != NULL);

        // TODO: compute the principal components
        /* Note: dsyev_ returns the eigenvectors in a form of matrix, where each **row**
         * is one eigenvector. This matrix is stored in C.
         */
        int c_offset = n - npc;
        int c_row = 0;
        for(int i=0; i<m; i++)
        {
                // gather the data from the i-th column into the buf buffer
                cp_column_in_buffer(A, n, m, i, buf);
                
                // pick a principal component (i.e an eigenvector = row vector of C)
                for(int pc=0; pc<npc; pc++) // keep the first npc principal components
                {
                        c_row = c_offset + pc;
                        
                        for(int j=0; j<n; j++) // iterate through the eigenvector
                                PCReduced[i* npc + pc] += buf[j] * C[c_row * n + j];
                }
        }

        delete[] buf;

        // TODO: Report the compression ratio
        std::cout << "COMPRESSION RATIO=" << std::setprecision(4) <<  ((double)(n - npc)/n)  * 100 << "%\n";

        t_elapsed += omp_get_wtime();
        std::cout << "PCREDUCED TIME=" << t_elapsed << " seconds\n";
        ///////////////////////////////////////////////////////////////////////////

        double end_t = omp_get_wtime();
        std::cout << "OVERALL TIME=" << end_t - start_t << " seconds\n";

        ///////////////////////////////////////////////////////////////////////////
        // // TODO: 6
        // double *Z = new (std::nothrow) double[m*n]; // memory for reconstructed image
        // assert(Z != NULL);

        // for (int i = 0; i < m; i++)
        // {
        //         for (int j = 0; j < n; j++)
        //         {
        //                 // TODO: Reconstruct image here.  Don't forget to denormalize.  The
        //                 // dimension of the reconstructed image is m x n (rows x columns).
        //                 // Z[i*n + j] = ...
        //         }
        // }

        // // Write the reconstructed image in ascii format.  You can view the image
        // // in Matlab with the show_image.m script.
        // write_ascii(out_filename, Z, m, n);
        // ///////////////////////////////////////////////////////////////////////////

        // // cleanup
        delete[] work;
        delete[] W;
        delete[] C;
        // delete[] Z;
        delete[] PCReduced;
        delete[] A;
        delete[] AMean;
        delete[] AStd;

        return 0;
}
