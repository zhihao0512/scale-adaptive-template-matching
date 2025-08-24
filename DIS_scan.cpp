#include <mex.h>
#include <vector>
#include <xtgmath.h>
#include "Mat.h"
#include <algorithm>

void computescore(int* pnnf, double* pscore, const int Tr, const int Tc, const int Ir, const int Ic)
{
	int N = Tr*Tc;
	for (int colI = 0; colI < Ic - Tc + 1; ++colI)
	{
		std::vector<int> matchsum(N + 1);
		for (int rowI = 0; rowI < Ir - Tr + 1; ++rowI)
		{
			if (rowI == 0)
			{
				for (int i = 0; i < Tr; ++i)
				{
					for (int j = 0; j < Tc; ++j)
					{
						matchsum[pnnf[(colI + j)*(Ir)+rowI + i]]++;						
					}					
				}
			}
			else
			{
				for (int j = 0; j < Tc; ++j)
				{
					matchsum[pnnf[(colI + j)*(Ir)+rowI - 1]]--;
				}
				for (int j = 0; j < Tc; ++j)
				{
					matchsum[pnnf[(colI + j)*(Ir)+rowI + Tr - 1]]++;
				}
			}
			double sscore = 0.0;

			for (int i = 0; i < matchsum.size(); ++i)
			{
				if (matchsum[i] != 0)
					sscore++;
			}
			pscore[colI*(Ir - Tr + 1) + rowI] = sscore/N;
		}
		
	}
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	size_t Irows = mxGetM(prhs[0]);
	size_t Icols = mxGetN(prhs[0]);
	int* nnfpArray = (int32_T *)mxGetData(prhs[0]);

	int Trows = (int)(mxGetScalar(prhs[1]));
	int Tcols = (int)(mxGetScalar(prhs[2]));

	mxArray *outMat;
	outMat = mxCreateDoubleMatrix((mwSize)Irows-Trows+1, (mwSize)Icols-Tcols+1, mxREAL);
	plhs[0] = outMat;
	
	computescore(nnfpArray, mxGetPr(outMat), (int)Trows, (int)Tcols, (int)Irows, (int)Icols);
}