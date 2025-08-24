#include <mex.h>
#include <vector>
#include <xtgmath.h>
#include "Mat.h"
#include <algorithm>

void computescore(int* pnnf, int* pmask, double* pscore, const int Tr, const int Tc, const int Ir, const int Ic, const int Sr, const int Sc, const double _alpha = 20.0, const double _gamma = 0.5)
{
	int N = Sr * Sc;
	int Nt = Tr * Tc;

	int padc = _gamma *sqrt(Nt);
	int padr = _gamma *sqrt(Nt);
	int area = (2 * padc + Sc)*(2 * padr + Sr) - N;
	for (int colI = 0; colI < Ic - Sc + 1; ++colI)
	{
		std::vector<int> matchnum(Nt + 1);
		for (int rowI = 0; rowI < Ir - Sr + 1; ++rowI)
		{
			int overlap = (std::min(colI + padc + Sc, Ic) - std::max(colI - padc, 0))*(std::min(rowI + padr + Sr, Ir) - std::max(rowI - padr, 0)) - N;
			if (rowI == 0)
			{
				for (int i = -padr; i < padr + Sr; ++i)
				{
					for (int j = -padc; j < padc + Sc; ++j)
					{
						if ((colI + j) >= 0 && (colI + j) < Ic && (rowI + i) >= 0 && (rowI + i) < Ir && (!(i >= 0 && i < Sr&&j >= 0 && j < Sc)))
							matchnum[pnnf[(colI + j)*(Ir)+rowI + i]]++;
					}
				}
			}
			else
			{
				for (int j = -padc; j < padc + Sc; ++j)
				{
					if ((colI + j) >= 0 && (colI + j) < Ic && (rowI - padr - 1) >= 0 && (rowI - padr - 1) < Ir)
						matchnum[pnnf[(colI + j)*(Ir)+rowI - padr - 1]]--;
				}
				for (int j = -padc; j < padc + Sc; ++j)
				{
					if ((colI + j) >= 0 && (colI + j) < Ic && (rowI + padr + Sr - 1) >= 0 && (rowI + padr + Sr - 1) < Ir)
						matchnum[pnnf[(colI + j)*(Ir)+rowI + padr + Sr - 1]]++;
				}
				for (int j = 0; j < Sc; ++j)
				{
					matchnum[pnnf[(colI + j)*(Ir)+rowI - 1]]++;
				}
				for (int j = 0; j < Sc; ++j)
				{
					matchnum[pnnf[(colI + j)*(Ir)+rowI + Sr - 1]]--;
				}
			}

			std::vector<double> matchsum(Nt + 1);
			double sscore = 0.0;
			if (pmask[(colI + Sc / 2)*Ir + rowI + Sr / 2])
			{
				for (int i = 0; i < Sr; ++i)
				{
					for (int j = 0; j < Sc; ++j)
					{
						int idx = (colI + j)*(Ir)+rowI + i;
						int Trowidx = (pnnf[idx] - 1) % (Tr);
						int Tcolidx = (pnnf[idx] - 1) / (Tr);

						double def = (Trowidx - (double)i*Tr / Sr)*(Trowidx - (double)i*Tr / Sr) + (Tcolidx - (double)j*Tc / Sc)*(Tcolidx - (double)j*Tc / Sc);
						if (def < 6 * (double)Nt / _alpha)
							matchsum[pnnf[(colI + j)*(Ir)+rowI + i]] += exp(-_alpha * def / Nt);
					}					
				}
				for (int i = 0; i < matchsum.size(); ++i)
				{
					if (matchsum[i] != 0)
						sscore += (1 - exp(-(double)matchsum[i] * Nt / N))/(1+ (double)matchnum[i] / overlap * 3 * Nt);
				}
			}

			pscore[colI*(Ir - Sr + 1) + rowI] = sscore;
		}
		
	}
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	size_t Irows = mxGetM(prhs[0]);
	size_t Icols = mxGetN(prhs[0]);
	int* nnfpArray = (int32_T *)mxGetData(prhs[0]);
	
	int* mask = (int32_T *)mxGetData(prhs[1]);
	int Trows = (int)(mxGetScalar(prhs[2]));
	int Tcols = (int)(mxGetScalar(prhs[3]));
	int Srows = (int)(mxGetScalar(prhs[4]));
	int Scols = (int)(mxGetScalar(prhs[5]));
	double alpha = (double)(mxGetScalar(prhs[6]));
	double gamma = (double)(mxGetScalar(prhs[7]));

	mxArray *outMat;
	outMat = mxCreateDoubleMatrix((mwSize)Irows-Srows+1, (mwSize)Icols-Scols+1, mxREAL);
	plhs[0] = outMat;

	computescore(nnfpArray, mask, mxGetPr(outMat), (int)Trows, (int)Tcols, (int)Irows, (int)Icols, (int)Srows, (int)Scols, (double)alpha, (double)gamma);
}