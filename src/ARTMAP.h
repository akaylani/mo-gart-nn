//////////////////////////////////////////////////////////////////////
//
// This code was developed by Assem Kaylani (akaylani@gmail.com) 
// and Dr. Michael Georgiopoulos (michaelg@ucf.edu).
//
// ARTMAP class definition. This class implements the Fuzzy ARTMAP algorithm
// (FAM) described in [7], Ellipsoidal ARTMAP (EAM) as described in [3], Gaussian 
// ARTMAP (GAM) as described in [8], semi-supervised architectures (ssFAM, ssEAM 
// and ssGAM) descibed in [4]. 
//
// [3] G. Anagnostopoulos, "Novel approaches in adaptive resonance theory 
// for machine learning," Ph.D. dissertation, University of Central
// Florida, Orlando, May 2001.
//
// [4] G. C. Anagnostopoulos, M. Bharadwaj, M. Georgiopoulos, S. J. Verzi, 
// and G. L. Heileman, "Exemplar-based pattern recognition via
// semi-supervised learning," in Proceedings of the 2003 International 
// Joint Conference on Neural Networks (IJCNN ’03), vol. 4, Portland,
// Oregon, USA, 2003, pp. 2782–2787.
//
// [6] A. Koufakou, M. Georgiopoulos, G. Anagnostopoulos, and T. Kasparis, 
// "Cross-validation in Fuzzy ARTMAP for large databases," Neural
// Networks, vol. 14, pp. 1279–1291, 2001.
//
// [7] G. A. Carpenter, S. Grossberg, N. Markuzon, J. H. Reynolds, and D. B. 
// Rosen, "Fuzzy ARTMAP: A neural network architecture for incremental 
// supervised learning of analog multidimensional maps," IEEE Transactions 
// on Neural Networks, vol. 3, pp. 698–713, 1992.
//
// [8] J. R. Williamson, "Gaussian ARTMAP: A neural network for fast 
// incremental learning of noisy multidimensional maps," Neural Networks,
// vol. 9, no. 5, pp. 881–897, 1996.
// 
//////////////////////////////////////////////////////////////////////

#pragma once

#include "dutil.h"
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define unif() (((double) rand()) / (double) RAND_MAX)

double ltqnorm(double p);

//////////////////////////////////////////////////////////////////////
//
// Class definition of a base category in an ARTMAP network. 
// This class must be inherited.
//
//////////////////////////////////////////////////////////////////////
class CategoryBase
{
public:
	int				label;				//or double* output;
	unsigned int	numAttribs;		
	double			CMF_value;			// Ro
	double			CCF_value;			// T
	unsigned int	n;					// number of petterns that selected this category

	unsigned int	active;				// number of petterns that selected this category (reset every epoch). During test this is the number of patterns that selected the category
	unsigned int	numPatternsLearned; // Number of patterns that affected the size and location of the category. During testing this is the number of patters that were correctly classified.
	unsigned int	toleratedPatternsCount; // Number of patterns that selected this category and allowed to be encoded by it, but are of the wrong label.
	double			cf;					// Confidence factor calculated externally based on performance on validation set.

	CategoryBase()
	{
		label = -1;
		numAttribs = 0;
		CMF_value = 0;
		CCF_value = 0;
		n = 0;
		active = 0;
		numPatternsLearned = 0;
		toleratedPatternsCount = 0;
		cf = 0;
	}

	~CategoryBase()
	{
	}

	virtual void copyCat(CategoryBase* cat)
	{
		label = cat->label;
		numAttribs = cat->numAttribs;
		n = cat->n;

		active = cat->active;
		numPatternsLearned = cat->numPatternsLearned;		
		toleratedPatternsCount = cat->toleratedPatternsCount;
		cf = cat->cf;
	}

	virtual void freeResource() {}
	virtual void presentPattern(TableRow* row){};
	virtual unsigned int learnPattern(TableRow* row) {return 0;}
	virtual void mutate(double severityFactor){};
	virtual void initRandom(unsigned int numClasses){};
	virtual void getDescription(double* catf){};
	virtual void setDescription(double* catf){};
	virtual unsigned int getCategoryType(){return 0;}
};

//////////////////////////////////////////////////////////////////////
//
// Class definition of a FAM category in a FAM network. 
//
//////////////////////////////////////////////////////////////////////
class FAMCategory : CategoryBase
{
private:
	double*			w;
	double			sizeW;
	double			choiceParameter;

public:

	FAMCategory (unsigned int Ma, double beta, double initWeight)
	{
		numAttribs = Ma;
		w = (double*) malloc(Ma * 2 * sizeof(double));
		sizeW = 0;

		for(unsigned int i = 0; i < (Ma * 2); i++)
		{
			w[i] = initWeight;
			sizeW += initWeight;
		}

		choiceParameter = beta;
	}

	FAMCategory(FAMCategory* cat)
	{
		copyCat(cat);
		
		w = (double*) malloc(numAttribs * 2 * sizeof(double));		
		for(unsigned int i = 0; i < (numAttribs * 2); i++)
			w[i] = cat->w[i];
		
		sizeW = cat->sizeW;		
		choiceParameter = cat->choiceParameter;
	}

	~FAMCategory ()
	{
		freeResource();
	}

	void freeResource()
	{
		if (w) free(w);
		w = 0;
	}

	void presentPattern(TableRow* row)
	{		
		double ret = 0.0;
		for(unsigned int i = 0; i < numAttribs; i++)
		{
			unsigned int j = i + numAttribs;
			double u = *((double*) row->fields[i]);
			if (u < w[i]) ret += u; else ret += w[i];

			double v = 1.0 - u;
			if (v < w[j]) ret += v; else ret += w[j];
		}

		CCF_value = (ret /(choiceParameter + sizeW));
		CMF_value = (ret / numAttribs);
	}

	unsigned int learnPattern(TableRow* row)
	{
		unsigned int ret = 0;
		sizeW = 0;
		for(unsigned int i = 0; i < numAttribs; i++)
		{
			unsigned int j = i + numAttribs;
			double u = *((double*) row->fields[i]);
			if (u < w[i]) 
			{
				w[i] = u;
				ret = 1;
			}

			double v = 1.0 - u;
			if (v < w[j]) 
			{
				w[j] = v;
				ret = 1;
			}

			sizeW += (w[i] + w[j]);
		}

		n++;
		active++;
		numPatternsLearned += ret;
		return ret;
	}

public:
	unsigned int getCategoryType(){return 0;}
	void mutate(double severityFactor)
	{
		if (unif() > 0.5) 
		{
			sizeW = 0;
			for(unsigned int i = 0; i < numAttribs; i++)
			{
				unsigned int j = i + numAttribs;
				double u = w[i];
				double v = 1.0 - w[j];
				double rv = severityFactor * ltqnorm(unif()); // gaussuan 
				u += rv;
				if (u > v) u = v;				
				if (u < 0.0) u = 0.0;				
				w[i] = u;

				sizeW += (w[i] + w[j]);
			}
		}
		else
		{
			sizeW = 0;
			for(unsigned int i = 0; i < numAttribs; i++)
			{
				unsigned int j = i + numAttribs;
				double u = w[i];
				double v = 1 - w[j];
				double rv = severityFactor * ltqnorm(unif()); // gaussuan 
				v += rv;
				if (v > 1.0) v = 1.0;
				if (v < u) v = u;			
				w[j] = 1.0 - v;

				sizeW += (w[i] + w[j]);
			}
		}
	}

	void initRandom(unsigned int numClasses)
	{
		sizeW = 0;
		for(unsigned int i = 0; i < numAttribs; i++)
		{
			double p1 = unif();
			double p2 = unif();
			double u = 0.0;
			double v = 1.0;

			if (p1 > p2) 
			{
				u = p2;
				v = p1;
			}
			else
			{
				u = p1;
				v = p2;
			}

			w[i] = u;
			w[i + numAttribs] = 1.0 - v;

			sizeW += (w[i] + w[i + numAttribs]);
		}
		label = rand() % numClasses;
	}

	void getDescription(double* catf)
	{
		for(unsigned int i = 0; i < (2*numAttribs); i++)
		{
			catf[i] = w[i];
		}
		catf[2*numAttribs] = cf;
	}	
	void setDescription(double* catf)
	{
		sizeW = 0;
		for(unsigned int i = 0; i < (2*numAttribs); i++)
		{
			w[i] = catf[i];
			sizeW += w[i];
		}

		cf = catf[2*numAttribs];
	}
};

//////////////////////////////////////////////////////////////////////
//
// Class definition of an EAM category in an EAM network. 
//
//////////////////////////////////////////////////////////////////////
class EAMCategory : CategoryBase
{
private:
	double*			m;			// center
	double*			d;			// Direction vector
	double			R;			// Half the major axis
	double			mu;			// axis Ratio
	double			alpha;		// choice parameter	
	double			D;

	// Current values for row presentation
	double*			distVect;	// Current distance vector
	double			dis_e_sq;
	double			dis_m;
	double			max_d;

public:

	EAMCategory(unsigned int Ma, double a, double u)
	{
		numAttribs = Ma;
		m = (double*) malloc(Ma * sizeof(double));
		d = (double*) malloc(Ma * sizeof(double));
		distVect = (double*) malloc(Ma * sizeof(double));
		for(unsigned int i = 0; i < numAttribs; i++)
		{
			m[i] = 0.0;
			d[i] = 0.0;
			distVect[i] = 0.0;
		}
		R = 0;
		mu = u;
		alpha = a;
		D = sqrt((double) Ma) / mu;
		dis_e_sq = 0.0;
		dis_m = 0.0;
		max_d = 0.0;
	}

	EAMCategory(EAMCategory* cat)
	{
		copyCat(cat);

		m = (double*) malloc(numAttribs * sizeof(double));
		d = (double*) malloc(numAttribs * sizeof(double));
		distVect = (double*) malloc(numAttribs * sizeof(double));
		for(unsigned int i = 0; i < numAttribs; i++)
		{
			m[i] = cat->m[i];
			d[i] = cat->d[i];
			distVect[i] = cat->distVect[i];
		}
		R = cat->R;
		mu = cat->mu;
		alpha = cat->alpha;
		D = sqrt((double) numAttribs) / mu;
		n = cat->n;
		dis_e_sq = 0.0;
		dis_m = 0.0;
		max_d = 0.0;
	}

	~EAMCategory()
	{
		freeResource();
	}

	void freeResource()
	{
		if (m) free(m); m = 0;
		if (d) free(d); d = 0;
		if (distVect) free(distVect); distVect = 0;
	}

	void presentPattern(TableRow* row)
	{
		if (!n) return;
		double dProd = 0.0;
		dis_e_sq = 0.0;
		for(unsigned int i = 0; i < numAttribs; i++)
		{
			double pattern_m = *((double*) row->fields[i]);
			double aux = pattern_m - m[i];
			dProd += aux * d[i];
			dis_e_sq += aux * aux;
			distVect[i] = aux;
		}
		dis_m = (sqrt(dis_e_sq - ((1 - (mu * mu)) * dProd * dProd))) / mu;
		max_d = dis_m > R ? dis_m : R;
		double CF_Numer = (D - R - max_d);

		CCF_value = CF_Numer / (D - (2 * R) + alpha);
		CMF_value = CF_Numer / D;
	}

	unsigned int learnPattern(TableRow* row)
	{
		unsigned int ret = 0;
		if (label == -1)
		{
			for(unsigned int i = 0; i < numAttribs; i++)
			{
				double pattern_m = *((double*) row->fields[i]);
				m[i] = pattern_m;
				d[i] = 0.0;				
			}
			R = 0;
			ret = 1;
			numPatternsLearned++;
		}
		else
		{
			/// TODO: This check is added without enough evaluation.
			/// when a node that has one pattern is presented again with the same pattern
			/// the distance is 0.0. 
			/// For now.. just do NOT update.
			if (dis_e_sq != 0.0) 
				if(dis_m > R)
				{
					// Update M
					for(unsigned int i = 0; i < numAttribs; i++)
					{
						m[i] = m[i] + 0.5 * ((max_d - R) / dis_m) * distVect[i];
					}

					double RNew;
					double eDis = sqrt(dis_e_sq);
					if (n == 1)
					{
						RNew = eDis / 2.0;
						for(unsigned int i = 0; i < numAttribs; i++)
						{
							d[i] = distVect[i] / eDis;
						}
					}
					else
					{
						RNew = R + 0.5 * (max_d - R);
					}

					R = RNew;				
					ret = 1;
					numPatternsLearned++;
				}
		}
		n++;
		active++;

		return ret;
	}


private:

public:
	unsigned int getCategoryType(){return 1;}
	void mutate(double severityFactor)
	{
		for(unsigned int i = 0; i < numAttribs; i++)
		{
			double p1 = severityFactor * ltqnorm(unif()); // gaussian
			m[i] += p1; 
			if (m[i] > 1.0) m[i] = 1.0;
			if (m[i] < 0.0) m[i] = 0.0;
		}

		if (unif() > 0.5) 
		{
			double p1 = severityFactor * ltqnorm(unif()); // gaussian
			R += p1;
			if (R > 1.0) R = 1.0;
			if (R <= 0) R = 0.0001;
		}
		else
		{
			double p1 = severityFactor * ltqnorm(unif()); // gaussian
			mu += p1;
			if (mu <= 0) mu = 0.0001;
			if (mu > 1.0) mu = 1.0;
		}
	}

	void initRandom(unsigned int numClasses)
	{
		for(unsigned int i = 0; i < numAttribs; i++)
		{
			double p1 = unif();
			m[i] = p1;
			d[i] = 0.0;
		}
		mu = 1.0;
		R = unif();
		
		label = rand() % numClasses;
	}

	void getDescription(double* catf)
	{
		for(unsigned int i = 0; i < numAttribs; i++)
		{
			catf[i] = m[i];
		}

		for(unsigned int i = 0; i < numAttribs; i++)
		{
			catf[i + numAttribs] = d[i];
		}

		catf[numAttribs * 2] = R;
		catf[numAttribs * 2 + 1] = mu;
		catf[numAttribs * 2 + 2] = alpha;
		catf[numAttribs * 2 + 3] = D;
		catf[numAttribs * 2 + 4] = cf;
	}

	void setDescription(double* catf)
	{
		for(unsigned int i = 0; i < numAttribs; i++)
		{
			m[i] = catf[i];
		}
		
		for(unsigned int i = 0; i < numAttribs; i++)
		{
			d[i] = catf[i + numAttribs];
		}

		R = catf[numAttribs * 2];
		mu = catf[numAttribs * 2 + 1];
		alpha = catf[numAttribs * 2 + 2];
		D = catf[numAttribs * 2 + 3];
		cf = catf[numAttribs * 2 + 4];
	}
};

//////////////////////////////////////////////////////////////////////
//
// Class definition of a GAM category in a GAM network. 
//
//////////////////////////////////////////////////////////////////////
class GAMCategory  : CategoryBase
{
private:
	double*			mu;		// The mean of the data that have activated and were encoded by node j
	double*			s;		// The standard deviation vector of the data that have activated and were encoded by node j
	double*			v;		
	double			gamma;  // 
	double			sigmaProd;

public:

	GAMCategory(unsigned int Ma, double gm)
	{
		numAttribs = Ma;
		mu = (double*) malloc(Ma * sizeof(double));
		s = (double*) malloc(Ma * sizeof(double));
		v = (double*) malloc(Ma * sizeof(double));

		for(unsigned int i = 0; i < Ma; i++)
		{
			mu[i] = 0;
			s[i] = 0;
		}

		gamma = gm;
		sigmaProd = 1.0;
	}

	GAMCategory(GAMCategory* cat)
	{
		copyCat(cat);

		mu = (double*) malloc(numAttribs * sizeof(double));
		s = (double*) malloc(numAttribs * sizeof(double));
		v = (double*) malloc(numAttribs * sizeof(double));
		gamma = cat->gamma;
		sigmaProd = cat->sigmaProd;

		for(unsigned int i = 0; i < numAttribs; i++)
		{
			mu[i] = cat->mu[i];
			s[i] = cat->s[i];
			v[i] = cat->v[i];
		}
	}

	~GAMCategory()
	{
		freeResource();
	}

	void freeResource()
	{
		if (mu) free(mu); mu = 0;
		if (s) free(s); s = 0;
		if (v) free(v); v = 0;
	}

	void presentPattern(TableRow* row)
	{
		if (!n) return;
		double sum = 0.0;		
		for(unsigned int m = 0; m < numAttribs; m++)
		{
			double pattern_m = *((double*) row->fields[m]);
			double aux = (mu[m] - pattern_m) / s[m];
			sum += aux * aux;			
		}
		
		double log_CMF_value = - sum / numAttribs; //-0.5 * sum;
		//double xxx = exp(-0.5 * sum);
		CMF_value = exp(log_CMF_value);
		CCF_value = (exp(-0.5 * sum)) * n / sigmaProd; // CMF_value * n / sigmaProd;
	}

	unsigned int learnPattern(TableRow* row)
	{
		unsigned int ret = 0;
		if (label == -1)
		{
			sigmaProd = 1.0;
			for(unsigned int m = 0; m < numAttribs; m++)
			{
				double pattern_m = *((double*) row->fields[m]);
				mu[m] = pattern_m;
				v[m] = (pattern_m * pattern_m) + (gamma * gamma);
				s[m] = gamma;
				sigmaProd *= gamma;
			}

			ret = 1;
			numPatternsLearned++;
		}
		else
		{
			double aux1 = 1.0 / (double) (n + 1);
			double aux2 = 1.0 - aux1;
			sigmaProd = 1.0;
			for(unsigned int m = 0; m < numAttribs; m++)
			{
				double pattern_m = *((double*) row->fields[m]);
				mu[m] = (aux2 * mu[m]) + (aux1 * pattern_m);
				v[m] = (aux2 * v[m]) + (aux1 * (pattern_m * pattern_m));
				double sTemp = sqrt(v[m] - (mu[m] * mu[m]));
				//double temp = mu[m] - pattern_m;
				//double sTemp = sqrt(aux2 * s[m] * s[m] + aux1 * temp * temp);
				s[m] = sTemp;
				sigmaProd *= sTemp;
			}

			ret = 1;
			numPatternsLearned++;
		}
		n++;
		active++;
		return ret;
	}

public:
	unsigned int getCategoryType(){return 2;}
	void mutate(double severityFactor)
	{
		if (unif() > 0.50) 
		{
			for(unsigned int i = 0; i < numAttribs; i++)
			{
				double p1 = severityFactor * ltqnorm(unif()); // gaussian
				mu[i] += p1;
				if (mu[i] > 1.0) mu[i] = 1.0;
				if (mu[i] < 0.0) mu[i] = 0.0;
			}
		}
		else
		{
			sigmaProd = 1.0;
			for(unsigned int i = 0; i < numAttribs; i++)
			{
				double p2 = severityFactor * ltqnorm(unif()); // gaussian
				s[i] += p2;					  
				if (s[i] <= 0.0) s[i] = 0.0001;
				sigmaProd *= s[i];
			}
		}

		//n += severityFactor * ltqnorm(unif()); //this is useless
	}

	void initRandom(unsigned int numClasses)
	{
		sigmaProd = 1.0;
		for(unsigned int i = 0; i < numAttribs; i++)
		{
			double p1 = unif();
			mu[i] = p1;
			double p2 = 0.1 + 0.8 * unif();
			s[i] = p2;
			sigmaProd *= s[i]; 
		}
		n = rand();
		
		label = rand() % numClasses;
	}

	void getDescription(double* catf)
	{
		for(unsigned int i = 0; i < numAttribs; i++)
		{
			catf[i] = mu[i];
		}

		for(unsigned int i = 0; i < numAttribs; i++)
		{
			catf[i + numAttribs] = s[i];
		}

		for(unsigned int i = 0; i < numAttribs; i++)
		{
			catf[i + 2 * numAttribs] = v[i];
		}

		catf[numAttribs * 3] = gamma;
		catf[numAttribs * 3 + 1] = sigmaProd;
		catf[numAttribs * 3 + 2] = cf;
	}

	void setDescription(double* catf)
	{
		for(unsigned int i = 0; i < numAttribs; i++)
		{
			mu[i] = catf[i];
		}

		for(unsigned int i = 0; i < numAttribs; i++)
		{
			s[i] = catf[i + numAttribs];
		}

		for(unsigned int i = 0; i < numAttribs; i++)
		{
			v[i] = catf[i + 2 * numAttribs];
		}

		gamma = catf[numAttribs * 3];
		sigmaProd = catf[numAttribs * 3 + 1];
		cf = catf[numAttribs * 3 + 2];
	}
};

//////////////////////////////////////////////////////////////////////
//
// Class definition of a ARTMAP network. categoryType will determin 
// if this will be FAM, EAM or GAM network.
//
//////////////////////////////////////////////////////////////////////
class ARTMAP
{
	// Public variables
public:
	ListClass*		categories;				// List of categroies	
	double			baselineVigilance;		// Baseline Vigilance
	unsigned int	numAttribs;				// Number of attributes (features) of the classification problem
	unsigned int	maxEpochs;				// Max number of epochs
	FixedList*		classLst;				// List of possible class labels
	unsigned int	categoryType;			// 0 FAM, 1 EAM, 2 GAM

	unsigned int	validationErrors;		// Number of errors observed on the validation set
	double			validationPErr;			// Fraction of errors
	unsigned int	validationCt;			// Total number of validation points tested
	double			fitness;				// fitness value of network
	
	unsigned int	paretoFront;			// True if this is a member of pareto front

	unsigned int	maxCategories;			// Maximum number of categories allowed

private:
	// Local variables
	unsigned int	stop;					// Flag to stop training
	double			currentVigilance;		// Vigilance
	int				learnFlag;				// flag used to indicate that training caused a change in the network during last epoch
	
	// Inputs
	// Data properties
	unsigned int	classAttrib;			// Zero-based index of the class attribute column in the data

	// Training options
	
	double			initParam;				// For FAM, this is initial wij, for GAM this is gamma		
	double			choiceParam;			// Choice parameter used in FAM and EAM
	double			vigilanceSafety;		// This value is used to increase the vigilance during the training of ART networks. This is needed when a category is matching a pattern of incorrect label. See the FAM algorithm	literature for more information.			
	

	// Stoping parameters	
	double			minImproveErr;			// The percentage (represented as decimal, eg. 0.05 for 5%) that improvement needs to be observed from one epoch to the next. if no improvement is observed for a number of epochs, the ART stops training possibly before reaching maxEpochs to avoid over training.
	double			catTol;					// Tolerance of categories to encode wrong patterns. If other than zero, this would implement semi-supervised learning (see literature)

public:

	ARTMAP(
		// Data properties
		unsigned int	_numAttribs,		// Number of attributes (features) of the classification problem 
		unsigned int	_classAttrib,		// Zero-based index of the class attribute column in the data	

		// Training options
		unsigned int	_categoryType,		// 0 FAM, 1 EAM, 2 GAM
		double			_initParam,			// For FAM, this is initial wij, for GAM this is gamma		
		double			_choiceParam,		// Choice parameter used in FAM and EAM
		double			_baselineVigilance,	// Baseline Vigilance
		double			_vigilanceSafety,	// This value is used to increase the vigilance during the training of ART networks. This is needed when a category is matching a pattern of incorrect label. See the FAM algorithm	literature for more information.
		double			_catTol,			// Tolerance of categories to encode wrong patterns. If other than zero, this would implement semi-supervised learning (see literature)
		unsigned int	_maxCategories,		// Maximum number of categories allowed

		// Stoping parameters
		unsigned int	_maxEpochs,			// Max number of epochs
		double			_minImproveErr		// The percentage (represented as decimal, eg. 0.05 for 5%) that improvement needs to be observed from one epoch to the next. if no improvement is observed for a number of epochs, the ART stops training possibly before reaching maxEpochs to avoid over training.
		);
	ARTMAP(ARTMAP* nn);
	~ARTMAP();
	int Train(TableClass* inputData, unsigned int offset);
	int Test(TableClass* testData, unsigned int maxTest, unsigned int* numTested, unsigned int score, unsigned int maxErr, unsigned int offset, char* filename);	
	CategoryBase* CopyCategory(CategoryBase* catSource);	
	int Store(char* fileName);
private:

	void DeleteAllCategories();
	CategoryBase* AddCategory();	
	double presentPattern(TableRow* row);
	int evaluateStoppingCriteria();
	void setLabel(CategoryBase* cat, TableRow* row);
	unsigned int checkLabel(CategoryBase* cat, TableRow* row);
};