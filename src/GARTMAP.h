//////////////////////////////////////////////////////////////////////
//
// This code was developed by Assem Kaylani (akaylani@gmail.com) 
// and Dr. Michael Georgiopoulos (michaelg@ucf.edu).
//
// GARTMAP class definition. 
// Genetic evolution of ART (Fuzzy ARTMAP, Ellipsoidal ARTMAP and 
// Gaussian ARTMAP using a multi-objective evolutionary approach.
// Please see the following publication for detail description. 
// It was the intent that the code implemented here would match
// exactly the algorithm described in this paper:
//
// [1] A. Kaylani, M. Georgiopoulos, M. Mollaghasemi, 
// G. C. Anagnostopoulos, C. Sentelle, M. Zhong, "An Adaptive 
// Multi-Objective Approach to Evolving ART Architectures," 
// IEEE Transactions on Neural Networks, Vol. 21, 
// Issue 4, pp. 529-550, 2010.
//
//////////////////////////////////////////////////////////////////////

#include "ARTMAP.h"
#include <memory.h>
#include <string.h>
#include <time.h>

class GARTMAP
{
public:
	ListClass*		archive;				// Archive. A list structure used to store the archive of pareto-optimal networks. Denoted as "A" in [1].

private:
	FixedList*		population;				// Population. A list structure used to store the evolved population at every generation. Denoted "P" in [1].
	FixedList*		classLst;				// A list of the call labels in the classifcation problem at hand.

	unsigned int	categoryType;			// 0 FAM, 1 EAM, 2 GAM
	unsigned int	maxGen;					// Maximum number of generations
	double			pPrune;					// Probably of applying the prune operator on a given network
	double			pCrossOver;				// Probabity of applying crossover operation to produce children. 
	double			pMutate;				// Probably of applying the mutate operator on a given network
	double			mutateFactor;			// A multiplier that controls mutation severity. A good value was found to be 0.05. Performance was not found to be very sensitive over small variations.
	unsigned int	minCategories;			// Minimum number of categories allowed in a network.
	unsigned int	maxCategories;			// Maximum number of categories allowed in a network.
	unsigned int	numClasses;				// Number of classes (labels) in the classification problem
	unsigned int	targetAttrib;			// The Zero-based index of the column that contains the class label for the training and validation data retrieved using the above SQL statements 
	unsigned int	selectionSetSize;		// Selection set size in tournament selection when determining crossover candidates
	unsigned int	skipClassInclusion;		// ClassInClusion condition is enforcing at least one category of every class label to be present in the network. A non-zero value skips this condition and allow formation of networks (using crossover or pruning) that might not include categories for all classes.

	// Stopping Criteria
	unsigned int	gensNoImprvToStop;		// Number of generations to allow to pass with no change to the Pareto set before search is called off.
	unsigned int	stopEvolution;			// A flag indicating that search should be called off

	double			pErrCi;					// Minimum difference in prediction error for two networks to be considered having different accuracy

	// Cross Validation statistics for 
	// calculating confidence factors (CF)
	// of categories
	unsigned int	vxMaxErr;
	unsigned int	vxMaxTest;
	unsigned int	vxOffset;
	double			vxBestErr;

	double*			max_pcc; 
	double*			max_ns;
	double			cfWeight;

public:
	GARTMAP(){}
	GARTMAP(
		unsigned int	_targetAttrib,		// The Zero-based index of the column that contains the class label for the training and validation data retrieved using the above SQL statements

		// ART Options					
		unsigned int	_categoryType,		// 0 FAM, 1 EAM, 2 GAM
		double			_initParam,			// For FAM, this is initial wij, for GAM this is gamma		
		double			_choiceParam,		// Choice parameter used in FAM and EAM
		double			_catErrTol,			// Used in ART training. Tolerance of categories to encode wrong patterns. If other than zero, this would implement semi-supervised learning (see literature [4]).
		double			_vigilanceSafety,	// This value is used to increase the vigilance during the training of ART networks. This is needed when a category is matching a pattern of incorrect label. See the FAM algorithm	literature for more information.				
		unsigned int	_maxCategories,		// The maximum categories (nodes) the ART network is allowed to create, after which training will stop.
		unsigned int	_maxEpochs,			// Number of epochs used in traing initial population of ART networks
		double			_minImproveErr,		// Not used. The percentage (represented as decimal, eg. 0.05 for 5%) that improvement needs to be observed from one epoch to the next. if no improvement is observed for a number of epochs, the ART stops training possibly before reaching maxEpochs to avoid over training. This mechanism is not used from GARTMAP and therefore this paramater is ignored.

		// GA Options
		unsigned int	_maxGen,			// Maximum number of generations
		unsigned int	_popSize,			// Population size of the GA
		double			_vigMin,			// Minumum value of baseline vigilance used to train initation population
		double			_vigMax,			// Maximum value of baseline vigilance. Values are varied in popSize equal increments between vigMin and vigMax.
		double			_pPrune,			// Probably of applying the prune operator on a given network
		double			_pMutate,			// Probably of applying the mutate operator on a given network
		unsigned int	_selectionSetSize	// Selection set size in tournament selection when determining crossover candidates
		);	

	~GARTMAP();
	int Train(TableClass* inputData, TableClass* validationData);
	int Store(char* storagePath);
	int ParetoCompare(ARTMAP* nn1, ARTMAP* nn2);

private:
	int Evolve(TableClass* validationData, unsigned int gens);
	int EvaluatePopulation(TableClass* validationData);
	int CalculateConfidenceFactor();
	int ArchivePareto();
	int CompareNetErr(ARTMAP* nn1, ARTMAP* nn2);
	int AssignFitness();
	int Reproduce();
	int MutateNet(ARTMAP* nn);
	int PruneNet(ARTMAP* nn);
	int CheckClassInclusion(ListClass* categories, int catToDel);
	int CheckClassInclusion(ListClass* categories1, ListClass* categories2, unsigned int split1, unsigned int split2);

public:
	ARTMAP* GetARTMAP(unsigned int i);
};