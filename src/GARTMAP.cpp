//////////////////////////////////////////////////////////////////////
//
// This code was developed by Assem Kaylani (akaylani@gmail.com) 
// and Dr. Michael Georgiopoulos (michaelg@ucf.edu).
//
// GARTMAP class definition. 
// 
//////////////////////////////////////////////////////////////////////

#include "GARTMAP.h"

//////////////////////////////////////////////////////////////////////
// GARTMAP::GARTMAP
// 
// Constructor. Initialize GARTMAP object and initialize the initial
// population of ART networks
//
//////////////////////////////////////////////////////////////////////
GARTMAP::GARTMAP(
		unsigned int	_targetAttrib,		// The Zero-based index of the column that contains the class label for the training and validation data retrieved using the above SQL statements

		// ART Options					
		unsigned int	_categoryType,		// 0 FAM, 1 EAM, 2 GAM
		double			_initParam,			// For FAM, this is initial wij, for GAM this is gamma		
		double			_choiceParam,		// Choice parameter used in FAM and EAM
		double			_catTol,			// Used in ART training. Tolerance of categories to encode wrong patterns. If other than zero, this would implement semi-supervised learning (see literature [4]).
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
		)
{
	population = new FixedList(_popSize);
	archive = new ListClass();
	classLst = 0;

	categoryType = _categoryType;
	maxGen = _maxGen;			
	pPrune = _pPrune;
	pCrossOver = 1.0;
	pMutate = _pMutate;
	mutateFactor = 0.05;
	minCategories = 2;	//_minCategories;		
	maxCategories = _maxCategories;				
	numClasses = 2;
	targetAttrib = _targetAttrib;
	selectionSetSize = _selectionSetSize;
	skipClassInclusion = 0;

	// stopping criteria
	gensNoImprvToStop = 10;
	stopEvolution = 0;	

	pErrCi = 0.005; // default chosen to be 0.005

	vxMaxErr = 0;
	vxMaxTest = 0;
	vxOffset = 1;	
	
	vxBestErr = 1.0;

	max_pcc = 0;
	max_ns = 0;
	cfWeight = 0.5;

	// create the initial population
	for(unsigned int i = 0; i < _popSize; i++)
	{
		unsigned int catType = categoryType;
		if (categoryType == 3) catType = (i % 2);

		ARTMAP* nn = new ARTMAP(
			targetAttrib,
			targetAttrib,				
			catType,
			_initParam,
			_choiceParam,
			_vigMin + (i * (_vigMax - _vigMin) / (_popSize - 1)),
			_vigilanceSafety,
			_catTol,
			_maxCategories,
			_maxEpochs, //
			_minImproveErr);

		population->AddItem((void*) nn);
	}
}

//////////////////////////////////////////////////////////////////////
// GARTMAP::~GARTMAP
// 
// Free resources. 
//
//////////////////////////////////////////////////////////////////////
GARTMAP::~GARTMAP()
{
	if (population)
	{
		for(unsigned int i = 0; i < population->nCount; i++)
		{
			ARTMAP* nn = (ARTMAP*) population->item[i];
			delete nn;
		}

		delete population;
		population = 0;
	}

	if (archive)
	{
		for(unsigned int i = 0; i < archive->Count(); i++)
		{
			ARTMAP* nn = (ARTMAP*) archive->item[i];
			delete nn;
		}

		delete archive;
		archive = 0;
	}

	
	if (classLst) delete classLst;

	if (max_pcc) delete max_pcc;
	if (max_ns) delete max_ns;
}

//////////////////////////////////////////////////////////////////////
// GARTMAP::Train
// 
// Training start with initialization of initial population of ART 
// networks using ART rules (Fuzzy ARTMAP, Ellipsoidal ARTMAP or 
// Gaussian ARTMAP). Then this initial population is evolved using
// a multi-objective evolutionary algorithm. See [1] for more details.
// See Fig 4. in [1] for the high level Pseudo Code of MO-GART 
// Algorithm
//
//////////////////////////////////////////////////////////////////////
int GARTMAP::Train(TableClass* inputData, TableClass* validationData)
{
	// store the list of classes in this dataseet
	classLst = inputData->GetAttribSpaceCopy(targetAttrib);
	numClasses = classLst->nCount;
	minCategories = numClasses;

	// initialize space where intermediate calculation of confidence factors (CF) 
	max_pcc = (double*) malloc(sizeof(double) * numClasses * population->nCount);
	max_ns = (double*) malloc(sizeof(double) * numClasses * population->nCount);
	
	// Make sure that rows in the data are to be presented in a random order
	inputData->MakeRowsArrayRandomized();
	validationData->MakeRowsArrayRandomized();

	// Train initial population (P) using the usual FAM, EAM or GAM rules
	for (unsigned int i = 0; i < population->nCount; i++)
	{
		ARTMAP* nn = (ARTMAP*) population->item[i];					
		nn->classLst = classLst;
		nn->Train(inputData, i + 1);
		nn->baselineVigilance = 0.0;
	}

	// Evolve!
	unsigned int gens = Evolve(validationData, maxGen);

	return gens;
}

//////////////////////////////////////////////////////////////////////
// GARTMAP::Train
// 
// the initial population is evolved using a multi-objective 
// evolutionary algorithm. See [1] for more details.
// See Fig 4. in [1] for the high level Pseudo Code of MO-GART 
// Algorithm. 
//
//////////////////////////////////////////////////////////////////////
int GARTMAP::Evolve(TableClass* validationData, unsigned int gens)
{	
	stopEvolution = 0;	
	vxMaxTest = 0;
	vxMaxErr = 0;		
	vxOffset = 1;		
	
	unsigned int noImprovementCt = 0;
	unsigned int gen = 0;
	
	// Loop for a maximum of gens generations.
	for(gen = 0; gen <= gens; gen++) 
	{	
		// if not imporvement is observed for a number of generations
		if (noImprovementCt >= gensNoImprvToStop) 
		{
			// set stopping flag
			stopEvolution = 1;
			vxMaxTest = 0; 
		}

		// Evaluate population P
		EvaluatePopulation(validationData);

		// Update the pareto archive (A)
		int paretoChange = ArchivePareto();

		// Update number of consqutive times no improvement to the pareto archive is made
		if (paretoChange) noImprovementCt = 0; else noImprovementCt++;

		// if stopping criteria met, stop evolution
		if (stopEvolution) break;

		// Selection and Reproduction
		this->Reproduce();
	}

	return gen;
}

//////////////////////////////////////////////////////////////////////
// GARTMAP::EvaluatePopulation
// 
// The population is evaluated. This is done by presenting the 
// validtion set and measuring the accuracy of prediction of each 
// network in the population. The second objective, network size, is
// simply the number of categories in the network. After evaluation
// the archive is updating to include the new set of non-dominated
// networks (of high accurace and small size).
//
//////////////////////////////////////////////////////////////////////
int GARTMAP::EvaluatePopulation(TableClass* validationData)
{		
	vxBestErr = 1.0;

	// Evaluate each network in the population by presenting to it
	// the validation data.
	for(unsigned int i = 0; i < population->nCount; i++)
	{
		ARTMAP* nn = (ARTMAP*) population->item[i];
		nn->Test(validationData, vxMaxTest, 0, 1, vxMaxErr, vxOffset, 0);
		if (vxBestErr > nn->validationPErr) vxBestErr = nn->validationPErr;
	}

	// Calculate conf factors for categories.
	CalculateConfidenceFactor();
	
	return 0;
}	

//////////////////////////////////////////////////////////////////////
// GARTMAP::CalculateConfidenceFactor
// 
// Calculate the confidence factor of categories
//
//////////////////////////////////////////////////////////////////////
int GARTMAP::CalculateConfidenceFactor()
{
	// Initialize memory space that will hold category stats used to calculate CF
	memset(max_pcc, 0, sizeof(double) * numClasses * population->nCount);
	memset(max_ns, 0, sizeof(double) * numClasses * population->nCount);

	// Calculate category stats to calculate CF
	for(unsigned int i = 0; i < population->nCount; i++)
	{
		ARTMAP* nn = (ARTMAP*) population->item[i];
		for(unsigned int j = 0; j < nn->categories->nCount; j++)
		{
			CategoryBase* cat = (CategoryBase*) nn->categories->item[j];

			unsigned int classIndex = cat->label;

			unsigned int ii = i * this->numClasses + classIndex;
			if (cat->active > max_ns[ii]) max_ns[ii] = cat->active;
			double pcc = cat->numPatternsLearned / (cat->active + 0.0001);
			if (pcc > max_pcc[ii]) max_pcc[ii] = pcc;
		}

		for (unsigned int j = 0; j < nn->categories->nCount; j++)
		{
			CategoryBase* cat = (CategoryBase*) nn->categories->item[j];
			unsigned int classIndex = cat->label;
			unsigned int ii = i * this->numClasses + classIndex;

			// assign each categroy confidence factor according to calculation above
			// this calculation is described in Eq. 4 in [1].
			cat->cf = cfWeight * (cat->numPatternsLearned / (max_pcc[ii] * cat->active + 0.0001)) + (1 - cfWeight) * (cat->active / (max_ns[ii] + 0.00001));
		}
	}

	return 0;
}

//////////////////////////////////////////////////////////////////////
// GARTMAP::ArchivePareto
// 
// For each solution (network) in the population P, compare it with
// archive solution. if this solution in non-dominated, add it to the 
// archive. Also, remove from the archive any solution that is 
// dominated by this new solution. Returns non-zero value if the 
// archive A was changed in this operation.
//
//////////////////////////////////////////////////////////////////////
int GARTMAP::ArchivePareto()
{
	unsigned int ret = 0;
	
	// loop for every solution/network in population P
	for(unsigned int i = 0; i < population->nCount; i++)
	{
		ARTMAP* nn = (ARTMAP*) population->item[i];
		nn->paretoFront = 1;

		// search archive solutions
		for(int j = archive->Count() - 1; j >= 0; j--)
		{
			ARTMAP* nn1 = (ARTMAP*) archive->GetNth(j);

			// >= or  >
			int cres = ParetoCompare(nn, nn1);

			if ((cres == 0) || (cres == 2))			
			{
				//solution from P is dominated by solution in the archive. Mark it as non pareto.
				nn->paretoFront = 0;
				break;
			}

			if (cres == 1)			
			{
				// archive solution is dominated by solution from P. Remove the archive solution.
				archive->Remove(j);
				delete nn1;

				ret = 1; // mark a change in archive: a solution was removed
			}
		}

		if (nn->paretoFront)
		{
			// no solution in archive A was found to dominate solution from P. copy this solution from P into archive A
			// copy nn to archive
			ARTMAP* cpynn = new ARTMAP(nn);
			archive->Add(cpynn);
			cpynn->categories->Count();

			nn->paretoFront = 0;
			ret = 1; // mark a change in archive: a solution was added.
		}
	}

	// calling count() improves performance
	archive->Count();

	return ret;
}

//////////////////////////////////////////////////////////////////////
// GARTMAP::ParetoCompare
// 
// Compares two solutions (networks) based on size (small is good) and 
// validation error (small is good). 
//
//////////////////////////////////////////////////////////////////////
int GARTMAP::ParetoCompare(ARTMAP* nn1, ARTMAP* nn2)
{
	int nerr = CompareNetErr(nn1, nn2);
	if ((nn1->categories->nCount == nn2->categories->nCount) && (nerr == 0)) return 0; //equal err, equal size
	if ((nn1->categories->nCount <= nn2->categories->nCount) && ((nerr == 1) || (nerr == 0))) return 1; // network 1 dominates
	if ((nn1->categories->nCount >= nn2->categories->nCount) && ((nerr == 2) || (nerr == 0))) return 2; // network 2 dominates
	return -1; //no domination.
}

//////////////////////////////////////////////////////////////////////
// GARTMAP::CompareNetErr
// 
// Compares two solutions (networks) errors
//
//////////////////////////////////////////////////////////////////////
int GARTMAP::CompareNetErr(ARTMAP* nn1, ARTMAP* nn2)
{
	//if (abs(nn1->validationPErr - nn2->validationPErr) < pErrCi) return 0; 
	if (nn1->validationPErr < nn2->validationPErr) return 1;
	if (nn1->validationPErr > nn2->validationPErr) return 2;
	return 0;
}

//////////////////////////////////////////////////////////////////////
// GARTMAP::Reproduce
// 
// Reproduction involves Selecting parents and reproducing children. 
// The selection process creates a temporary population P', where 
// the parent chromosomes/solutions/networks used to create the next
// generation are selected. The chromosomes in the archive A and 
// population P are assigned fitness values based on dominance 
// relationship (see [1]). 
//
// The parents are then chosen using a deterministic binary 
// tournament selection with replacement, as follows: For
// each parent, randomly select two chromosomes from the combined 
// set of A and P, and choose, the chromosome with the smallest 
// fitness value. Boundary solutions, which are networks with smallest 
// error rate and smallest size, are ensured to be copied in 
// the set of parents P'.
//
// Once the selection step determines the parents, reproduction 
// operators are used to create individuals for the next
// generation. The two well-known operators for reproduction 
// in GAs are crossover and mutation. In this work, in
// addition to crossover, two mutation-based operators are proposed. 
// The first is referred to as the Mutation operator and it performs 
// Gaussian mutations on the weights of the categories of the 
// ARTMAP network. The second operator, referred to as the Prune 
// operator, prunes a network by deleting a number of categories 
// from that network (structural mutation).
//
//////////////////////////////////////////////////////////////////////
int GARTMAP::Reproduce()
{
	// Calculate fitness as described in [1]. The calculated values
	// will be used in the reproduction phase to select parents
	AssignFitness();

	unsigned int bestErrIdx = 0;
	unsigned int bestSizeIdx = 0;	

	unsigned int numCand = population->Count() + archive->Count();
	//unsigned int MAX_SEL_POOL = 30;
	//numCand = MAX_SEL_POOL;
	//if (numCand < archive->nCount) numCand = archive->nCount;

	// list of parents
	FixedList* catLists = new FixedList(numCand);
	double*	tempFit = (double*) malloc(numCand * sizeof(double));

	// Add archive solutions to list of parents
	// find the most accurate and the smallest network
	for(unsigned int i = 0; i < archive->nCount; i++)
	{
		ARTMAP* nn = (ARTMAP*) archive->item[i];
		catLists->AddItem(nn->categories);
		tempFit[i] = nn->fitness;

		if (CompareNetErr(nn, ((ARTMAP*) archive->item[bestErrIdx])) == 1) bestErrIdx = i;
		if (nn->categories->nCount < ((ARTMAP*) archive->item[bestSizeIdx])->categories->nCount) bestSizeIdx = i;
	}

	// pin boundary solutions (the most accurate and the smallest network)
	// give these two solutions highest possible fitness
	tempFit[bestErrIdx] = 0;
	tempFit[bestSizeIdx] = 0;

	for(unsigned int i = 0; i < population->nCount; i++)
	{
		ARTMAP* nn = (ARTMAP*) population->item[i];
		catLists->AddItem(nn->categories);
		tempFit[i + archive->nCount] = nn->fitness;
	}

	//int numFromPop = MAX_SEL_POOL - archive->nCount;
	//for(int i = 0; i < numFromPop; i++)
	//{						
	//	ARTMAP* nn1 = (ARTMAP*) population->item[0];

	//	for(unsigned int j = 0; j < population->nCount; j++)
	//	{
	//		ARTMAP* nn = (ARTMAP*) population->item[j];

	//		if ((nn->fitness > 0.99) && (nn->fitness < nn1->fitness))
	//		{
	//			nn1 = nn;
	//		}
	//	}
	//	
	//	if (nn1->fitness < 999999)
	//	{
	//		catLists->AddItem(nn1->categories);
	//		tempFit[i + archive->nCount] = nn1->fitness;
	//		nn1->fitness = 1000000;
	//	}
	//}

	// Reproduce children for next population
	for(unsigned int i = 0; i < population->nCount; i++)
	{
		ARTMAP* nn = (ARTMAP*) population->item[i];
		nn->categories = new ListClass();

		unsigned int doCrossOver = unif() < pCrossOver;
		unsigned int doPrune = unif() < pPrune;
		unsigned int doMutate = unif() < pMutate; 

		unsigned int randomID1 = catLists->nCount - 1;
		unsigned int randomID2 = catLists->nCount - 1;

		double fit1 = 10000000;
		double fit2 = 10000000;

		// select 2 parents
		if (selectionSetSize > 0)
		{
			// Choose 2 networks, each from a set of selectionSetSize that has the best fitness.			
			for (unsigned int j = 0; j < selectionSetSize; j++)
			{
				unsigned int rn = rand() % catLists->nCount;
				if (tempFit[rn] <= fit1) {randomID1 = rn; fit1 = tempFit[rn];}

				rn = rand() % catLists->nCount;
				if (tempFit[rn] <= fit2) {randomID2 = rn; fit2 = tempFit[rn];}
				
				if (randomID1 == randomID2) randomID2 = rand() % catLists->nCount;
			}	
		}
		else
		{	
			// if selectionSetSize = 0 then use Fitness proportionate selection
			double totalFitness = 0;
			for (unsigned int i = 0; i < catLists->nCount; i++)
				totalFitness +=  (1.0 - tempFit[i]);

			double rn = unif();
			double cumFit = 0;
			for (unsigned int i = 0; i < catLists->nCount; i++)
			{
				cumFit += ((1.0 - tempFit[i]) / totalFitness);
				if (rn < cumFit) 
				{
					randomID1 = i;
					break;
				}
			}

			for (unsigned int k = 0; k < 3; k++)
			{
				rn = unif();
				cumFit = 0;
				for (unsigned int i = 0; i < catLists->nCount; i++)
				{
					cumFit += ((1.0 - tempFit[i]) / totalFitness);
					if (rn < cumFit) 
					{
						randomID2 = i;
						break;
					}
				}

				if (randomID1 != randomID2) break;
			}
		}


		// make first individual in parent list is the most accurate seen so far
		if (i == 0)
		{
			randomID1 = bestErrIdx;
			doCrossOver = 0;
		}

		// make second individual in parent list is the smallest seen so far
		if (i == 1)
		{
			randomID1 = bestSizeIdx;
			doCrossOver = 0;
		}

		// make the 3rd as the crossover of best and smallest
		if (i == 2)
		{
			randomID1 = bestErrIdx;
			randomID2 = bestSizeIdx;
		}


		// Get the selected sets of categories for cross-over
		ListClass* cats1 = (ListClass*) catLists->item[randomID1];
		ListClass* cats2 = (ListClass*) catLists->item[randomID2];

		// Get two random split points
		unsigned int split1 = cats1->nCount;
		unsigned int split2 = cats2->nCount;


		if (doCrossOver) 
		{
			split1 = rand() % cats1->nCount;
			split2 = rand() % cats2->nCount;
			// make sure number of categories is within limit
			for (unsigned int i = 0; i < 300; i++)
			{
				cats1->Count();
				unsigned int newNumCat = split1 + cats2->Count() - split2;
				if ((newNumCat > maxCategories) || //(newNumCat < minCategories) || 
					(newNumCat < numClasses) ||
					(CheckClassInclusion(cats1, cats2, split1, split2) == 0))
				{
					split1 = rand() % cats1->nCount;
					split2 = rand() % cats2->nCount;
				}
				else
					break;
			}
		}

		// Produce child: We choose to apply cross over and then 
		// Prune and then mutate operators to produce children

		// Apply the cross over genetic operator
		for (unsigned int j = 0; j < split1; j++)
			nn->CopyCategory((CategoryBase*) cats1->item[j]);

		for (unsigned int j = split2; j < cats2->Count(); j++)
			nn->CopyCategory((CategoryBase*) cats2->item[j]);

		// Apply the Prune genetic operator
		if (doPrune) PruneNet(nn);

		// Apply the Mutation genetic operator
		if (doMutate) MutateNet(nn);

	}

	// Delete old generation
	for(unsigned int i = archive->nCount; i < catLists->nCount; i++)
	{
		ListClass* categories = (ListClass*) catLists->item[i];
		if (categories) 
		{
			for(unsigned int j = 0; j < categories->Count(); j++)
			{
				CategoryBase* cat = (CategoryBase*) categories->item[j];
				cat->freeResource();
				delete cat;
			}
			delete (categories);
			categories = 0;
		}
	}
	delete catLists;
	free(tempFit);

	return 0;
}

//////////////////////////////////////////////////////////////////////
// GARTMAP::AssignFitness
// 
// Assign fitness to each solution in population and archive. See
// [1], equation 3 for more information.
// The chromosomes in the archive A and population P are assigned 
// fitness values based on dominance relationship, as suggested 
// in SPEA2 (see [54] in [1]). In this scheme, each individual is 
// assigned a strength value equal to the number of solutions it 
// dominates. After that, a raw fitness, R(x), is assigned for each 
// individual as the sum of the strengths of all its dominators in 
// both A and P. The raw fitness is then adjusted as follows. For
// each individual, x, the distance, in objective space, to the 
// k-th nearest neighbor is found and denoted as sig(x). The value 
// of k is chosen to be the square root of the sum of the size of 
// the archive and population. The fitness of each individual is 
// then calculated using equation 3. Fitness = R(x) + (1/(sig + 2))
//
//////////////////////////////////////////////////////////////////////
int GARTMAP::AssignFitness()
{
	double* strengths = (double*) malloc((archive->nCount + population->nCount) * sizeof(double));
	double* rowFitness = (double*) malloc(population->nCount * sizeof(double));
	double* distances = (double*) malloc((archive->nCount + population->nCount) * sizeof(double));
	unsigned int maxSize = 0;

	// k is chosen to be the square root of the sum of the size of 
	// the archive and population
	unsigned int k = (unsigned int) sqrt((double) archive->nCount + population->nCount);

	// Each individual is assigned a strength value equal to the number of solutions it dominates.
	// Calculate strength for archive A solutions
	for(unsigned int i = 0; i < archive->Count(); i++)
	{
		ARTMAP* nna = (ARTMAP*) archive->item[i];
		unsigned int numDom = 0;
		for(unsigned int j = 0; j < population->nCount; j++)
		{
			ARTMAP* nnp = (ARTMAP*) population->item[j];
			if (ParetoCompare(nna, nnp) == 1) numDom++ ;
		}

		strengths[i] = numDom;

		if (maxSize < nna->categories->nCount) maxSize = nna->categories->nCount;

		nna->fitness = 0;
	}

	// calculate strength for population P solutions
	for(unsigned int i = 0; i < population->Count(); i++)
	{
		ARTMAP* nna = (ARTMAP*) population->item[i];
		unsigned int numDom = 0;
		for(unsigned int j = 0; j < population->nCount; j++)
		{
			ARTMAP* nnp = (ARTMAP*) population->item[j];
			if (ParetoCompare(nna, nnp) == 1) numDom++ ;
		}

		strengths[i + archive->nCount] = numDom;

		if (maxSize < nna->categories->nCount) maxSize = nna->categories->nCount;
	}

	// Raw fitness, R(x), is assigned for each individual as the 
	// sum of the strengths of all its dominators in 
	// both A and P
	for(unsigned int i = 0; i < population->nCount; i++)
	{
		ARTMAP* nnp = (ARTMAP*) population->item[i];
		rowFitness[i] = 0.0;
		for(unsigned int j = 0; j < archive->nCount; j++)
		{
			ARTMAP* nna = (ARTMAP*) archive->item[j];
			if (ParetoCompare(nna, nnp) == 1) rowFitness[i] += strengths[j];
		}
	}
	for(unsigned int i = 0; i < population->nCount; i++)
	{
		ARTMAP* nnp = (ARTMAP*) population->item[i];			
		for(unsigned int j = 0; j < population->nCount; j++)
		{
			ARTMAP* nnpp = (ARTMAP*) population->item[j];
			if (ParetoCompare(nnpp, nnp) == 1) rowFitness[i] += strengths[j + archive->nCount];
		}

		nnp->fitness = rowFitness[i];
	}

	// The raw fitness is then adjusted as follows. For
	// each individual, x, the distance, in objective space, to the 
	// k-th nearest neighbor is found and denoted as sig(x). The value 
	// of k is chosen to be the square root of the sum of the size of 
	// the archive and population. The fitness of each individual is 
	// then calculated using equation 3. Fitness = R(x) + (1/(sig + 2))
	for(unsigned int i = 0; i < (archive->nCount); i++)
	{
		ARTMAP* nn = (ARTMAP*) archive->item[i];
		for(unsigned int j = 0; j < (archive->nCount); j++)
		{
			ARTMAP* nn1 = (ARTMAP*) archive->item[j];
			double derr = nn->validationPErr - nn1->validationPErr;
			double dsize1 = ((double)nn->categories->nCount) - ((double)nn1->categories->nCount);
			double dsize = dsize1 / ((double) maxSize); // normalize objective

			double dist = sqrt(derr*derr + dsize*dsize);
			
			// use insetion sorting to keep the distances array sorted
			unsigned int k = 0;
			for (k = j; k > 0 && dist < distances[k - 1]; k--)
				distances[k] = distances[k - 1];

			distances[k] = dist;
		}

		for(unsigned int j = archive->nCount; j < (population->nCount + archive->nCount); j++)
		{
			ARTMAP* nn1 = (ARTMAP*) population->item[j - archive->nCount];
			double derr = nn->validationPErr - nn1->validationPErr;
			double dsize1 = ((double)nn->categories->nCount) - ((double)nn1->categories->nCount);
			double dsize = dsize1 / ((double) maxSize);

			double dist = sqrt(derr*derr + dsize*dsize);

			// use insetion sorting to keep the distances array sorted
			unsigned int k = 0;
			for (k = j; k > 0 && dist < distances[k - 1]; k--)
				distances[k] = distances[k - 1];

			distances[k] = dist;
		}

		// k-th nearest neighbor
		double sig = distances[k];

		// Assign fitness to archive solution. 
		nn->fitness += (1 / (sig + 2));
	}

	for(unsigned int i = 0; i < population->nCount; i++)
	{
		ARTMAP* nn = (ARTMAP*) population->item[i];
		for(unsigned int j = 0; j < (archive->nCount); j++)
		{
			ARTMAP* nn1 = (ARTMAP*) archive->item[j];
			double derr = nn->validationPErr - nn1->validationPErr;
			double dsize1 = ((double)nn->categories->nCount) - ((double)nn1->categories->nCount);
			double dsize = dsize1 / ((double) maxSize);

			double dist = sqrt(derr*derr + dsize*dsize);

			// use insetion sorting to keep the distances array sorted
			unsigned int k = 0;
			for (k = j; k > 0 && dist < distances[k - 1]; k--)
				distances[k] = distances[k - 1];

			distances[k] = dist;
		}

		for(unsigned int j = archive->nCount; j < (population->nCount + archive->nCount); j++)
		{
			ARTMAP* nn1 = (ARTMAP*) population->item[j - archive->nCount];
			double derr = nn->validationPErr - nn1->validationPErr;
			double dsize1 = ((double)nn->categories->nCount) - ((double)nn1->categories->nCount);
			double dsize = dsize1 / ((double) maxSize);

			double dist = sqrt(derr*derr + dsize*dsize);

			// use insetion sorting to keep the distances array sorted
			unsigned int k = 0;
			for (k = j; k > 0 && dist < distances[k - 1]; k--)
				distances[k] = distances[k - 1];

			distances[k] = dist;
		}

		// k-th nearest neighbor
		double sig = distances[k];

		// Assign fitness to solution/network. See Eq. 3 in [1]
		nn->fitness += (1 / (sig + 2));
	}

	free(distances);
	free(rowFitness);
	free(strengths);
	
	return 0;
}

//////////////////////////////////////////////////////////////////////
// GARTMAP::MutateNet
// 
// Apply the mutation operator on a solution/network. The mutation 
// operator is applied on every category in the network. See [1]
// for more details
//
//////////////////////////////////////////////////////////////////////
int GARTMAP::MutateNet(ARTMAP* nn)
{
	unsigned int numCat = nn->categories->Count();
	for(unsigned int c = 0; c < numCat; c++)
	{
		CategoryBase* cat = (CategoryBase*) nn->categories->item[c];
		double sevf = (1 - cat->cf) * mutateFactor;
		cat->mutate(sevf);
	}

	return 0;
}

//////////////////////////////////////////////////////////////////////
// GARTMAP::PruneNet
// 
// Apply the Prune operator on a solution/network. The prune 
// operator is applied on every category in the network. See [1]
// for more details
//
//////////////////////////////////////////////////////////////////////
int GARTMAP::PruneNet(ARTMAP* nn)
{
	unsigned int numCat = nn->categories->Count();
	
	for (int j = numCat - 1; j >= 0; j--)
	{
		double t = unif();
		CategoryBase* cat = (CategoryBase*) nn->categories->GetNth(j);
		if (cat->cf < t)
		{
			if (CheckClassInclusion(nn->categories, (int) j))
			{
				cat->freeResource();
				nn->categories->Remove(j);
				delete(cat);					
			}
		}
	}

	nn->categories->Count();

	return 0;
}

//////////////////////////////////////////////////////////////////////
// GARTMAP::CheckClassInclusion
// 
// Returns non-zero if the deletetion of a category will result in  
// a network that will still contain all class labels
//
//////////////////////////////////////////////////////////////////////
int GARTMAP::CheckClassInclusion(ListClass* categories, int catToDel)
{
	if (skipClassInclusion) return 1;

	FixedList* tempClassLst = new FixedList(numClasses);
	for(unsigned int i = 0; i < categories->Count(); i++)
	{
		if (catToDel == (int) i) continue;

		CategoryBase* cat = (CategoryBase*) categories->item[i];

		unsigned int noMch = 1;
		for(unsigned int j = 0; j < tempClassLst->nCount; j++)
			if (cat->label == (int) (tempClassLst->item[j])) noMch = 0;
		
		if (noMch) tempClassLst->AddItem((void*) cat->label);

		if (tempClassLst->nCount == numClasses) break;
	}

	unsigned int ret = 0;
	if (tempClassLst->nCount == numClasses) ret = 1;
	delete tempClassLst;
	return ret;
}

//////////////////////////////////////////////////////////////////////
// GARTMAP::CheckClassInclusion
// 
// Returns non-zero if the crossover of two category sets will result 
// in a network that will contain all class labels
//
//////////////////////////////////////////////////////////////////////
int GARTMAP::CheckClassInclusion(ListClass* categories1, ListClass* categories2, unsigned int split1, unsigned int split2)
{
	if (skipClassInclusion) return 1;

	FixedList* tempClassLst = new FixedList(numClasses);
	for(unsigned int i = 0; i < split1; i++)
	{
		CategoryBase* cat = (CategoryBase*) categories1->item[i];

		unsigned int noMch = 1;
		for(unsigned int j = 0; j < tempClassLst->nCount; j++)
			if (cat->label == (int) (tempClassLst->item[j])) noMch = 0;
		
		if (noMch) tempClassLst->AddItem((void*) cat->label);

		if (tempClassLst->nCount == numClasses) break;
	}

	for(unsigned int i = split2; i < categories2->Count(); i++)
	{			
		CategoryBase* cat = (CategoryBase*) categories2->item[i];

		unsigned int noMch = 1;
		for(unsigned int j = 0; j < tempClassLst->nCount; j++)
			if (cat->label == (int) (tempClassLst->item[j])) noMch = 0;
		
		if (noMch) tempClassLst->AddItem((void*) cat->label);

		if (tempClassLst->nCount == numClasses) break;
	}

	unsigned int ret = 0;
	if (tempClassLst->nCount == numClasses) ret = 1;
	delete tempClassLst;
	return ret;
}

//////////////////////////////////////////////////////////////////////
// GARTMAP::GetARTMAP
// 
// Returns a pointer to the i-th network
//
//////////////////////////////////////////////////////////////////////
ARTMAP* GARTMAP::GetARTMAP(unsigned int i)
{
	if (i < archive->Count())
		return (ARTMAP*) archive->item[i];
	else
		return 0;
}

//////////////////////////////////////////////////////////////////////
// GARTMAP::Store
// 
// Stores the archive network in comma-seprated files located in the
// specified path.
//
//////////////////////////////////////////////////////////////////////
int GARTMAP::Store(char* storagePath)
{
	if (!storagePath) return -1;
	if (!(*storagePath)) return -1;

	int ret = -1;
	char* filepath = (char*) malloc(strlen(storagePath) + 12);
	strcpy(filepath, storagePath);

	//Store resulting networks into files
	for(unsigned int i = 0; i < archive->Count(); i++)
	{		
		ARTMAP* nn = (ARTMAP*) archive->item[i];
		sprintf(filepath + strlen(storagePath), "\\%06d.csv", i);
		ret += nn->Store(filepath);		
	}

	free(filepath);

	return ret;
}

