//////////////////////////////////////////////////////////////////////
// 
// This code was developed by Assem Kaylani (akaylani@gmail.com) 
// and Dr. Michael Georgiopoulos (michaelg@ucf.edu).
// 
// For more information please direct commmunication to:
// Dr. Michael Georgiopoulos
// Department of EECS
// University of Central Florida
// 4000 Central Florida Boulevard
// Harris Engineering Center, 358
// Orlando, FL 32816
// michaelg@ucf.edu
//
// Disclaimer: 
// ==========
// FREE	TO MODIFY AND USE
// Refereces mentioned below are to be cited as appropriate.
// 
// SOFTWARE CODE PROVIDED AS IS
// Authors do not warrant that the software will be error free or 
// operate without interruption. Authors are not liable for any 
// damages arising out of the use of or inability to use this 
// code.
// 
// Algorithms implemented: 
// ======================
//
// >> MO-GART: see Kaylani et. al.
// [1] A. Kaylani, M. Georgiopoulos, M. Mollaghasemi, 
// G. C. Anagnostopoulos, C. Sentelle, M. Zhong, "An Adaptive 
// Multi-Objective Approach to Evolving ART Architectures," 
// IEEE Transactions on Neural Networks, Vol. 21, 
// Issue 4, pp. 529-550, 2010.
//
// >> Combined multiobjective Genetic ART: see
// [2]
//
// >> Ellipsoidal ARTMAP: see Anagnastopolous, et al
// [3] G. Anagnostopoulos, "Novel approaches in adaptive resonance theory 
// for machine learning," Ph.D. dissertation, University of Central
// Florida, Orlando, May 2001.
//
// >> Semi supervised FAM, EAM, GAM: see Anagnastoplos et al
// [4] G. C. Anagnostopoulos, M. Bharadwaj, M. Georgiopoulos, S. J. Verzi, 
// and G. L. Heileman, “Exemplar-based pattern recognition via
// semi-supervised learning,” in Proceedings of the 2003 International 
// Joint Conference on Neural Networks (IJCNN ’03), vol. 4, Portland,
// Oregon, USA, 2003, pp. 2782–2787.
//
// >> Cross Validation in FAM: see Kufaku et al
// [6] A. Koufakou, M. Georgiopoulos, G. Anagnostopoulos, and T. Kasparis, 
// "Cross-validation in Fuzzy ARTMAP for large databases," Neural
// Networks, vol. 14, pp. 1279–1291, 2001.
//
// Other algorithms:
// ================
//
// >> Fuzzy ARTMAP: see Carpenter et al
// [7] G. A. Carpenter, S. Grossberg, N. Markuzon, J. H. Reynolds, and D. B. 
// Rosen, “Fuzzy ARTMAP: A neural network architecture for incremental 
// supervised learning of analog multidimensional maps,” IEEE Transactions 
// on Neural Networks, vol. 3, pp. 698–713, 1992.
//
// >> Gaussain ARTMAP: see Williamson
// [8] J. R. Williamson, “Gaussian ARTMAP: A neural network for fast 
// incremental learning of noisy multidimensional maps,” Neural Networks,
// vol. 9, no. 5, pp. 881–897, 1996.
//
//////////////////////////////////////////////////////////////////////

#include "ARTMAP.h"
#include "GARTMAP.h"

#define EXPT_INT extern "C" __declspec(dllexport) unsigned int 
#define EXPT_DBL extern "C" __declspec(dllexport) double

//////////////////////////////////////////////////////////////////////
// GARTMAP_Train
//
// Trains a set of networks, based on input data that is (split into 
// a training and a validation set). Training is using MO-GART 
// algorithm descried in [1].
// 
//////////////////////////////////////////////////////////////////////
EXPT_INT GARTMAP_Train(
    // Training and validation data
	char*			trainDataConStr,	// ODBC connection string for the database containing the training data used to training the initial population.
	char*			trainDataSqlStr,	// SQL statement selecting the training data (input patterns) from the database
	char*			validDataConStr,	// ODBC connection string for the database containing the validation data used to evaluate network accuracy during genetic evolution.
	char*			validDataSqlStr,	// SQL statement selecting the validation data from the database
	unsigned int	targetAttrib,		// The Zero-based index of the column that contains the class label for the training and validation data retrieved using the above SQL statements

	// ART Options					
	unsigned int	categoryType,		// 0 FAM, 1 EAM, 2 GAM
	double			initParam,			// For FAM, this is initial wij, for GAM this is gamma		
	double			choiceParam,		// Choice parameter used in FAM and EAM
	double			catTol,				// Used in ART training. Tolerance of categories to encode wrong patterns. If other than zero, this would implement semi-supervised learning (see literature [4]).
	double			vigilanceSafety,	// This value is used to increase the vigilance during the training of ART networks. This is needed when a category is matching a pattern of incorrect label. See the FAM algorithm	literature for more information.			
	unsigned int	maxCategories,		// The maximum categories (nodes) the ART network is allowed to create, after which training will stop.
	unsigned int	maxEpochs,			// Number of epochs used in traing initial population of ART networks
	double			minImproveErr,		// Not used. The percentage (represented as decimal, eg. 0.05 for 5%) that improvement needs to be observed from one epoch to the next. if no improvement is observed for a number of epochs, the ART stops training possibly before reaching maxEpochs to avoid over training. This mechanism is not used from GARTMAP and therefore this paramater is ignored.

	// GA Options
	unsigned int	maxGen,				// Maximum number of generations
	unsigned int	popSize,			// Population size of the GA
	double			vigMin,				// Minumum value of baseline vigilance used to train initation population
	double			vigMax,				// Maximum value of baseline vigilance. Values are varied in popSize equal increments between vigMin and vigMax.
	double			pPrune,				// Probably of applying the prune operator on a given network
	double			pMutate,			// Probably of applying the mutate operator on a given network
	unsigned int	selectionSetSize,	// Selection set size in tournament selection when determining crossover candidates

	// Storage
	char*			storagePath,		// Path where comma-separated files representing the pareto archive ART networks are stored. 

	// Output
	double*			telapsed,			// Returns the total time required for MO-GART training (Total time to train ARTs and evolve them)
	unsigned int*	actualGens			// Actual number of generations produced before the algorithm stopped. 
	)
{
	if (!trainDataConStr || !trainDataSqlStr) return 0;
	if (!validDataConStr || !validDataSqlStr) return 0;

	// Read training dataset into memory
	ODBCTableClass* inputData = new ODBCTableClass(trainDataConStr, trainDataSqlStr);
	if (!inputData->numRows) 
	{
		if (inputData) delete inputData;
		return -1;
	}

	// Read validation dataset into memory
	ODBCTableClass* validationData = new ODBCTableClass(validDataConStr, validDataSqlStr);
	if (!validationData->numRows) 
	{
		if (inputData) delete inputData;
		if (validationData) delete validationData;
		return -1;
	}

	//The srand function sets the starting point for generating a 
	//series of pseudorandom integers. To reinitialize the generator 
	//to the same sequence, 
	//use 1 as the seed argument. Any other value for seed sets 
	//the generator to a random starting point. rand retrieves 
	//the pseudorandom numbers that are generated. 
	//Calling rand before any call to srand generates 
	//the same sequence as calling srand with seed passed as 1.
#if _DEBUG
	srand(1);
	//srand( (unsigned)time( NULL ) );
#else
	srand( (unsigned)time( NULL ) );
#endif

	// initialize object
	GARTMAP* gartObj = new GARTMAP(	
							targetAttrib,							

							// ART Options					
							categoryType,
							initParam,
							choiceParam,
							catTol,
							vigilanceSafety,					
							maxCategories,
							maxEpochs,
							minImproveErr,

							// GA Options
							maxGen,
							popSize,
							vigMin,
							vigMax,														
							pPrune,
							pMutate,
							selectionSetSize);


	clock_t start = clock(); // mark time at start
	// train using MO-GART algorithm
	*actualGens = gartObj->Train(inputData, validationData);
	clock_t finish = clock(); // mark time at end
	*telapsed = (double) (finish - start) / CLOCKS_PER_SEC;

	// free resources
	delete inputData;
	delete validationData;

	// store networks produce into files
	gartObj->Store(storagePath);

	// return the object pointer
	return (unsigned int) gartObj;
}

//////////////////////////////////////////////////////////////////////
// CGARTMAP_Train
//
// Trains a set of networks, based on input data that is (split into 
// a training and a validation set). Training is using CMO-GART 
// algorithm descried in [2]. This algorithm trains 3 MO-GART (a 
// MO-GFAM, a MO-GEAM and a MO-GGAM) and combines the pareto solutions
// from the 3 algorthm to provide the best performers of global 
// archive of networks that could be of any type (FAM, EAM or GAM).
// 
//////////////////////////////////////////////////////////////////////
EXPT_INT CGARTMAP_Train(
    // Training and validation data
	char*			trainDataConStr,	// ODBC connection string for the database containing the training data used to training the initial population.
	char*			trainDataSqlStr,	// SQL statement selecting the training data (input patterns) from the database
	char*			validDataConStr,	// ODBC connection string for the database containing the validation data used to evaluate network accuracy during genetic evolution.
	char*			validDataSqlStr,	// SQL statement selecting the validation data from the database
	unsigned int	targetAttrib,		// The Zero-based index of the column that contains the class label for the training and validation data retrieved using the above SQL statements

	// ART Options					
	unsigned int	categoryType,		// Not used. 0 FAM, 1 EAM, 2 GAM
	double			initParam,			// For FAM, this is initial wij, for GAM this is gamma		
	double			choiceParam,		// Choice parameter used in FAM and EAM
	double			catTol,				// Used in ART training. Tolerance of categories to encode wrong patterns. If other than zero, this would implement semi-supervised learning (see literature).
	double			vigilanceSafety,	// This value is used to increase the vigilance during the training of ART networks. This is needed when a category is matching a pattern of incorrect label. See the FAM algorithm	literature for more information.			
	unsigned int	maxCategories,		// The maximum categories (nodes) the ART network is allowed to create, after which training will stop.
	unsigned int	maxEpochs,			// Number of epochs used in traing initial population of ART networks
	double			minImproveErr,		// Not used. The percentage (represented as decimal, eg. 0.05 for 5%) that improvement needs to be observed from one epoch to the next. if no improvement is observed for a number of epochs, the ART stops training possibly before reaching maxEpochs to avoid over training. This mechanism is not used from GARTMAP and therefore this paramater is ignored.

	// GA Options
	unsigned int	maxGen,				// Maximum number of generations
	unsigned int	popSize,			// Population size of the GA
	double			vigMin,				// Minumum value of baseline vigilance used to train initation population
	double			vigMax,				// Maximum value of baseline vigilance. Values are varied in popSize equal increments between vigMin and vigMax.
	double			pPrune,				// Probably of applying the prune operator on a given network
	double			pMutate,			// Probably of applying the mutate operator on a given network
	unsigned int	selectionSetSize,	// Selection set size in tournament selection

	// Storage
	char*			storagePath,		// Path where comma-separated files representing the pareto archive ART networks are stored. 

	// Output
	double*			telapsed,			// Returns the total time required for MO-GART training (Total time to train ARTs and evolve them)
	unsigned int*	actualGens			// Actual number of generations produced before the algorithm stopped. 
	)
{
	if (!trainDataConStr || !trainDataSqlStr) return 0;
	if (!validDataConStr || !validDataSqlStr) return 0;

	ODBCTableClass* inputData = new ODBCTableClass(trainDataConStr, trainDataSqlStr);
	if (!inputData->numRows) 
	{
		if (inputData) delete inputData;
		return -1;
	}

	ODBCTableClass* validationData = new ODBCTableClass(validDataConStr, validDataSqlStr);
	if (!validationData->numRows) 
	{
		if (inputData) delete inputData;
		if (validationData) delete validationData;
		return -1;
	}

	//The srand function sets the starting point for generating a 
	//series of pseudorandom integers. To reinitialize the generator 
	//to the same sequence, 
	//use 1 as the seed argument. Any other value for seed sets 
	//the generator to a random starting point. rand retrieves 
	//the pseudorandom numbers that are generated. 
	//Calling rand before any call to srand generates 
	//the same sequence as calling srand with seed passed as 1.
#if _DEBUG
	srand(1);
	//srand( (unsigned)time( NULL ) );
#else
	srand( (unsigned)time( NULL ) );
#endif

	GARTMAP* mogfam = new GARTMAP(	
							targetAttrib,							

							// ART Options					
							0,	//FAM category type
							initParam,
							choiceParam,
							catTol,
							vigilanceSafety,					
							maxCategories,
							maxEpochs,
							minImproveErr,

							// GA Options
							maxGen,
							popSize,
							vigMin,
							vigMax,														
							pPrune,
							pMutate,
							selectionSetSize);

	clock_t start = clock();
	// train with MO-GFAM algorithm
	*actualGens = mogfam->Train(inputData, validationData);
	clock_t finish = clock();
	*telapsed = (double) (finish - start) / CLOCKS_PER_SEC;
	
	// free resources
	delete inputData;
	delete validationData;	

	inputData = new ODBCTableClass(trainDataConStr, trainDataSqlStr);
	if (!inputData->numRows) 
	{
		if (inputData) delete inputData;
		return -1;
	}

	validationData = new ODBCTableClass(validDataConStr, validDataSqlStr);
	if (!validationData->numRows) 
	{
		if (inputData) delete inputData;
		if (validationData) delete validationData;
		return -1;
	}

	GARTMAP* mogeam = new GARTMAP(	
							targetAttrib,							

							// ART Options					
							1, //EAM category type
							initParam,
							0.001,				// choiceParam,
							catTol,
							vigilanceSafety,					
							maxCategories,
							maxEpochs,
							minImproveErr,

							// GA Options
							maxGen,
							popSize,
							vigMin,
							vigMax,														
							pPrune,
							pMutate,
							selectionSetSize);

	start = clock();
	// train with MO-GEAM algorithm
	*actualGens += mogeam->Train(inputData, validationData);
	finish = clock();
	*telapsed += (double) (finish - start) / CLOCKS_PER_SEC;
	
	// free resources
	delete inputData;
	delete validationData;	

	inputData = new ODBCTableClass(trainDataConStr, trainDataSqlStr);
	if (!inputData->numRows) 
	{
		if (inputData) delete inputData;
		return -1;
	}

	validationData = new ODBCTableClass(validDataConStr, validDataSqlStr);
	if (!validationData->numRows) 
	{
		if (inputData) delete inputData;
		if (validationData) delete validationData;
		return -1;
	}

	GARTMAP* moggam = new GARTMAP(	
							targetAttrib,							

							// ART Options					
							2, // GAM category type
							initParam,
							0.6,			// choiceParam,
							catTol,
							vigilanceSafety,					
							maxCategories,
							maxEpochs,
							minImproveErr,

							// GA Options
							maxGen,
							popSize,
							0.0,	// vigMin,
							0.5,	// vigMax,														
							pPrune,
							pMutate,
							selectionSetSize);

	start = clock();
	// train with MO-GGAM algorithm
	*actualGens += moggam->Train(inputData, validationData);
	finish = clock();
	*telapsed += (double) (finish - start) / CLOCKS_PER_SEC;

	// free resources
	delete inputData;
	delete validationData;	

	// add pareto archive from MO-GEAM into MO-GFAM
	unsigned int i = 0;
	while(ARTMAP* nn = mogeam->GetARTMAP(i++))	
	{
		// for each solution nn in MO-GEAM

		for(int j = mogfam->archive->Count() - 1; j >= 0; j--)
		{
			ARTMAP* nn1 = (ARTMAP*) mogfam->archive->GetNth(j);

			nn->classLst = nn1->classLst;

			// Compare nn to every solution in MO-GFAM archive
			int cres = mogfam->ParetoCompare(nn, nn1);

			if ((cres == 0) || (cres == 2))			
			{
				// mark nn as non-pareto
				nn->paretoFront = 0;
				break;
			}

			if (cres == 1)			
			{
				// nn dominates. remove deminated solution nn1 from MO-GFAM
				mogfam->archive->Remove(j);
				delete nn1;
			}
		}

		if (nn->paretoFront)
		{
			// copy nn to MO-GFAM archive
			ARTMAP* cpynn = new ARTMAP(nn);
			mogfam->archive->Add(cpynn);
			cpynn->categories->Count();
		}
	}
	// delete MO-GEAM object. It is not needed anymore.
	delete mogeam;

	// combine pareto archive from MO-GGAM into MO-GFAM
	i = 0;
	while(ARTMAP* nn = moggam->GetARTMAP(i++))	
	{
		for(int j = mogfam->archive->Count() - 1; j >= 0; j--)
		{
			ARTMAP* nn1 = (ARTMAP*) mogfam->archive->GetNth(j);

			nn->classLst = nn1->classLst;

			int cres = mogfam->ParetoCompare(nn, nn1);

			if ((cres == 0) || (cres == 2))			
			{
				nn->paretoFront = 0;
				break;
			}

			if (cres == 1)			
			{
				mogfam->archive->Remove(j);
				delete nn1;
			}
		}

		if (nn->paretoFront)
		{
			// copy nn to archive
			ARTMAP* cpynn = new ARTMAP(nn);
			mogfam->archive->Add(cpynn);
			cpynn->categories->Count();
		}
	}
	// delete MO-GGAM object
	delete moggam;

	// calling count() improves performance
	mogfam->archive->Count();

	// store resulting ART networks into files in the storage path.
	mogfam->Store(storagePath);

	// return point to object
	return (unsigned int) mogfam;
}

//////////////////////////////////////////////////////////////////////
// GARTMAP_Delete
//
// This must be called to free memory associated with the pointer 
// returned by calling GARTMAP_Train or CGARTMAP_Train. Failing to 
// call this function will cause a memory leak.
// 
//////////////////////////////////////////////////////////////////////
EXPT_INT GARTMAP_Delete(
	GARTMAP*		nn					// Pointer to GARTMAP objected (returned by calling GARTMAP_Train or CGARTMAP_Train)
	)
{
	// free resources
	if (nn) delete nn;
	return 0;
}

//////////////////////////////////////////////////////////////////////
// GARTMAP_GetARTMAP
//
// Returns a pointer to the (pareto) archive network with index i.
// This function will return with nn = NULL if i is out of bound.
// Note: ARTMAP_Delete should NOT be called on returned pointer.
// 
//////////////////////////////////////////////////////////////////////
EXPT_INT GARTMAP_GetARTMAP(
	GARTMAP*		nn,					// pointer for an ART network returned
	int				i					// index of the ART network in the archive
	)
{
	if (nn) return (unsigned int) nn->GetARTMAP(i);
	return 0;
}

//////////////////////////////////////////////////////////////////////
// ARTMAP_Test
// 
// Predict the labels for a test dataset. Then compares the predicted
// labels with actual labels in the data. The function returns the 
// fraction of errors in prediction.
//
//////////////////////////////////////////////////////////////////////
EXPT_DBL ARTMAP_Test(
	ARTMAP*	nn,							// Pointer to an ART network returned by calling GARTMAP_GetARTMAP or ARTMAP_TRAIN
	char*			testDataConStr,		// ODBC connection string for the database containing the test data.
	char*			testDataSqlStr,		// SQL statement selecting the test data patterns from the database.
	char*			filename,			// name of file used to store results. if NULL is passed, no results are stored.
	int				score,				// if other than 0, it will be assumed that test data includes the class (target) column. The predicted value will then be compared to the value in the last column in order to calculate network's predictive accuracy.	
	double*			telapsed			// Total time elapsed during testing of dataset. 
	)
{
	double ret = -1.0;
	unsigned int numTestRows;
	if (!nn) return -1;

	if (!testDataConStr || !testDataSqlStr) return 0;
	ODBCTableClass* testData = new ODBCTableClass(testDataConStr, testDataSqlStr);
	
	if (testData) 
	{
		numTestRows = testData->numRows;
		if (numTestRows) 
		{
			clock_t start = clock();
			ret = nn->Test(testData, 0, 0, score, 0, 1, filename);
			clock_t finish = clock();
			*telapsed = (double)(finish - start) / CLOCKS_PER_SEC;
		}
	
		delete testData;
	}

	return ret/numTestRows;
}

//////////////////////////////////////////////////////////////////////
// ARTMAP_NumCategories
// 
// Returns the number of categories of a given ART network
//
//////////////////////////////////////////////////////////////////////
EXPT_INT ARTMAP_NumCategories(
	ARTMAP*			nn					// Pointer to an ART network returned by calling GARTMAP_GetARTMAP or ARTMAP_TRAIN
	)
{
	if (nn) return nn->categories->Count();
	return 0;
}

//////////////////////////////////////////////////////////////////////
// ARTMAP_GetCategoryDescription
// 
// Returns information about an ART network category
//
//////////////////////////////////////////////////////////////////////
EXPT_INT ARTMAP_GetCategoryDescription(
	ARTMAP*			nn,					// Pointer to an ART network returned by calling GARTMAP_GetARTMAP or ARTMAP_TRAIN
	unsigned int	cati,				// Index of category within network
	double*			retf,				// Output: an array of doubles representing the category
	unsigned int*	lbl,				// Output: Class label associated with category 
	unsigned int*	type				// Output: Category type: 0 FAM, 1 EAM, 2 GAM
	)
{
	if ((cati < 0) || (cati >= nn->categories->Count())) return -1;
	CategoryBase* cat = (CategoryBase*) nn->categories->item[cati];
	*lbl = cat->label;
	cat->getDescription(retf);
	*type = cat->getCategoryType();
	return 0;
}

//////////////////////////////////////////////////////////////////////
// ARTMAP_SetCategoryDescription
// 
// This function must be called to free resources associated with 
// pointer returned by calling ARTMAP_Train.
//
//////////////////////////////////////////////////////////////////////
EXPT_INT ARTMAP_SetCategoryDescription(
	ARTMAP*			nn,					// Pointer to an ART network returned by calling GARTMAP_GetARTMAP or ARTMAP_TRAIN
	unsigned int	cati,				// Index of category within network
	double*			retf				// an array of doubles representing the category
	)
{
	if ((cati < 0) || (cati >= nn->categories->Count())) return -1;
	CategoryBase* cat = (CategoryBase*) nn->categories->item[cati];
	cat->setDescription(retf);
	return 0;
}

//////////////////////////////////////////////////////////////////////
// ARTMAP_DeleteCategory
// 
// Deletes a category from an ART network
//
//////////////////////////////////////////////////////////////////////
EXPT_INT ARTMAP_DeleteCategory(
	ARTMAP*			nn,					// Pointer to an ART network returned by calling GARTMAP_GetARTMAP or ARTMAP_TRAIN
	unsigned int	cati				// Index of category within network
	)
{
	if ((cati < 0) || (cati >= nn->categories->Count())) return -1;
	CategoryBase* cat = (CategoryBase*) nn->categories->GetNth(cati);	
	nn->categories->Remove(cati);
	cat->freeResource();
	delete cat;
	return nn->categories->Count();
}

//////////////////////////////////////////////////////////////////////
// ARTMAP_Train
//
// Trains an ART network, based on input data (that MIGHT be split into 
// a training and a validation set). Training could be using different 
// algorithms. 
//
// categoryType parameter allows choosing Fuzzy ARTMAP 
// (FAM) (see [7]), Ellipsoidal ARTMAP EAM (see [3]) or Gaussian 
// ARTMAP (GAM) (see [8]). 
//
// A non-zero value in catTol parameter cause algorithm to use 
// semi supervised learning concept outlined in [4].
//
// Passing a validation set is optional. If passed it will be used
// to stop learning when generalization performance start to decline.
// or does not improve by at least minImproveErr. 
// This concept is outlined in [6]
//
//////////////////////////////////////////////////////////////////////
EXPT_INT ARTMAP_Train(				  
	// Training and validation data
	char*			trainDataConStr,	// ODBC connection string for the database containing the training data used to training the initial population.
	char*			trainDataSqlStr,	// SQL statement selecting the training data (input patterns) from the database
	char*			validDataConStr,	// ODBC connection string for the database containing the validation data used to evaluate network accuracy during genetic evolution.
	char*			validDataSqlStr,	// SQL statement selecting the validation data from the database
	unsigned int	targetAttrib,		// The Zero-based index of the column that contains the class label for the training and validation data retrieved using the above SQL statements

	// ART Options			
	unsigned int	categoryType,		// 0 FAM, 1 EAM, 2 GAM
	double			initParam,			// For FAM, this is initial wij, for GAM this is gamma		
	double			choiceParam,		// Choice parameter used in FAM and EAM
	double			baselineVigilance,	// baseline vigilance 
	double			catTol,				// Tolerance of categories to encode wrong patterns. If other than zero, this would implement semi-supervised learning (see literature [4]).
	double			vigilanceSafety,	// This value is used to increase the vigilance during the training of ART networks. This is needed when a category is matching a pattern of incorrect label. See the FAM algorithm	literature for more information.			
	unsigned int	maxCategories,		// The maximum categories (nodes) the ART network is allowed to create, after which training will stop.
	unsigned int	maxEpochs,			// Number of epochs used in traing 
	double			minImproveErr,		// The percentage (represented as decimal, eg. 0.05 for 5%) that improvement needs to be observed from one epoch to the next. if no improvement is observed for a number of epochs, the ART stops training possibly before reaching maxEpochs to avoid over training. This mechanism is not used from GARTMAP and therefore this paramater is ignored.

	// Storage
	char*			fileName,			// Name of comma-separated file representing the ART network to be written after training. 
	
	double*			telapsed			// Returns the total time required for ART training
	)
{
	ARTMAP* nn = NULL;
	if (!trainDataConStr || !trainDataSqlStr) return 0;

	// load input dataset into memory
	ODBCTableClass* inputData = new ODBCTableClass(trainDataConStr, trainDataSqlStr);
	if (!inputData->numRows) 
	{
		if (inputData) delete inputData;
		return -1;
	}	

	inputData->MakeRowsArrayRandomized();

	// Try to load validation dataset into memory
	ODBCTableClass* validData = 0;
	if (validDataConStr && validDataSqlStr) 
		if (validDataConStr[0] && validDataSqlStr[0])
			validData = new ODBCTableClass(trainDataConStr, trainDataSqlStr);

	nn = new ARTMAP(
			targetAttrib,
			targetAttrib,			
			categoryType,
			initParam,
			choiceParam,
			baselineVigilance,
			vigilanceSafety,
			catTol,
			maxCategories,
			maxEpochs,
			minImproveErr);

	nn->classLst = inputData->GetAttribSpaceCopy(targetAttrib);

	clock_t start = clock();
	if (validData && validData->numRows)
	{
		// if cross validation to be used, the train for one epoch at time
		// and evaluate performance on validation set. 

		nn->maxEpochs = 1;

		double lastErr = 0.0;
		unsigned int numBad = 0;
		for(unsigned int itr = 0; itr < maxEpochs; itr++)
		{
			nn->Train(inputData, 1);

			// Get performance on validData	
			double err = nn->Test(validData, 0, 0, 1, 0, 1, 0);
		
			if (itr && ((err + err * minImproveErr) >= lastErr))  // cout unimprove
				numBad++;
			else // if performance improved, keep going, reset unimprove counter
				numBad = 0;

            if (numBad >= 10) break; // cout unimprove = 10 epochs without improvement then stop
						
			lastErr = err;									
		}

		// free resources
		delete (validData);
	}
	else
	{	
		// If no cross validation is required, just train for maxEpochs epochs
		nn->Train(inputData, 1);
	}
	clock_t finish = clock();

	*telapsed = (double)(finish - start) / CLOCKS_PER_SEC;

	// Store ART network into a file
	nn->Store(fileName);		
	
	// free resources
	delete inputData;
	
	return (unsigned int) nn;
}

//////////////////////////////////////////////////////////////////////
// ARTMAP_Delete
// 
// This function must be called to free resources associated with 
// pointer returned by calling ARTMAP_Train.
//
//////////////////////////////////////////////////////////////////////
EXPT_INT ARTMAP_Delete(
	ARTMAP*			nn					// Pointer to an ART network returned by calling ARTMAP_Train
	)
{
	if (nn->classLst) delete nn->classLst;
	if (nn) delete nn;
	return 0;
}