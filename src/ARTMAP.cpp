//////////////////////////////////////////////////////////////////////
//
// This code was developed by Assem Kaylani (akaylani@gmail.com) 
// and Dr. Michael Georgiopoulos (michaelg@ucf.edu).
//
// Implementation of the ARTMAP class. 
// 
//////////////////////////////////////////////////////////////////////

#include "ARTMAP.h"


//////////////////////////////////////////////////////////////////////
// ARTMAP::ARTMAP
//
// Constructor
// 
//////////////////////////////////////////////////////////////////////
ARTMAP::ARTMAP(
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
	)		
{		
	stop = 0;
	currentVigilance = 0;
	learnFlag = 0;

	// Data properties
	numAttribs = _numAttribs;
	classAttrib = _classAttrib;		

	// Stoping parameters
	maxEpochs = _maxEpochs;
	minImproveErr = _minImproveErr;

	// Training options
	categoryType = _categoryType;
	initParam = _initParam;
	choiceParam = _choiceParam;
	baselineVigilance = _baselineVigilance;
	vigilanceSafety = _vigilanceSafety;
	maxCategories = _maxCategories;

	categories = new ListClass();
	classLst = 0;
	catTol = _catTol;	

	validationErrors = 0;
	validationPErr = 1.0;
	validationCt = 0;
	fitness = 0;
	paretoFront = 0;
}

//////////////////////////////////////////////////////////////////////
// ARTMAP::ARTMAP
//
// Copy Constructor
// 
//////////////////////////////////////////////////////////////////////
ARTMAP::ARTMAP(
	ARTMAP*			nn					// Pointer to a ART (FAM, EAM or GAM) network to be cloned. 
	)
{
	stop = nn->stop;
	currentVigilance = nn->currentVigilance;
	learnFlag = nn->learnFlag;

	// Data properties
	numAttribs = nn->numAttribs;
	classAttrib = nn->classAttrib;		

	// Stoping parameters
	maxEpochs = nn->maxEpochs;
	minImproveErr = nn->minImproveErr;

	// Training options
	categoryType = nn->categoryType;
	initParam = nn->initParam;
	choiceParam = nn->choiceParam;
	baselineVigilance = nn->baselineVigilance;
	vigilanceSafety = nn->vigilanceSafety;
	maxCategories = nn->maxCategories;

	categories = new ListClass();
	for(unsigned int i = 0; i < nn->categories->Count(); i++)
		CopyCategory((CategoryBase*) nn->categories->item[i]);

	classLst = nn->classLst;
	catTol = nn->catTol;	

	validationErrors = nn->validationErrors;
	validationPErr = nn->validationPErr;
	validationCt = nn->validationCt;
	fitness = nn->fitness;
	paretoFront = nn->paretoFront;
}

//////////////////////////////////////////////////////////////////////
// ARTMAP::~ARTMAP
//
// Destructor
// 
//////////////////////////////////////////////////////////////////////
ARTMAP::~ARTMAP()
{
	// free resources
	DeleteAllCategories();
}

//////////////////////////////////////////////////////////////////////
// ARTMAP::~DeleteAllCategories
//
// Delete all categories in this network and free resources
// 
//////////////////////////////////////////////////////////////////////
void ARTMAP::DeleteAllCategories()
{
	if (categories) 
	{
		for(unsigned int i = 0; i < categories->Count(); i++)
		{
			CategoryBase* cat = (CategoryBase*) categories->item[i];
			cat->freeResource();
			delete cat;
		}
		delete (categories);
	}
	categories = 0;
}

//////////////////////////////////////////////////////////////////////
// ARTMAP::AddCategory
//
// Add a category to the network using default values
// 
//////////////////////////////////////////////////////////////////////
CategoryBase* ARTMAP::AddCategory()
{
	if (maxCategories && (categories->nCount >= this->maxCategories)) 
	{
		stop = 1;
		return 0;
	}

	CategoryBase* cat;
	switch (categoryType)
	{
	case 0:
		cat = (CategoryBase*) new FAMCategory(numAttribs, choiceParam, initParam);
		break;
	case 1:
		cat = (CategoryBase*) new EAMCategory(numAttribs, choiceParam, initParam);
		break;
	case 2:
		cat = (CategoryBase*) new GAMCategory(numAttribs, choiceParam);
		break;
	}
		
	categories->Add((void*) cat);
	return cat;
}

//////////////////////////////////////////////////////////////////////
// ARTMAP::CopyCategory
//
// Add a category to the network that is a copy of another catogory
// 
//////////////////////////////////////////////////////////////////////
CategoryBase* ARTMAP::CopyCategory(CategoryBase* catSource)
{
	CategoryBase* cat;
	switch (catSource->getCategoryType())
	{
	case 0:
		cat = (CategoryBase*) new FAMCategory((FAMCategory*) catSource);
		break;
	case 1:
		cat = (CategoryBase*) new EAMCategory((EAMCategory*) catSource);
		break;
	case 2:
		cat = (CategoryBase*) new GAMCategory((GAMCategory*) catSource);
		break;
	}
	
	categories->Add((void*) cat);	
	return cat;
}

//////////////////////////////////////////////////////////////////////
// ARTMAP::Train
//
// Main training loop
// 
//////////////////////////////////////////////////////////////////////
int ARTMAP::Train(TableClass* inputData, unsigned int offset)
{
	try
	{	
		// add the first category
		AddCategory();

		// repeate for maxEpochs epochs
		for(unsigned int itr = 0; itr < maxEpochs && !stop; itr++)
		{
			//present all patterns to the network
			learnFlag = 0;
			for (int j = offset - 1; j >= 0; j--)
				for(unsigned int i = j; i < inputData->numRows && !stop; i += offset)
					presentPattern(inputData->rows[i]);

			if (learnFlag == 0) break;
		}

		// remove uncommittied nodes
		for (int j = categories->nCount - 1; j >= 0; j--)
		{
			CategoryBase* cat = (CategoryBase*) categories->GetNth(j);
			if (cat->n <= 0) 
			{
				cat->freeResource();
				categories->Remove(j);
				delete(cat);
			}
		}
		return 0;
	}
	catch (...)
	{
		return -1;
	}
}

//////////////////////////////////////////////////////////////////////
// ARTMAP::presentPattern
//
// Present one pattern to the network and take required action for 
// learning it.
//
//////////////////////////////////////////////////////////////////////
double ARTMAP::presentPattern(TableRow* row)
{	
	currentVigilance = baselineVigilance;	
	
	// Sort the categories acording to their CCF in DECENDING order 
	// using the standard insertion sorting algorithm
	unsigned int numCat = categories->Count();
	for(unsigned int i = 0; i < numCat; i++)
	{
		CategoryBase* cat = (CategoryBase*) categories->item[i];
		cat->presentPattern(row);
		double resp = cat->CCF_value;

		unsigned int j = 0;
		for (j = i; j > 0 && resp > ((CategoryBase*) categories->item[j - 1])->CCF_value; j--)
			categories->item[j] = categories->item[j - 1];

		categories->item[j] = cat;
	}

	// loop for all nodes starting with highest CCF, 
	// finding the node the will satisfy the vigilance requirement
	for(unsigned int i = 0; i < numCat; i++)
	{
		CategoryBase* catMax = (CategoryBase*) categories->item[i];

		// If uncommited node, then learn new pattern
		if (catMax->numPatternsLearned <= 0) 
		{		
			learnFlag += catMax->learnPattern(row);
			setLabel(catMax, row); 
			AddCategory();			
			break;
		}

		//check vigilance
		double vig = catMax->CMF_value;
		if (vig >= currentVigilance)
		{
			if (checkLabel(catMax, row))
			{
				// label matches
				learnFlag += catMax->learnPattern(row);				
				break;
			}
			else
			{
				if ((((double) catMax->toleratedPatternsCount + 1) / ((double) catMax->n)) < catTol)
				{
					learnFlag += catMax->learnPattern(row);	
					catMax->toleratedPatternsCount++;
					break;
				}
				// label does not match, raise the current vigilance to
				currentVigilance = vig + vigilanceSafety;
				
				// disqualify and select another node
			}
		}
		else
		{
			// disqualify and select another node
		}
	}

	return 0;
}

//////////////////////////////////////////////////////////////////////
// ARTMAP::evaluateStoppingCriteria
//
// return 1 if required to stop. More complex stopping criteria 
// can be implemented here in the future.
//////////////////////////////////////////////////////////////////////
int ARTMAP::evaluateStoppingCriteria()
{
	// return 1 if criteria not met
	// return 0 if criteria met. That is stop training.
	if (stop) return 0; else return 1;
} 

//////////////////////////////////////////////////////////////////////
// ARTMAP::setLabel
//
// Set the class label for a given category to match a given pattern
//////////////////////////////////////////////////////////////////////
void ARTMAP::setLabel(
	CategoryBase*	cat, 
	TableRow*		row
	)
{
	cat->label = classLst->FindFirst(row->fields[classAttrib]);
}

//////////////////////////////////////////////////////////////////////
// ARTMAP::checkLabel
//
// return 1 if the class label for a given category 
// matches a given pattern.
//////////////////////////////////////////////////////////////////////
unsigned int ARTMAP::checkLabel(
	CategoryBase*	cat, 
	TableRow*		row
	)
{
	if (compareVariables(classLst->item[cat->label], row->fields[classAttrib], classLst->type) == 0) return 1;
	return 0;
}

//////////////////////////////////////////////////////////////////////
// ARTMAP::Test
// 
// Predict the labels for a test dataset. Then compares the predicted
// labels with actual labels in the data. The function returns the 
// fraction of errors in prediction.
//
//////////////////////////////////////////////////////////////////////
int ARTMAP::Test(
	TableClass*		testData,			// Test Dataset
	unsigned int	maxTest,			// Max number of patterns to test. if 0, then test all
	unsigned int*	numTested,			// Actual number of patterns tested
	unsigned int	score,				// if other than 0, it will be assumed that test data includes the class (target) column. The predicted value will then be compared to the value in the last column in order to calculate network's predictive accuracy.	
	unsigned int	maxErr,				// maximum error (fraction of incorrectly predicted labels) after which this process is terminated,
	unsigned int	offset,				// starting point to test in the dataset, allowing the caller to randomly select a sequence to be used in evaluation. (usefull for some statistical approaches)
	char*			filename			// name of file used to store results. if NULL is passed, no results are stored.
	)
{
	validationErrors = 0;
	FILE* stream = 0;
	if (filename && *filename) if(filename[0]) stream  = fopen(filename, "w+" );

	unsigned int numCat = categories->Count();
	for(unsigned int i = 0; i < numCat; i++)
	{
		// Reset all categories
		CategoryBase* cat = (CategoryBase*) categories->item[i];
		cat->active = 0;
		cat->numPatternsLearned = 0;
	}

	unsigned int testCt = 0;
	// Loop for all test patterns
	for(int nn = offset - 1; nn >= 0; nn--)
	for(unsigned int n = nn; n < testData->numRows; n += offset)
	{
		if (maxErr && (validationErrors >= maxErr)) break;
		if (maxTest && (testCt >= maxTest)) break;
		testCt++;

		TableRow* row = testData->rows[n];
		
		// find category with max CCF and CMF >= basline vigilance
		CategoryBase* catMax = 0;
		double maxResp = -1.0 * testData->numColumns;		
		for(unsigned int i = 0; i < numCat; i++)
		{
			CategoryBase* cat = (CategoryBase*) categories->item[i];
			cat->presentPattern(row);				

			if ((cat->CCF_value > maxResp) && (cat->CMF_value >= baselineVigilance))
			{
				catMax = cat;
				maxResp = catMax->CCF_value;
			}
		}
		
		if (!catMax) 
		{
			validationErrors++;
			continue;
		}
			
		catMax->active++;

		if (score) 
		{	
			if (!checkLabel(catMax, row))
				validationErrors++;
			else
				catMax->numPatternsLearned++;
		}

		if (stream) 
		{
			void* lbl = classLst->item[catMax->label];
			if (catMax) 
			{
				if (classLst->type == DUTIL_DOUBLE)
					fprintf(stream, "%f\n", *((double*) lbl));
				else if (classLst->type == DUTIL_CHAR)
					fprintf(stream, "%f\n", *((char*) lbl));
				else if (classLst->type == DUTIL_LONG)
					fprintf(stream, "%d\n", *((int*) lbl));
			}
			else
				fprintf(stream, "?\n");
		}
	}

	if (stream) fclose( stream );
	
	if (numTested) *numTested = testCt;
	validationCt = testCt;
	if (testCt) validationPErr = (double) validationErrors / (double) testCt;
	return validationErrors;
}

//////////////////////////////////////////////////////////////////////
// ARTMAP::Store
//
// Store the network as a list of categories in a comma-separated 
// text file format
//////////////////////////////////////////////////////////////////////
int ARTMAP::Store(char* fileName)
{
	if (!fileName) return -1;

	FILE* stream = fopen(fileName, "w" );
	if (!stream) return -1;

	unsigned int numNetParams = (2 * numAttribs);

	fprintf(stream, "Type,");

	// header
	switch(this->categoryType)
	{
	case 0:
		{
			for(unsigned int j = 0; j < numAttribs; j++)
				fprintf(stream, "Wi%d,", j);
			for(unsigned int j = 0; j < numAttribs; j++)
				fprintf(stream, "Wj%d,", j);
			
			numNetParams = (2 * numAttribs) + 1;
		}
		break;
	case 1:
		{
			for(unsigned int j = 0; j < numAttribs; j++)
				fprintf(stream, "m%d,", j);

			for(unsigned int j = 0; j < numAttribs; j++)
				fprintf(stream, "d%d,", j);

			fprintf(stream, "R,");
			fprintf(stream, "mu,");
			fprintf(stream, "alpha,");
			fprintf(stream, "D,");

			numNetParams = (2 * numAttribs) + 5;
		}
		break;
	case 2:
		{	
			for(unsigned int j = 0; j < numAttribs; j++)
				fprintf(stream, "mu%d,", j);

			for(unsigned int j = 0; j < numAttribs; j++)
				fprintf(stream, "s%d,", j);

			for(unsigned int j = 0; j < numAttribs; j++)
				fprintf(stream, "v%d,", j);

			fprintf(stream, "gamma,");
			fprintf(stream, "sigmaProd,");

			numNetParams = (3 * numAttribs) + 3;
		}
		break;
	}

	fprintf(stream, "CF, Class\n");
	// end header
	double* catf = (double*) malloc(sizeof(double) * numNetParams);

	// write categories, each in a line, values seprated by commas
	for(unsigned int i = 0; i < categories->Count(); i++)
	{
		CategoryBase* cat = (CategoryBase*) categories->item[i];
				
		fprintf(stream, "%d,", cat->getCategoryType());
		
		cat->getDescription(catf);
		for(unsigned int j = 0; j < numNetParams; j++)
			fprintf(stream, "%f,", catf[j]);

		void* lbl = classLst->item[cat->label];
		if (classLst->type == DUTIL_DOUBLE)
			fprintf(stream, "%f\n", *((double*) lbl));
		else if (classLst->type == DUTIL_CHAR)
			fprintf(stream, "%f\n", *((char*) lbl));
		else if (classLst->type == DUTIL_LONG)
			fprintf(stream, "%d\n", *((int*) lbl));			
	}

	free(catf);
	fclose(stream);

	return 0;
}
